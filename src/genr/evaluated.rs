use derive_more::Display;
use eyre::{Result, eyre};
use rand::seq::IndexedRandom;

use crate::eval::{Evaluator, State};
use crate::evolve::cfg::{
    Crossover, Duplicates, EvolveCfg, Mutation, Replacement, Selection, Survival,
};
use crate::evolve::evolver::RandState;
use crate::genr::member::Member;
use crate::genr::species::SpeciesId;
use crate::genr::unevaluated::UnevaluatedGenr;
use crate::ops::mutation::{mutate_lognorm, mutate_normal, mutate_rate};
use crate::ops::sampling::{multi_rws, rws, sus};

#[must_use]
#[derive(Display, Clone, PartialOrd, PartialEq)]
#[display("pop: {:>5}, best: {:5.5}", mems.len(), self.mems[0])]
pub struct EvaluatedGen<S: State> {
    pub mems: Vec<Member<S>>,
}

impl<S: State> EvaluatedGen<S> {
    pub fn new(mut mems: Vec<Member<S>>) -> Self {
        // Sort by base fitness. Selection should happen using selection
        // fitness. Generate survivors using base fitness, to make sure we keep
        // the top individuals.
        mems.sort_unstable_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        Self { mems }
    }

    pub fn mems(&self) -> &[Member<S>] {
        &self.mems
    }

    #[must_use]
    pub fn species_mems(&self, n: SpeciesId) -> Vec<Member<S>> {
        self.mems.iter().filter(|v| v.species == n).cloned().collect()
    }

    // Get list of species.
    #[must_use]
    pub fn species(&self) -> Vec<SpeciesId> {
        // Relies on species index assignment to be contigous from zero.
        let mut species: Vec<_> = self.mems.iter().map(|mem| mem.species).collect();
        species.sort_unstable();
        species.dedup();
        species
    }

    fn survivors(&self, survival: Survival, cfg: &EvolveCfg) -> Vec<Member<S>> {
        let mut mems = match survival {
            Survival::TopProportion(prop) => {
                // Ceiling so we don't miss keeping things for small sizes.
                // Use the target population size rather than the size of the
                // current generation so small generations don't have a smaller
                // number of survivors selected from them. This is useful for
                // with a small number of individuals.
                let num = (cfg.pop_size as f64 * prop).ceil() as usize;
                self.mems.iter().take(num).cloned().collect()
            }
            Survival::SpeciesTopProportion(prop) => {
                let mut survivors = Vec::new();
                let species = self.species();
                let num = (cfg.pop_size as f64 * prop / species.len() as f64).ceil() as usize;
                for id in species {
                    survivors.extend(self.species_mems(id).into_iter().take(num));
                }
                survivors
            }
            Survival::Youngest => {
                let mut survivors = self.mems.clone();
                survivors.sort_unstable_by_key(|mem| mem.age);
                // Drop oldest until we reach the population size.
                survivors.truncate(cfg.pop_size);
                survivors
            }
            Survival::Tournament(q) => {
                let mut survivors = Vec::new();
                let mut rng = rand::rng();
                for mem in &self.mems {
                    let opponents = self.mems.choose_multiple(&mut rng, q);
                    let wins = opponents.filter(|opp| opp.fitness > mem.fitness).count();
                    survivors.push((wins, mem));
                }
                survivors.sort_unstable_by_key(|(wins, _)| -(*wins as i64));
                survivors.into_iter().map(|(_, mem)| mem.clone()).collect()
            }
        };
        // Bump ages.
        for mem in &mut mems {
            mem.age += 1;
        }
        mems
    }

    fn selection(&self, selection: Selection) -> [Member<S>; 2] {
        let fitnesses = self.mems.iter().map(|v| v.selection_fitness).collect::<Vec<_>>();
        let idxs = match selection {
            Selection::Sus => sus(&fitnesses, 2),
            Selection::Roulette => multi_rws(&fitnesses, 2),
        };
        [self.mems[idxs[0]].clone(), self.mems[idxs[1]].clone()]
    }

    fn check_weights(weights: &[f64], l: usize) -> Result<()> {
        if weights.len() != l {
            return Err(eyre!("number of fixed weights {} doesn't match {}", weights.len(), l));
        }
        for &v in weights {
            if v < 0.0 {
                return Err(eyre!("weights must all be non-negative: {}", v));
            }
        }
        Ok(())
    }

    fn crossover<E: Evaluator<State = S>>(
        &self,
        crossover: &Crossover,
        eval: &E,
        s1: &mut Member<S>,
        s2: &mut Member<S>,
    ) -> Result<()> {
        match crossover {
            Crossover::Fixed(rates) => {
                s1.params.crossover.clone_from(rates);
                s2.params.crossover.clone_from(rates);
            }
            Crossover::Adaptive => {
                let lrate = 1.0 / (self.mems.len() as f64).sqrt();
                mutate_rate(&mut s1.params.crossover, 1.0, |v| mutate_normal(v, lrate).max(0.0));
                mutate_rate(&mut s2.params.crossover, 1.0, |v| mutate_normal(v, lrate).max(0.0));
            }
        }
        Self::check_weights(&s1.params.crossover, E::NUM_CROSSOVER)?;
        Self::check_weights(&s2.params.crossover, E::NUM_CROSSOVER)?;
        let idx = rws(&s1.params.crossover).unwrap();
        eval.crossover(&mut s1.state, &mut s2.state, idx);
        Ok(())
    }

    fn mutation<E: Evaluator<State = S>>(
        &self,
        mutation: &Mutation,
        eval: &E,
        s: &mut Member<S>,
    ) -> Result<()> {
        match mutation {
            Mutation::Fixed(rates) => {
                s.params.mutation.clone_from(rates);
            }
            Mutation::Adaptive => {
                // Apply every mutation with the given rate.
                // c' = c * e^(learning rate * N(0, 1))
                let lrate = 1.0 / (self.mems.len() as f64).sqrt();
                mutate_rate(&mut s.params.mutation, 1.0, |v| {
                    mutate_lognorm(v, lrate).clamp(0.0, 1.0)
                });
            }
        }
        Self::check_weights(&s.params.mutation, E::NUM_MUTATION)?;
        for (idx, &rate) in s.params.mutation.iter().enumerate() {
            eval.mutate(&mut s.state, rate, idx);
        }
        Ok(())
    }

    pub fn next_gen<E: Evaluator<State = S>>(
        &self,
        genfn: &mut (dyn RandState<S> + '_),
        stagnant: bool,
        cfg: &EvolveCfg,
        eval: &E,
    ) -> Result<UnevaluatedGenr<S>> {
        // Pick survivors:
        let mut new_mems = self.survivors(cfg.survival, cfg);
        // Min here to avoid underflow - can happen if we produce too many parents.
        new_mems.reserve(cfg.pop_size);

        // If stagnant, fill with random individuals.
        if stagnant {
            let num = match cfg.replacement {
                Replacement::ReplaceChildren(prop) => {
                    let remaining = cfg.pop_size as f64 - new_mems.len() as f64;
                    (prop * remaining).ceil().max(0.0) as usize
                }
            };
            for _ in 0..num {
                new_mems.push(Member::new::<E>((*genfn)(), cfg));
            }
        }

        // If DisallowDuplicates on, try up to NUM_TRIES times
        // to fill the population up.
        const NUM_TRIES: usize = 3;
        for _ in 0..NUM_TRIES {
            // Reproduce.
            while new_mems.len() < cfg.pop_size {
                let [mut s1, mut s2] = self.selection(cfg.selection);
                self.crossover(&cfg.crossover, eval, &mut s1, &mut s2).unwrap();
                self.mutation(&cfg.mutation, eval, &mut s1).unwrap();
                self.mutation(&cfg.mutation, eval, &mut s2).unwrap();
                new_mems.push(s1);
                new_mems.push(s2);
            }

            // Remove duplicates if we need to.
            if cfg.duplicates == Duplicates::DisallowDuplicates {
                new_mems.sort_unstable_by(|a, b| a.state.partial_cmp(&b.state).unwrap());
                new_mems.dedup_by(|a, b| a.state.eq(&b.state));
            }
        }
        Ok(UnevaluatedGenr::new(new_mems))
    }
}
