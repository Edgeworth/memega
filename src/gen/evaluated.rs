use derive_more::Display;
use eyre::{eyre, Result};

use crate::eval::{Evaluator, State};
use crate::evolve::cfg::{
    Crossover, Duplicates, EvolveCfg, Mutation, Replacement, Selection, Survival,
};
use crate::evolve::evolver::RandState;
use crate::gen::member::Member;
use crate::gen::species::SpeciesId;
use crate::gen::unevaluated::UnevaluatedGen;
use crate::ops::mutation::{mutate_lognorm, mutate_normal, mutate_rate};
use crate::ops::sampling::{multi_rws, rws, sus};

#[derive(Display, Clone, PartialOrd, PartialEq)]
#[display(fmt = "pop: {}, best: {}", "mems.len()", "self.mems[0]")]
pub struct EvaluatedGen<S: State> {
    pub mems: Vec<Member<S>>,
}

impl<S: State> EvaluatedGen<S> {
    #[must_use]
    pub fn new(mut mems: Vec<Member<S>>) -> Self {
        // Sort by base fitness. Selection should happen using selection
        // fitness. Generate survivors using base fitness, to make sure we keep
        // the top individuals.
        mems.sort_unstable_by(|a, b| b.base_fitness.partial_cmp(&a.base_fitness).unwrap());
        Self { mems }
    }

    #[must_use]
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
        match survival {
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
        }
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
        for &v in weights.iter() {
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
                s1.params.crossover = rates.clone();
                s2.params.crossover = rates.clone();
            }
            Crossover::Adaptive => {
                let lrate = 1.0 / (self.mems.len() as f64).sqrt();
                mutate_rate(&mut s1.params.crossover, 1.0, |v| mutate_normal(v, lrate).max(0.0));
                mutate_rate(&mut s2.params.crossover, 1.0, |v| mutate_normal(v, lrate).max(0.0));
            }
        };
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
                s.params.mutation = rates.clone();
            }
            Mutation::Adaptive => {
                // Apply every mutation with the given rate.
                // c' = c * e^(learning rate * N(0, 1))
                let lrate = 1.0 / (self.mems.len() as f64).sqrt();
                mutate_rate(&mut s.params.mutation, 1.0, |v| {
                    mutate_lognorm(v, lrate).clamp(0.0, 1.0)
                });
            }
        };
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
    ) -> Result<UnevaluatedGen<S>> {
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
        Ok(UnevaluatedGen::new(new_mems))
    }
}
