use crate::cfg::{Cfg, Crossover, Mutation, Selection, Stagnation, Survival};
use crate::gen::unevaluated::UnevaluatedGen;
use crate::gen::Params;
use crate::ops::mutation::{mutate_lognorm, mutate_normal, mutate_rate};
use crate::ops::sampling::{multi_rws, rws, sus};
use crate::runner::RandGenome;
use crate::{Evaluator, Genome, State};
use derive_more::Display;
use eyre::{eyre, Result};

#[derive(Display, Clone, PartialOrd, PartialEq)]
#[display(
    fmt = "base fitness {:.2}, selection fitness {:.2}",
    base_fitness,
    selection_fitness
)]
pub struct Member<T: Genome> {
    pub state: State<T>,
    pub base_fitness: f64, // Original fitness, generated by Evaluator fitness function.
    pub selection_fitness: f64, // Potentially adjusted fitness, for selection.
    pub species: usize,
}

#[derive(Display, Clone, PartialOrd, PartialEq)]
#[display(fmt = "pop: {}, best: {}", "mems.len()", "self.best()")]
pub struct EvaluatedGen<T: Genome> {
    mems: Vec<Member<T>>,
}

impl<T: Genome> EvaluatedGen<T> {
    pub fn new(mut mems: Vec<Member<T>>) -> Self {
        mems.sort_unstable_by(|a, b| {
            b.selection_fitness
                .partial_cmp(&a.selection_fitness)
                .unwrap()
        });
        Self { mems }
    }

    pub fn size(&self) -> usize {
        self.mems.len()
    }

    pub fn best(&self) -> Member<T> {
        self.mems[0].clone()
    }

    pub fn mean_base_fitness(&self) -> f64 {
        self.mems.iter().map(|v| v.base_fitness).sum::<f64>() / self.mems.len() as f64
    }

    pub fn num_dup(&self) -> usize {
        let mut mems_copy = self
            .mems
            .iter()
            .map(|v| &v.state.0)
            .cloned()
            .collect::<Vec<_>>();
        mems_copy.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        mems_copy.dedup();
        self.mems.len() - mems_copy.len()
    }

    pub fn num_species(&self) -> usize {
        // Relies on species index assignment to be contigous from zero.
        self.mems.iter().map(|mem| mem.species).max().unwrap_or(0) + 1
    }

    fn take_proportion(mems: &[Member<T>], prop: f64) -> Vec<State<T>> {
        // Ceiling so we don't miss keeping things for small sizes.
        let num = (mems.len() as f64 * prop).ceil() as usize;
        mems.iter().map(|v| &v.state).cloned().take(num).collect()
    }

    fn survivors(&self, survival: Survival) -> Vec<State<T>> {
        match survival {
            Survival::TopProportion(prop) => Self::take_proportion(&self.mems, prop),
            Survival::SpeciesTopProportion(prop) => {
                let mut by_species = vec![Vec::new(); self.num_species()];
                self.mems
                    .iter()
                    .for_each(|mem| by_species[mem.species].push(mem.clone()));
                by_species
                    .into_iter()
                    .map(|v| Self::take_proportion(&v, prop))
                    .flatten()
                    .collect()
            }
        }
    }

    fn selection(&self, selection: Selection) -> [State<T>; 2] {
        let fitnesses = self
            .mems
            .iter()
            .map(|v| v.selection_fitness)
            .collect::<Vec<_>>();
        let idxs = match selection {
            Selection::Sus => sus(&fitnesses, 2),
            Selection::Roulette => multi_rws(&fitnesses, 2),
        };
        [
            self.mems[idxs[0]].state.clone(),
            self.mems[idxs[1]].state.clone(),
        ]
    }

    fn check_weights(weights: &[f64], l: usize) -> Result<()> {
        if weights.len() != l {
            return Err(eyre!(
                "number of fixed weights {} doesn't match {}",
                weights.len(),
                l
            ));
        }
        for &v in weights.iter() {
            if v < 0.0 {
                return Err(eyre!("weights must all be non-negative: {}", v));
            }
        }
        Ok(())
    }

    fn crossover<E: Evaluator<Genome = T>>(
        &self,
        crossover: &Crossover,
        eval: &E,
        s1: &mut State<T>,
        s2: &mut State<T>,
    ) -> Result<()> {
        match crossover {
            Crossover::Fixed(rates) => {
                s1.1.crossover = rates.clone();
                s2.1.crossover = rates.clone();
            }
            Crossover::Adaptive => {
                let lrate = 1.0 / (self.size() as f64).sqrt();
                mutate_rate(&mut s1.1.crossover, 1.0, |v| {
                    mutate_normal(v, lrate).max(0.0)
                });
                mutate_rate(&mut s2.1.crossover, 1.0, |v| {
                    mutate_normal(v, lrate).max(0.0)
                });
            }
        };
        Self::check_weights(&s1.1.crossover, E::NUM_CROSSOVER)?;
        Self::check_weights(&s2.1.crossover, E::NUM_CROSSOVER)?;
        let idx = rws(&s1.1.crossover).unwrap();
        eval.crossover(&mut s1.0, &mut s2.0, idx);
        Ok(())
    }

    fn mutation<E: Evaluator<Genome = T>>(
        &self,
        mutation: &Mutation,
        eval: &E,
        s: &mut State<T>,
    ) -> Result<()> {
        match mutation {
            Mutation::Fixed(rates) => {
                s.1.mutation = rates.clone();
            }
            Mutation::Adaptive => {
                // Apply every mutation with the given rate.
                // c' = c * e^(learning rate * N(0, 1))
                let lrate = 1.0 / (self.size() as f64).sqrt();
                mutate_rate(&mut s.1.mutation, 1.0, |v| {
                    mutate_lognorm(v, lrate).clamp(0.0, 1.0)
                });
            }
        };
        Self::check_weights(&s.1.mutation, E::NUM_MUTATION)?;
        for (idx, &rate) in s.1.mutation.iter().enumerate() {
            eval.mutate(&mut s.0, rate, idx);
        }
        Ok(())
    }


    pub fn next_gen<E: Evaluator<Genome = T>>(
        &self,
        genfn: Option<&mut (dyn RandGenome<T> + '_)>,
        cfg: &Cfg,
        eval: &E,
    ) -> Result<UnevaluatedGen<T>> {
        // Pick survivors:
        let mut new_states = self.survivors(cfg.survival);
        // Min here to avoid underflow - can happen if we produce too many parents.
        new_states.reserve(cfg.pop_size);
        if let Some(genfn) = genfn {
            // Use custom generation function, e.g. for stagnation.
            while new_states.len() < cfg.pop_size {
                new_states.push(((*genfn)(), Params::new::<E>(cfg)));
            }
        } else {
            // Reproduce. If DisallowDuplicates on, try up to NUM_TRIES times
            // to fill the population up.
            const NUM_TRIES: usize = 3;
            for _ in 0..NUM_TRIES {
                while new_states.len() < cfg.pop_size {
                    let [mut s1, mut s2] = self.selection(cfg.selection);
                    self.crossover(&cfg.crossover, eval, &mut s1, &mut s2)
                        .unwrap();
                    self.mutation(&cfg.mutation, eval, &mut s1).unwrap();
                    self.mutation(&cfg.mutation, eval, &mut s2).unwrap();
                    new_states.push(s1);
                    new_states.push(s2);
                }

                // Remove duplicates if we need to.
                if cfg.stagnation == Stagnation::DisallowDuplicates {
                    new_states.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    new_states.dedup_by(|a, b| a.0.eq(&b.0));
                }
            }
        }
        Ok(UnevaluatedGen::new(new_states))
    }
}
