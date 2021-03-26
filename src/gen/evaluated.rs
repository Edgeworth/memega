use crate::cfg::{Cfg, Crossover, Duplicates, Mutation, Selection, Survival};
use crate::gen::species::NO_SPECIES;
use crate::gen::unevaluated::UnevaluatedGen;
use crate::gen::Params;
use crate::ops::mutation::{mutate_lognorm, mutate_normal, mutate_rate};
use crate::ops::sampling::{multi_rws, rws, sus};
use crate::runner::RandGenome;
use crate::{Evaluator, Genome, Mem};
use derive_more::Display;
use eyre::{eyre, Result};

#[derive(Display, Clone, PartialOrd, PartialEq)]
#[display(fmt = "pop: {}, best: {}", "mems.len()", "self.mems[0]")]
pub struct EvaluatedGen<G: Genome> {
    pub(crate) mems: Vec<Mem<G>>,
}

impl<G: Genome> EvaluatedGen<G> {
    pub fn new(mut mems: Vec<Mem<G>>) -> Self {
        // Sort by base fitness. Selection should happen using selection
        // fitness. Generate survivors using base fitness, to make sure we keep
        // the top individuals.
        mems.sort_unstable_by(|a, b| b.base_fitness.partial_cmp(&a.base_fitness).unwrap());
        Self { mems }
    }

    pub fn species_mems(&self, n: u64) -> Vec<Mem<G>> {
        self.mems
            .iter()
            .filter(|v| v.species == n)
            .cloned()
            .collect()
    }

    // Get list of species.
    pub fn species(&self) -> Vec<u64> {
        // Relies on species index assignment to be contigous from zero.
        let mut species: Vec<_> = self.mems.iter().map(|mem| mem.species).collect();
        species.sort_unstable();
        species.dedup();
        species
    }

    fn survivors(&self, survival: Survival, cfg: &Cfg) -> Vec<Mem<G>> {
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

    fn selection(&self, selection: Selection) -> [Mem<G>; 2] {
        let fitnesses = self
            .mems
            .iter()
            .map(|v| v.selection_fitness)
            .collect::<Vec<_>>();
        let idxs = match selection {
            Selection::Sus => sus(&fitnesses, 2),
            Selection::Roulette => multi_rws(&fitnesses, 2),
        };
        [self.mems[idxs[0]].clone(), self.mems[idxs[1]].clone()]
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

    fn crossover<E: Evaluator<Genome = G>>(
        &self,
        crossover: &Crossover,
        eval: &E,
        s1: &mut Mem<G>,
        s2: &mut Mem<G>,
    ) -> Result<()> {
        match crossover {
            Crossover::Fixed(rates) => {
                s1.params.crossover = rates.clone();
                s2.params.crossover = rates.clone();
            }
            Crossover::Adaptive => {
                let lrate = 1.0 / (self.mems.len() as f64).sqrt();
                mutate_rate(&mut s1.params.crossover, 1.0, |v| {
                    mutate_normal(v, lrate).max(0.0)
                });
                mutate_rate(&mut s2.params.crossover, 1.0, |v| {
                    mutate_normal(v, lrate).max(0.0)
                });
            }
        };
        Self::check_weights(&s1.params.crossover, E::NUM_CROSSOVER)?;
        Self::check_weights(&s2.params.crossover, E::NUM_CROSSOVER)?;
        let idx = rws(&s1.params.crossover).unwrap();
        eval.crossover(&mut s1.genome, &mut s2.genome, idx);
        Ok(())
    }

    fn mutation<E: Evaluator<Genome = G>>(
        &self,
        mutation: &Mutation,
        eval: &E,
        s: &mut Mem<G>,
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
            eval.mutate(&mut s.genome, rate, idx);
        }
        Ok(())
    }


    pub fn next_gen<E: Evaluator<Genome = G>>(
        &self,
        mut genfn: Option<&mut (dyn RandGenome<G> + '_)>,
        cfg: &Cfg,
        eval: &E,
    ) -> Result<UnevaluatedGen<G>> {
        // Pick survivors:
        let mut new_mems = self.survivors(cfg.survival, cfg);
        // Min here to avoid underflow - can happen if we produce too many parents.
        new_mems.reserve(cfg.pop_size);

        // If DisallowDuplicates on, try up to NUM_TRIES times
        // to fill the population up.
        const NUM_TRIES: usize = 3;
        for _ in 0..NUM_TRIES {
            if let Some(genfn) = &mut genfn {
                // Use custom generation function, e.g. for stagnation.
                while new_mems.len() < cfg.pop_size {
                    new_mems.push(Mem {
                        genome: (*genfn)(),
                        params: Params::new::<E>(cfg),
                        species: NO_SPECIES,
                        base_fitness: 0.0,
                        selection_fitness: 0.0,
                    });
                }
            } else {
                // Reproduce.
                while new_mems.len() < cfg.pop_size {
                    let [mut s1, mut s2] = self.selection(cfg.selection);
                    self.crossover(&cfg.crossover, eval, &mut s1, &mut s2)
                        .unwrap();
                    self.mutation(&cfg.mutation, eval, &mut s1).unwrap();
                    self.mutation(&cfg.mutation, eval, &mut s2).unwrap();
                    new_mems.push(s1);
                    new_mems.push(s2);
                }
            }

            // Remove duplicates if we need to.
            if cfg.duplicates == Duplicates::DisallowDuplicates {
                new_mems.sort_unstable_by(|a, b| a.genome.partial_cmp(&b.genome).unwrap());
                new_mems.dedup_by(|a, b| a.genome.eq(&b.genome));
            }
        }
        Ok(UnevaluatedGen::new(new_mems))
    }
}
