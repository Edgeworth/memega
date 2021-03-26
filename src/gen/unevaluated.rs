use crate::cfg::{Cfg, Niching, Species, EP};
use crate::gen::evaluated::{EvaluatedGen, Member};
use crate::gen::species::{DistCache, SpeciesInfo, NO_SPECIES};
use crate::gen::Params;
use crate::{Evaluator, Genome, State};
use eyre::{eyre, Result};
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Clone, PartialOrd, PartialEq)]
pub struct UnevaluatedGen<T: Genome> {
    states: Vec<State<T>>,
    base_fitness: Vec<f64>,
    species: SpeciesInfo,
    dists: DistCache,
}

impl<T: Genome> UnevaluatedGen<T> {
    pub fn initial<E: Evaluator>(genomes: Vec<T>, cfg: &Cfg) -> Self {
        let states = genomes
            .into_iter()
            .map(|genome| State {
                genome,
                params: Params::new::<E>(cfg),
                species: NO_SPECIES,
            })
            .collect();
        Self::new(states)
    }

    pub fn new(states: Vec<State<T>>) -> Self {
        if states.is_empty() {
            panic!("Generation must not be empty");
        }
        let species = SpeciesInfo::new(states.len());
        Self {
            states,
            base_fitness: Vec::new(),
            species,
            dists: DistCache::new(),
        }
    }

    pub fn evaluate<E: Evaluator<Genome = T>>(
        &mut self,
        cfg: &Cfg,
        eval: &E,
    ) -> Result<(EvaluatedGen<T>, f64)> {
        // First compute plain fitnesses.
        self.base_fitness = if cfg.par_fitness {
            self.states
                .par_iter_mut()
                .map(|s| eval.fitness(&s.genome))
                .collect()
        } else {
            self.states
                .iter_mut()
                .map(|s| eval.fitness(&s.genome))
                .collect()
        };

        // Check fitnesses are non-negative.
        if !self.base_fitness.iter().all(|&v| v >= 0.0) {
            return Err(eyre!("got negative fitness"));
        }

        // Speciate if necessary.
        match cfg.species {
            Species::None => {}
            Species::TargetNumber(target) => {
                self.dists.ensure(&self.states, cfg.par_dist, eval);
                let mut lo = 0.0;
                let mut hi = self.dists.max();
                while hi - lo > EP {
                    let r = (lo + hi) / 2.0;
                    self.species = self.dists.speciate(&self.states, r);
                    match self.species.num.cmp(&target) {
                        Ordering::Less => hi = self.species.radius,
                        Ordering::Equal => break,
                        Ordering::Greater => lo = self.species.radius,
                    }
                }
            }
        }

        // Transform fitness if necessary.
        let selection_fitness = match cfg.niching {
            Niching::None => self.base_fitness.clone(),
            Niching::SharedFitness => {
                self.dists.ensure(&self.states, cfg.par_dist, eval);
                self.dists.shared_fitness(&self.base_fitness, &self.species)
            }
        };

        // Assign species into states if speciated.
        for (i, &id) in self.species.ids.iter().enumerate() {
            self.states[i].species = id;
        }

        Ok((
            EvaluatedGen::new(
                (0..self.states.len())
                    .map(|i| Member {
                        state: self.states[i].clone(),
                        base_fitness: self.base_fitness[i],
                        selection_fitness: selection_fitness[i],
                    })
                    .collect(),
            ),
            self.dists.mean(),
        ))
    }
}
