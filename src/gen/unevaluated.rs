use crate::cfg::{Cfg, Niching, Species, EP};
use crate::gen::evaluated::{EvaluatedGen, Member};
use crate::gen::species::DistCache;
use crate::gen::Params;
use crate::{Evaluator, Genome, State};
use eyre::{eyre, Result};
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Clone, PartialOrd, PartialEq)]
pub struct UnevaluatedGen<T: Genome> {
    states: Vec<State<T>>,
    base_fitness: Vec<f64>,
    species: Vec<i32>,
    num_species: usize,
    species_radius: f64,
    dists: DistCache,
}

impl<T: Genome> UnevaluatedGen<T> {
    pub fn initial<E: Evaluator>(genomes: Vec<T>, cfg: &Cfg) -> Self {
        let states = genomes
            .into_iter()
            .map(|v| (v, Params::new::<E>(cfg)))
            .collect();
        Self::new(states)
    }

    pub fn new(states: Vec<State<T>>) -> Self {
        if states.is_empty() {
            panic!("Generation must not be empty");
        }
        Self {
            states,
            base_fitness: Vec::new(),
            species: Vec::new(),
            num_species: 1,
            species_radius: 1.0,
            dists: DistCache::empty(),
        }
    }

    fn ensure_dists<E: Evaluator<Genome = T>>(&mut self, cfg: &Cfg, eval: &E) {
        if self.dists.is_empty() {
            self.dists = DistCache::new(eval, &self.states, cfg.par_dist);
        }
    }

    fn assign_species(&mut self, dist: f64) -> usize {
        let n = self.states.len();
        self.species = vec![-1; n];
        let mut num_species = 0;
        for i in 0..n {
            if self.species[i] != -1 {
                continue;
            }
            self.species[i] = num_species as i32;
            for j in (i + 1)..n {
                if self.dists[(i, j)] <= dist {
                    self.species[j] = num_species as i32;
                }
            }
            num_species += 1;
        }
        num_species
    }

    pub fn evaluate<E: Evaluator<Genome = T>>(
        &mut self,
        cfg: &Cfg,
        eval: &E,
    ) -> Result<EvaluatedGen<T>> {
        // First compute plain fitnesses.
        self.base_fitness = if cfg.par_fitness {
            self.states
                .par_iter_mut()
                .map(|s| eval.fitness(&s.0))
                .collect()
        } else {
            self.states.iter_mut().map(|s| eval.fitness(&s.0)).collect()
        };

        // Check fitnesses are non-negative.
        if !self.base_fitness.iter().all(|&v| v >= 0.0) {
            return Err(eyre!("got negative fitness"));
        }

        // Speciate if necessary.
        match cfg.species {
            Species::None => {}
            Species::TargetNumber(target) => {
                self.ensure_dists(cfg, eval);
                let mut lo = 0.0;
                let mut hi = self.dists.max();
                // TODO: tests
                while hi - lo > EP {
                    self.species_radius = (lo + hi) / 2.0;
                    self.num_species = self.assign_species(self.species_radius);
                    match self.num_species.cmp(&target) {
                        Ordering::Less => hi = self.species_radius,
                        Ordering::Equal => break,
                        Ordering::Greater => lo = self.species_radius,
                    }
                }
            }
        }

        // Transform fitness if necessary.
        // TODO: tests
        let selection_fitness = match cfg.niching {
            Niching::None => self.base_fitness.clone(),
            Niching::SharedFitness => {
                self.ensure_dists(cfg, eval);

                // Compute alpha as: radius / num_species ^ (1 / dimensionality)
                let alpha = self.species_radius / self.num_species as f64;
                let n = self.states.len();
                let mut fitness = self.base_fitness.clone();

                // Compute fitness as F'(i) = F(i) / sum of 1 - (d(i, j) / species_radius) ^ alpha.
                for i in 0..n {
                    let mut sum = 0.0;
                    for j in 0..n {
                        let d = self.dists[(i, j)];
                        if d < self.species_radius {
                            sum += 1.0 - (d / self.species_radius).powf(alpha)
                        }
                    }
                    fitness[i] /= sum;
                }
                fitness
            }
        };
        Ok(EvaluatedGen::new(
            (0..self.states.len())
                .map(|i| Member {
                    state: self.states[i].clone(),
                    base_fitness: self.base_fitness[i],
                    selection_fitness: selection_fitness[i],
                    species: *self.species.get(i).unwrap_or(&0) as usize,
                })
                .collect(),
        ))
    }
}
