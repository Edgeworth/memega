use crate::cfg::{Cfg, Niching, Species, EP};
use crate::gen::evaluated::EvaluatedGen;
use crate::gen::species::{DistCache, SpeciesInfo};
use crate::{Evaluator, Genome, Mem};
use eyre::{eyre, Result};
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Clone, PartialOrd, PartialEq)]
pub struct UnevaluatedGen<G: Genome> {
    pub(crate) mems: Vec<Mem<G>>,
    pub(crate) species: SpeciesInfo,
    pub(crate) dists: DistCache,
}

impl<G: Genome> UnevaluatedGen<G> {
    pub fn initial<E: Evaluator>(genomes: Vec<G>, cfg: &Cfg) -> Self {
        let mems = genomes
            .into_iter()
            .map(|genome| Mem::new::<E>(genome, cfg))
            .collect();
        Self::new(mems)
    }

    pub fn new(mems: Vec<Mem<G>>) -> Self {
        if mems.is_empty() {
            panic!("Generation must not be empty");
        }
        Self {
            mems,
            species: SpeciesInfo::new(),
            dists: DistCache::new(),
        }
    }

    pub fn evaluate<E: Evaluator<Genome = G>>(
        &mut self,
        cfg: &Cfg,
        eval: &E,
    ) -> Result<EvaluatedGen<G>> {
        // First compute plain fitnesses.
        if cfg.par_fitness {
            self.mems
                .par_iter_mut()
                .for_each(|s| s.base_fitness = eval.fitness(&s.genome))
        } else {
            self.mems
                .iter_mut()
                .for_each(|s| s.base_fitness = eval.fitness(&s.genome))
        };

        // Check fitnesses are non-negative.
        if !self.mems.iter().map(|v| v.base_fitness).all(|v| v >= 0.0) {
            return Err(eyre!("got negative fitness"));
        }

        // Sort by fitnesses.
        self.mems
            .sort_unstable_by(|a, b| b.base_fitness.partial_cmp(&a.base_fitness).unwrap());

        // Speciate if necessary.
        match cfg.species {
            Species::None => {}
            Species::TargetNumber(target) => {
                self.dists.ensure(&self.mems, cfg.par_dist, eval);
                let mut lo = 0.0;
                let mut hi = self.dists.max();
                let mut ids = Vec::new();
                while hi - lo > EP {
                    let r = (lo + hi) / 2.0;
                    (ids, self.species) = self.dists.speciate(&self.mems, r);
                    match self.species.num.cmp(&target) {
                        Ordering::Less => hi = self.species.radius,
                        Ordering::Equal => break,
                        Ordering::Greater => lo = self.species.radius,
                    }
                }
                // Assign species into mems if speciated.
                for (i, &id) in ids.iter().enumerate() {
                    self.mems[i].species = id;
                }
            }
        }

        // Transform fitness if necessary.
        match cfg.niching {
            Niching::None => {
                for v in self.mems.iter_mut() {
                    v.selection_fitness = v.base_fitness;
                }
            }
            Niching::SharedFitness(radius) => {
                const ALPHA: f64 = 6.0; // Default alpha between 5 and 10.
                self.dists.ensure(&self.mems, cfg.par_dist, eval);
                self.dists.shared_fitness(&mut self.mems, radius, ALPHA);
            }
            Niching::SpeciesSharedFitness => {
                self.dists.ensure(&self.mems, cfg.par_dist, eval);
                self.dists
                    .species_shared_fitness(&mut self.mems, &self.species);
            }
        };

        Ok(EvaluatedGen::new(self.mems.clone()))
    }
}
