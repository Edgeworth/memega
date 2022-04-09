use std::cmp::Ordering;

use approx::relative_eq;
use eyre::{eyre, Result};
use rayon::prelude::*;

use crate::eval::{Evaluator, State};
use crate::evolve::cfg::{EvolveCfg, Niching, Species};
use crate::gen::evaluated::EvaluatedGen;
use crate::gen::member::Member;
use crate::gen::species::{DistCache, SpeciesInfo};

#[derive(Clone, PartialOrd, PartialEq)]
pub struct UnevaluatedGen<S: State> {
    pub mems: Vec<Member<S>>,
    pub species: SpeciesInfo,
    pub dists: DistCache,
}

impl<S: State> UnevaluatedGen<S> {
    #[must_use]
    pub fn initial<E: Evaluator>(states: Vec<S>, cfg: &EvolveCfg) -> Self {
        let mems = states.into_iter().map(|state| Member::new::<E>(state, cfg)).collect();
        Self::new(mems)
    }

    #[must_use]
    pub fn new(mems: Vec<Member<S>>) -> Self {
        assert!(!mems.is_empty(), "Generation must not be empty");
        Self { mems, species: SpeciesInfo::new(), dists: DistCache::new() }
    }

    pub fn evaluate<E: Evaluator<State = S>>(
        &mut self,
        gen_count: usize,
        cfg: &EvolveCfg,
        eval: &E,
    ) -> Result<EvaluatedGen<S>> {
        // First compute plain fitnesses.
        if cfg.par_fitness {
            self.mems
                .par_iter_mut()
                .for_each(|s| s.base_fitness = eval.fitness(&s.state, gen_count));
        } else {
            self.mems.iter_mut().for_each(|s| s.base_fitness = eval.fitness(&s.state, gen_count));
        };

        // Check fitnesses are non-negative.
        if !self.mems.iter().map(|v| v.base_fitness).all(|v| v >= 0.0) {
            return Err(eyre!("got negative fitness"));
        }

        // Sort by fitnesses.
        self.mems.sort_unstable_by(|a, b| b.base_fitness.partial_cmp(&a.base_fitness).unwrap());

        // Speciate if necessary.
        match cfg.species {
            Species::None => {}
            Species::TargetNumber(target) => {
                self.dists.ensure(&self.mems, cfg.par_dist, eval);
                let mut lo = 0.0;
                let mut hi = self.dists.max();
                let mut ids = Vec::new();
                while !relative_eq!(lo, hi, epsilon = 1.0e-6) {
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
                for v in &mut self.mems {
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
                self.dists.species_shared_fitness(&mut self.mems, &self.species);
            }
        };

        Ok(EvaluatedGen::new(self.mems.clone()))
    }
}
