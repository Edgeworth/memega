use std::cmp::Ordering;

use approx::relative_eq;
use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::eval::{Evaluator, State};
use crate::evolve::cfg::{EvolveCfg, Niching, Species};
use crate::genr::evaluated::EvaluatedGen;
use crate::genr::member::Member;
use crate::genr::species::{DistCache, SpeciesInfo};

#[must_use]
#[derive(Clone, PartialOrd, PartialEq)]
pub struct UnevaluatedGenr<S: State> {
    pub mems: Vec<Member<S>>,
    pub species: SpeciesInfo,
    pub dists: DistCache,
}

impl<S: State> UnevaluatedGenr<S> {
    pub fn initial<E: Evaluator>(states: Vec<S>, cfg: &EvolveCfg) -> Self {
        let mems = states.into_iter().map(|state| Member::new::<E>(state, cfg)).collect();
        Self::new(mems)
    }

    pub fn new(mems: Vec<Member<S>>) -> Self {
        assert!(!mems.is_empty(), "Generation must not be empty");
        Self { mems, species: SpeciesInfo::new(), dists: DistCache::new() }
    }

    pub fn evaluate<E: Evaluator<State = S>>(
        &mut self,
        inputs: &[E::Data],
        cfg: &EvolveCfg,
        eval: &E,
    ) -> Result<EvaluatedGen<S>> {
        // First compute plain fitnesses.
        let compute = |s: &mut Member<S>| -> Result<()> {
            s.fitness = eval.multi_fitness(&s.state, inputs, cfg.fitness_reduction)?;
            Ok(())
        };
        if cfg.par_fitness {
            self.mems.par_iter_mut().try_for_each(compute)?;
        } else {
            self.mems.iter_mut().try_for_each(compute)?;
        }

        // Check fitnesses are non-negative and finite.
        if !self.mems.iter().map(|v| v.fitness).all(|v| v >= 0.0 && v.is_finite()) {
            return Err(Error::EvolverError("got negative or non-finite fitness".to_string()));
        }

        // Sort by fitnesses.
        self.mems.sort_unstable_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Speciate if necessary.
        match cfg.species {
            Species::None => {}
            Species::TargetNumber(target) => {
                self.dists.ensure(&self.mems, cfg.par_dist, eval)?;
                let mut lo = 0.0;
                let mut hi = self.dists.max();
                let mut ids;

                loop {
                    let r = f64::midpoint(lo, hi);
                    (ids, self.species) = self.dists.speciate(&self.mems, r);
                    match self.species.num.cmp(&target.get()) {
                        Ordering::Less => hi = self.species.radius,
                        Ordering::Equal => break,
                        Ordering::Greater => lo = self.species.radius,
                    }

                    if relative_eq!(lo, hi, epsilon = 1.0e-6) {
                        break;
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
                    v.selection_fitness = v.fitness;
                }
            }
            Niching::SharedFitness(radius) => {
                const ALPHA: f64 = 6.0; // Default alpha between 5 and 10.
                self.dists.ensure(&self.mems, cfg.par_dist, eval)?;
                self.dists.shared_fitness(&mut self.mems, radius, ALPHA);
            }
            Niching::SpeciesSharedFitness => {
                self.dists.ensure(&self.mems, cfg.par_dist, eval)?;
                self.dists.species_shared_fitness(&mut self.mems, &self.species);
            }
        }

        Ok(EvaluatedGen::new(self.mems.clone()))
    }
}

#[cfg(test)]
mod tests {
    use std::fmt;
    use std::num::NonZeroUsize;

    use super::UnevaluatedGenr;
    use crate::Result;
    use crate::eval::Evaluator;
    use crate::evolve::cfg::{EvolveCfg, Species};

    #[derive(Clone, PartialOrd, PartialEq, Debug)]
    struct TestState(u8);

    impl fmt::Display for TestState {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    #[derive(Debug, Default)]
    struct ConstFitnessEval;

    impl Evaluator for ConstFitnessEval {
        type State = TestState;
        type Data = ();

        fn crossover(&self, _s1: &mut Self::State, _s2: &mut Self::State, _idx: usize) {}
        fn mutate(&self, _s: &mut Self::State, _rate: f64, _idx: usize) {}

        fn fitness(&self, _s: &Self::State, _data: &Self::Data) -> Result<f64> {
            Ok(1.0)
        }

        fn distance(&self, _s1: &Self::State, _s2: &Self::State) -> Result<f64> {
            Ok(0.0)
        }
    }

    #[test]
    fn evaluate_assigns_species_when_all_dists_zero() -> Result<()> {
        let eval = ConstFitnessEval;
        let cfg = EvolveCfg::new(4)
            .set_species(Species::TargetNumber(NonZeroUsize::new(3).unwrap()))
            .set_par_dist(false);
        let mut genr = UnevaluatedGenr::initial::<ConstFitnessEval>(
            vec![TestState(0), TestState(1), TestState(2), TestState(3)],
            &cfg,
        );
        let evaluated = genr.evaluate(&[()], &cfg, &eval)?;
        assert!(evaluated.mems.iter().all(|m| m.species == 1));
        Ok(())
    }
}
