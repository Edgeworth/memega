use eyre::Result;

use crate::eval::Evaluator;
use crate::evolve::evolver::Evolver;
use crate::evolve::result::EvolveResult;
use crate::harness::cfg::{HarnessCfg, Termination};

/// Runs evolution with the given parameters and prints some info.
pub struct Harness {
    cfg: HarnessCfg,
}

impl Harness {
    #[must_use]
    pub fn new(cfg: HarnessCfg) -> Self {
        Self { cfg }
    }

    pub fn evolve<E: Evaluator>(&self, mut evolver: Evolver<E>) -> Result<EvolveResult<E::State>> {
        let mut ret = None;
        for i in 0.. {
            match self.cfg.termination() {
                Termination::FixedGenerations(gen) => {
                    if i >= gen {
                        break;
                    }
                }
            }
            let mut r = evolver.run_iter()?;
            if let Some(print_gen) = self.cfg.print_gen() && i % print_gen == 0 {
                println!("Generation {}: {}", i, r.nth(0).base_fitness);
            }
            if let Some(print_summary) = self.cfg.print_summary() && i % print_summary == 0 {
                println!("{}", evolver.summary(&mut r));
                println!("{}", evolver.summary_sample(&mut r, 5));
            }
            ret = Some(r);
        }
        Ok(ret.unwrap())
    }
}
