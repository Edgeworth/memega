use std::mem::swap;
use std::time::{Duration, Instant};

use crate::cfg::Cfg;
use crate::eval::Evaluator;
use crate::evaluators::hyper::eval::{HyperEvaluator, HyperState, StatFn};
use crate::evolve::evolver::{CreateEvolverFn, Evolver};
use crate::evolve::result::Stats;

pub struct HyperBuilder {
    stat_fns: Vec<Box<dyn StatFn>>,
    pop_size: usize,
    num_crossover: usize,
    num_mutation: usize,
    sample_dur: Duration,
}

impl HyperBuilder {
    #[must_use]
    pub fn new(pop_size: usize, sample_dur: Duration) -> Self {
        Self { stat_fns: Vec::new(), pop_size, num_crossover: 0, num_mutation: 0, sample_dur }
    }

    /// Add a evolver for which we should optimise the hyperparameters for.
    /// Adding multiple evolvers will optimise a common set of hyperparameters
    /// over all of them.
    pub fn add<F: CreateEvolverFn<E>, E: Evaluator>(&mut self, max_fitness: f64, f: F) {
        self.num_crossover = self.num_crossover.max(E::NUM_CROSSOVER);
        self.num_mutation = self.num_mutation.max(E::NUM_MUTATION);
        let sample_dur = self.sample_dur;
        self.stat_fns.push(Box::new(move |cfg| {
            let mut evolver = f(cfg);
            let st = Instant::now();
            let mut r1 = None;
            let mut r2 = None;
            while (Instant::now() - st) < sample_dur {
                swap(&mut r1, &mut r2);
                r2 = Some(evolver.run_iter().unwrap());
            }

            // Get the last run that ran in time.
            if let Some(mut r) = r1 {
                let mut stats = Stats::from_result(&mut r);
                stats.best_fitness /= max_fitness;
                stats.mean_fitness /= max_fitness;
                Some(stats)
            } else {
                None
            }
        }));
    }

    #[must_use]
    pub fn build(self, cfg: Cfg) -> Evolver<HyperEvaluator> {
        let pop_size = self.pop_size;
        let num_crossover = self.num_crossover;
        let num_mutation = self.num_mutation;
        let state_fn = move || HyperState::rand(pop_size, num_crossover, num_mutation);
        Evolver::new(HyperEvaluator::new(self.stat_fns), cfg, state_fn)
    }
}
