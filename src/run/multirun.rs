use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::cfg::Cfg;
use crate::eval::Evaluator;
use crate::run::result::RunResult;
use crate::run::runner::{CreateRunnerFn, Runner};

/// Run multiple instances of a runner.
pub fn multirun<F: CreateRunnerFn<E>, E: Evaluator + Sized>(
    num_runs: usize,
    num_generations: usize,
    cfg: &Cfg,
    f: F,
) -> Vec<(Runner<E>, RunResult<E::Genome>)> {
    let runners: Vec<Runner<E>> = (0..num_runs).map(|_| f(cfg.clone())).collect();
    runners
        .into_par_iter()
        .map(|mut runner| {
            let mut r = runner.run_iter().unwrap();
            for _ in 0..num_generations {
                r = runner.run_iter().unwrap();
            }
            (runner, r)
        })
        .collect()
}