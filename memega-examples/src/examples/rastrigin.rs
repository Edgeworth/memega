use std::f64::consts::PI;

use memega::cfg::Cfg;
use memega::eval::FitnessFn;
use memega::run::runner::Runner;

use crate::examples::func::{func_runner, FuncEvaluator, FuncState};

#[must_use]
pub fn rastrigin_runner(dim: usize, cfg: Cfg) -> Runner<FuncEvaluator<impl FitnessFn<FuncState>>> {
    func_runner(
        dim,
        -5.12,
        5.12,
        |s: &FuncState, _| {
            const A: f64 = 10.0;
            let mut v = 0.0;
            for &x in s.iter() {
                v += A + x * x - A * (2.0 * PI * x).cos();
            }
            // Convert to a maximisation problem
            1.0 / (1.0 + v)
        },
        cfg,
    )
}
