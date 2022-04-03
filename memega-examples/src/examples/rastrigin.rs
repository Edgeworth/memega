use std::f64::consts::PI;

use memega::cfg::Cfg;
use memega::eval::Evaluator;
use memega::evolve::evolver::Evolver;

use crate::examples::func::{func_evolver, FuncState};

#[must_use]
pub fn rastrigin_evolver(dim: usize, cfg: Cfg) -> Evolver<impl Evaluator> {
    func_evolver(
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
