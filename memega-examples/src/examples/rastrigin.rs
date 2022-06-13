use std::f64::consts::PI;

use memega::eval::Evaluator;
use memega::evolve::cfg::EvolveCfg;
use memega::evolve::evolver::Evolver;

use crate::examples::func::{func_evolver, FuncState};

pub fn rastrigin_evolver(dim: usize, cfg: EvolveCfg) -> Evolver<impl Evaluator<Data = ()>> {
    func_evolver(
        dim,
        -5.12,
        5.12,
        |s: &'_ FuncState, _: &'_ _| {
            const A: f64 = 10.0;
            let mut v = 0.0;
            for &x in s.iter() {
                v += A + x * x - A * (2.0 * PI * x).cos();
            }
            // Convert to a maximisation problem
            Ok(1.0 / (1.0 + v))
        },
        cfg,
    )
}
