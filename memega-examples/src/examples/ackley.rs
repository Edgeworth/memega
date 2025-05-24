use std::f64::consts::{E, PI};

use memega::eval::Evaluator;
use memega::evolve::cfg::EvolveCfg;
use memega::evolve::evolver::Evolver;

use crate::examples::func::{FuncState, func_evolver};

pub fn ackley_evolver(dim: usize, cfg: EvolveCfg) -> Evolver<impl Evaluator<Data = ()>> {
    func_evolver(
        dim,
        -32.768,
        32.768,
        |s: &'_ FuncState, (): &'_ _| {
            const A: f64 = 20.0;
            const B: f64 = 0.2;
            const C: f64 = 2.0 * PI;
            let d = s.len() as f64;
            let mut squares = 0.0;
            let mut cos = 0.0;
            for &x in s.iter() {
                squares += x * x;
                cos += (C * x).cos();
            }
            let squares = -B * (squares / d).sqrt();
            let cos = cos / d;
            let v = -A * squares.exp() - cos.exp() + A + E;
            // Convert to a maximisation problem
            Ok(1.0 / (1.0 + v))
        },
        cfg,
    )
}
