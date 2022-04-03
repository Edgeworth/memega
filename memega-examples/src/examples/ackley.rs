use std::f64::consts::{E, PI};

use memega::cfg::Cfg;
use memega::eval::Evaluator;
use memega::evolve::evolver::Evolver;

use crate::examples::func::{func_evolver, FuncState};

#[must_use]
pub fn ackley_evolver(dim: usize, cfg: Cfg) -> Evolver<impl Evaluator> {
    func_evolver(
        dim,
        -32.768,
        32.768,
        |s: &FuncState, _| {
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
            1.0 / (1.0 + v)
        },
        cfg,
    )
}
