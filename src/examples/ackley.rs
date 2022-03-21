use std::f64::consts::{E, PI};

use crate::cfg::Cfg;
use crate::eval::FitnessFn;
use crate::examples::func::{func_runner, FuncEvaluator, FuncState};
use crate::run::runner::Runner;

#[must_use]
pub fn ackley_runner(dim: usize, cfg: Cfg) -> Runner<FuncEvaluator<impl FitnessFn<FuncState>>> {
    func_runner(
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
