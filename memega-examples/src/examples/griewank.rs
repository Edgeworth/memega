use memega::cfg::Cfg;
use memega::eval::FitnessFn;
use memega::run::runner::Runner;

use crate::examples::func::{func_runner, FuncEvaluator, FuncState};

#[must_use]
pub fn griewank_runner(dim: usize, cfg: Cfg) -> Runner<FuncEvaluator<impl FitnessFn<FuncState>>> {
    func_runner(
        dim,
        -10000.0,
        10000.0,
        |s: &FuncState, _| {
            let mut add = 0.0;
            let mut mul = 1.0;
            for (i, &x) in s.iter().enumerate() {
                add += x * x;
                mul *= (x / (i as f64 + 1.0).sqrt()).cos();
            }
            let v = 1.0 + add / 4000.0 - mul;
            // Convert to a maximisation problem
            1.0 / (1.0 + v)
        },
        cfg,
    )
}
