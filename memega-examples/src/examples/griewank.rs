use memega::eval::Evaluator;
use memega::evolve::cfg::EvolveCfg;
use memega::evolve::evolver::Evolver;

use crate::examples::func::{func_evolver, FuncState};

#[must_use]
pub fn griewank_evolver(dim: usize, cfg: EvolveCfg) -> Evolver<impl Evaluator<Data = ()>> {
    func_evolver(
        dim,
        -10000.0,
        10000.0,
        |s: &'_ FuncState, _: &'_ _| {
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
