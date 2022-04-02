use std::collections::HashMap;

use memega::cfg::Cfg;
use memega::eval::Evaluator;
use memega::evaluators::lgp::builder::lgp_fitness_evolver;
use memega::evaluators::lgp::cfg::LgpCfg;
use memega::evaluators::lgp::eval::State;
use memega::evaluators::lgp::vm::exec::LgpExec;
use memega::evolve::evolver::Evolver;
use num_traits::ToPrimitive;
use rand::Rng;
use savage_core::expression::{Expression, Rational};

#[must_use]
pub fn lgp_fitness(s: &State, _gen: usize, target: &str) -> f64 {
    const NUM_SAMPLES: usize = 100;

    let expr: Expression = target.parse().unwrap();

    let mut fitness = 0.0;
    for _ in 0..NUM_SAMPLES {
        let mut r = rand::thread_rng();
        let x = r.gen_range(-100.0..100.0);

        let mut expr_ctx = HashMap::new();
        let x_expr = Expression::from(Rational::from_float(x).unwrap());
        expr_ctx.insert("x".to_string(), x_expr);
        let ans = expr.evaluate(expr_ctx).unwrap();
        let ans = match ans {
            Expression::Integer(integer) => integer.to_f64().unwrap(),
            Expression::Rational(ratio, _) => ratio.to_f64().unwrap(),
            _ => panic!("should be number output: {}", ans),
        };

        let mut reg = vec![0.0; s.cfg.num_reg()]; // Space for work and answer.
        reg[1] = -1.0;
        reg[2] = 1.0;
        reg[3] = x;
        let mut exec = LgpExec::new(&reg, &s.ops, s.cfg.max_iter());
        exec.run();


        fitness += 1.0 / (1.0 + (ans - exec.reg(0)).abs());
    }
    fitness / NUM_SAMPLES as f64 + 0.1 / (1.0 + s.ops.len() as f64)
}

#[must_use]
pub fn lgp_evolver(target: String, cfg: Cfg) -> Evolver<impl Evaluator> {
    lgp_fitness_evolver(LgpCfg::new(), cfg, move |s: &State, gen| lgp_fitness(s, gen, &target))
}
