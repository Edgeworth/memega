use std::collections::HashMap;

use memega::eval::Evaluator;
use memega::evaluators::lgp::builder::lgp_fitness_evolver;
use memega::evaluators::lgp::cfg::LgpEvaluatorCfg;
use memega::evaluators::lgp::eval::LgpState;
use memega::evaluators::lgp::vm::lgpvm::LgpVm;
use memega::evolve::cfg::EvolveCfg;
use memega::evolve::evolver::Evolver;
use num_traits::ToPrimitive;
use rand::Rng;
use savage_core::expression::{Expression, Rational};

const NUM_REG: usize = 2;
const NUM_CONST: usize = 4;
const OUTPUT_REG: u8 = 0;

#[must_use]
pub fn lgp_fitness(s: &LgpState, _gen: usize, target: &str) -> f64 {
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

        let regs: [f64; NUM_REG] = [0.0, 0.0];
        let constants: [f64; NUM_CONST] = [0.0, -1.0, 1.0, x];
        let cfg = s.lgpvmcfg(&regs, &constants);
        let mut exec = LgpVm::new(&cfg);
        exec.run();

        fitness += 1.0 / (1.0 + (ans - exec.mem(OUTPUT_REG)).abs());
    }
    fitness / NUM_SAMPLES as f64
}

#[must_use]
pub fn lgp_evolver(
    target: String,
    lgpcfg: LgpEvaluatorCfg,
    cfg: EvolveCfg,
) -> Evolver<impl Evaluator> {
    lgp_fitness_evolver(
        lgpcfg.set_num_reg(NUM_REG).set_num_const(NUM_CONST).set_output_regs(&[OUTPUT_REG]),
        cfg,
        move |s: &LgpState, gen| lgp_fitness(s, gen, &target),
    )
}
