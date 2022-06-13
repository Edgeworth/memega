use std::collections::HashMap;

use eyre::{eyre, Result};
use memega::eval::Evaluator;
use memega::evaluators::lgp::builder::lgp_fitness_evolver;
use memega::evaluators::lgp::cfg::LgpEvaluatorCfg;
use memega::evaluators::lgp::eval::LgpState;
use memega::evaluators::lgp::vm::lgpvm::LgpVm;
use memega::evolve::cfg::EvolveCfg;
use memega::evolve::evolver::Evolver;
use memega::train::sampler::DataSampler;
use num_traits::ToPrimitive;
use savage_core::expression::{Expression, Rational};

const NUM_REG: usize = 2;
const NUM_CONST: usize = 4;
const OUTPUT_REG: u8 = 0;

pub fn expr_fitness(s: &LgpState, x: f64, target: &str) -> Result<f64> {
    let expr: Expression = target.parse().map_err(|_| eyre!("failed to parse expression"))?;

    let mut expr_ctx = HashMap::new();
    let x_expr = Expression::from(Rational::from_float(x).ok_or_else(|| eyre!("invalid x"))?);
    expr_ctx.insert("x".to_string(), x_expr);
    let ans = expr.evaluate(expr_ctx).map_err(|_| eyre!("failed to evaluate expression"))?;
    let ans = match ans {
        Expression::Integer(integer) => integer.to_f64().ok_or_else(|| eyre!("invalid y"))?,
        Expression::Rational(ratio, _) => ratio.to_f64().ok_or_else(|| eyre!("invalid y"))?,
        _ => panic!("should be number output: {ans}"),
    };

    let regs: [f64; NUM_REG] = [0.0, 0.0];
    let constants: [f64; NUM_CONST] = [0.0, -1.0, 1.0, x];
    let cfg = s.lgpvmcfg(&regs, &constants);
    let mut exec = LgpVm::new(&cfg);
    exec.run();

    Ok(1.0 / (1.0 + (ans - exec.mem(OUTPUT_REG)).abs()))
}

#[must_use]
pub struct ExprDataSampler {
    train: Vec<f64>,
    valid: Vec<f64>,
}

impl Default for ExprDataSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl ExprDataSampler {
    pub fn new() -> Self {
        const START: f64 = -100.0;
        const END: f64 = 100.0;
        // Strange numbers to give more diversity in decimal representation.
        const TRAIN: usize = 99;
        const VALID: usize = 9;
        let train = (0..TRAIN).map(|x| x as f64 / TRAIN as f64 * (END - START) + START).collect();
        let valid = (0..VALID).map(|x| x as f64 / VALID as f64 * (END - START) + START).collect();
        Self { train, valid }
    }
}

impl DataSampler<f64> for ExprDataSampler {
    fn train(&self, _gen: usize) -> Vec<f64> {
        self.train.clone()
    }

    fn valid(&self, _gen: usize) -> Vec<f64> {
        self.valid.clone()
    }

    fn test(&self, _gen: usize) -> Vec<f64> {
        vec![]
    }
}

pub fn expr_evolver(
    target: String,
    lgpcfg: LgpEvaluatorCfg,
    cfg: EvolveCfg,
) -> Evolver<impl Evaluator<Data = f64>> {
    lgp_fitness_evolver(
        lgpcfg.set_num_reg(NUM_REG).set_num_const(NUM_CONST).set_output_regs(&[OUTPUT_REG]),
        cfg,
        move |s: &'_ LgpState, data: &'_ f64| expr_fitness(s, *data, &target),
    )
}
