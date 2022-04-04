use std::fmt;

use rand::prelude::SliceRandom;
use rand::Rng;

use crate::eval::Evaluator;
use crate::evaluators::lgp::cfg::LgpEvaluatorCfg;
use crate::evaluators::lgp::vm::cfg::LgpVmCfg;
use crate::evaluators::lgp::vm::disasm::lgp_disasm;
use crate::evaluators::lgp::vm::op::Op;
use crate::ops::crossover::crossover_kpx;
use crate::ops::distance::dist_fn;
use crate::ops::mutation::{mutate_insert, mutate_reset, mutate_scramble, mutate_swap};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct LgpState {
    pub ops: Vec<Op>, // Contains program code for linear genetic programming.
    pub cfg: LgpEvaluatorCfg,
}

impl fmt::Display for LgpState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&lgp_disasm(&self.ops))
    }
}

impl LgpState {
    #[must_use]
    pub fn new(ops: Vec<Op>, cfg: LgpEvaluatorCfg) -> Self {
        Self { ops, cfg }
    }

    #[must_use]
    pub fn lgpvmcfg(&self, regs: &[f64], constants: &[f64]) -> LgpVmCfg {
        assert!(regs.len() == self.cfg.num_reg(), "regs length mismatch");
        assert!(constants.len() == self.cfg.num_const(), "constants length mismatch");
        LgpVmCfg::new().set_code(&self.ops).set_regs(regs).set_constants(constants)
    }
}

pub struct LgpEvaluator {
    cfg: LgpEvaluatorCfg,
}

impl LgpEvaluator {
    #[must_use]
    pub fn new(cfg: LgpEvaluatorCfg) -> Self {
        Self { cfg }
    }
}

impl Evaluator for LgpEvaluator {
    type State = LgpState;
    const NUM_CROSSOVER: usize = 2;
    const NUM_MUTATION: usize = 7;

    fn crossover(&self, s1: &mut LgpState, s2: &mut LgpState, idx: usize) {
        match idx {
            0 => {} // Do nothing.
            1 => {
                // Two point crossover.
                crossover_kpx(&mut s1.ops, &mut s2.ops, 2);
            }
            _ => panic!("unknown crossover strategy"),
        };
    }

    fn mutate(&self, s: &mut LgpState, rate: f64, idx: usize) {
        let mut r = rand::thread_rng();
        if r.gen::<f64>() > rate {
            return;
        }
        let code_size = s.ops.len();
        let op = self.cfg.rand_op();
        match idx {
            0 => mutate_swap(&mut s.ops),
            1 => mutate_insert(&mut s.ops),
            2 => mutate_reset(&mut s.ops, op),
            3 => mutate_scramble(&mut s.ops),
            4 => {
                // Add new random instruction.
                if code_size < self.cfg.max_code() {
                    s.ops.insert(r.gen_range(0..code_size), op);
                }
            }
            5 => {
                // Remove random instruction.
                if code_size > 1 {
                    s.ops.remove(r.gen_range(0..code_size));
                }
            }
            6 => {
                // Micro-mutation
                self.cfg.mutate(s.ops.choose_mut(&mut r).unwrap());
            }
            _ => panic!("unknown mutation strategy"),
        }
    }

    fn fitness(&self, _: &LgpState, _gen: usize) -> f64 {
        unimplemented!()
    }

    fn distance(&self, s1: &LgpState, s2: &LgpState) -> f64 {
        dist_fn(&s1.ops, &s2.ops, 1.0, Op::dist)
    }
}
