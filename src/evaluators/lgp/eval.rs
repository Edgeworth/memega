use std::fmt;

use rand::prelude::SliceRandom;
use rand::Rng;
use smallvec::SmallVec;

use crate::eval::Evaluator;
use crate::evaluators::lgp::cfg::LgpEvaluatorCfg;
use crate::evaluators::lgp::vm::cfg::LgpVmCfg;
use crate::evaluators::lgp::vm::disasm::lgp_disasm;
use crate::evaluators::lgp::vm::op::Op;
use crate::evaluators::lgp::vm::optimize::LgpOptimizer;
use crate::ops::crossover::crossover_kpx;
use crate::ops::distance::dist_fn;
use crate::ops::mutation::{mutate_insert, mutate_reset, mutate_scramble, mutate_swap};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct LgpState {
    ops_unopt: Vec<Op>, // Contains program code for linear genetic programming.
    num_reg: usize,
    num_const: usize,
    output_regs: SmallVec<[u8; 8]>,
}

impl fmt::Display for LgpState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops_opt = self.ops_opt();
        writeln!(
            f,
            "Unopt code len: {}, Opt code len: {}, Diff: {}",
            self.ops_unopt.len(),
            ops_opt.len(),
            self.ops_unopt.len() - ops_opt.len()
        )?;
        write!(f, "{}", lgp_disasm(&ops_opt))
    }
}

impl LgpState {
    #[must_use]
    pub fn new(ops_unopt: Vec<Op>, num_reg: usize, num_const: usize, output_regs: &[u8]) -> Self {
        Self { ops_unopt, num_reg, num_const, output_regs: output_regs.into() }
    }

    #[must_use]
    pub fn lgpvmcfg(&self, regs: &[f64], constants: &[f64]) -> LgpVmCfg {
        assert!(regs.len() == self.num_reg, "regs length mismatch");
        assert!(constants.len() == self.num_const, "constants length mismatch");
        LgpVmCfg::new().set_code(&self.ops_opt()).set_regs(regs).set_constants(constants)
    }

    #[must_use]
    pub fn num_reg(&self) -> usize {
        self.num_reg
    }

    #[must_use]
    pub fn num_const(&self) -> usize {
        self.num_const
    }

    #[must_use]
    pub fn ops_unopt(&self) -> &[Op] {
        &self.ops_unopt
    }

    #[must_use]
    pub fn ops_unopt_mut(&mut self) -> &mut Vec<Op> {
        &mut self.ops_unopt
    }

    #[must_use]
    pub fn ops_opt(&self) -> Vec<Op> {
        // Optimise code operations for the purposes of running the code.
        LgpOptimizer::new(self.ops_unopt(), &self.output_regs).optimize()
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
                crossover_kpx(s1.ops_unopt_mut(), s2.ops_unopt_mut(), 2);
            }
            _ => panic!("unknown crossover strategy"),
        };
    }

    fn mutate(&self, s: &mut LgpState, rate: f64, idx: usize) {
        let mut r = rand::thread_rng();
        if r.gen::<f64>() > rate {
            return;
        }
        let code_size = s.ops_unopt().len();
        let op = self.cfg.rand_op();
        match idx {
            0 => mutate_swap(s.ops_unopt_mut()),
            1 => mutate_insert(s.ops_unopt_mut()),
            2 => mutate_reset(s.ops_unopt_mut(), op),
            3 => mutate_scramble(s.ops_unopt_mut()),
            4 => {
                // Add new random instruction.
                if code_size < self.cfg.max_code() {
                    s.ops_unopt_mut().insert(r.gen_range(0..code_size), op);
                }
            }
            5 => {
                // Remove random instruction.
                if code_size > 1 {
                    s.ops_unopt_mut().remove(r.gen_range(0..code_size));
                }
            }
            6 => {
                // Micro-mutation
                self.cfg.mutate(s.ops_unopt_mut().choose_mut(&mut r).unwrap());
            }
            _ => panic!("unknown mutation strategy"),
        }
    }

    fn fitness(&self, _: &LgpState, _gen: usize) -> f64 {
        unimplemented!()
    }

    fn distance(&self, s1: &LgpState, s2: &LgpState) -> f64 {
        dist_fn(s1.ops_unopt(), s2.ops_unopt(), 1.0, Op::dist)
    }
}
