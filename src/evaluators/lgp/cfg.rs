use enumset::EnumSet;
use rand::prelude::IteratorRandom;
use rand::Rng;
use strum::IntoEnumIterator;

use crate::evaluators::lgp::vm::op::Op;
use crate::evaluators::lgp::vm::opcode::{Opcode, Operand};
use crate::ops::mutation::{mutate_creep, mutate_normal};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd)]
pub struct LgpCfg {
    num_reg: usize,
    max_code: usize,
    /// Maximum number of iterations to run.
    max_iter: usize,
    /// Max label value:
    max_label: usize,
    /// Number of significant figures the immediate value can have. This is
    /// useful to control how much precision loaded float values can be.
    imm_sf: usize,
    opcodes: EnumSet<Opcode>,
}

impl LgpCfg {
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_reg: 4,
            max_code: 10,
            max_iter: 20,
            max_label: 1,
            imm_sf: 2,
            opcodes: Opcode::iter().collect(),
        }
    }

    #[must_use]
    pub fn rand_op(&self) -> Op {
        let mut r = rand::thread_rng();
        let mut op = Op::new(self.opcodes.iter().choose(&mut r).unwrap(), [0, 0, 0]);
        for i in 0..op.data.len() {
            match op.code.operand(i) {
                Operand::None => {}
                Operand::Register => op.data[i] = r.gen_range(0..self.num_reg) as u8,
                Operand::Immediate => {
                    let (lo, hi) = Op::imm_range();
                    let v: f64 = r.gen_range(lo..=hi);
                    op.set_imm_f64(Self::round_sf(v, self.imm_sf()));
                }
                Operand::Label => op.data[i] = r.gen_range(0..self.max_label()) as u8,
            }
        }
        op
    }

    fn round_sf(v: f64, sf: usize) -> f64 {
        let digits = v.abs().log10().ceil() as i32;
        let power = 10f64.powi(digits - sf as i32);
        (v / power).round() * power
    }

    // Micro-mutation of the instruction without changing the opcode.
    pub fn mutate(&self, op: &mut Op) {
        let mut r = rand::thread_rng();
        let num_operands = op.num_operands();
        if num_operands == 0 {
            return;
        }
        let idx = r.gen_range(0..num_operands);
        match op.code.operand(idx) {
            Operand::None => {}
            Operand::Register => op.data[idx] = r.gen_range(0..self.num_reg) as u8,
            Operand::Immediate => {
                // Large/small mutation.
                let stddev = if r.gen::<bool>() { 10.0 } else { 1.0 };
                let v = mutate_normal(op.imm_value(), stddev);
                op.set_imm_f64(Self::round_sf(v, self.imm_sf()));
            }
            Operand::Label => {
                let max = self.max_label() as i32;
                op.data[idx] = mutate_creep(op.data[idx] as i32, max).clamp(0, max) as u8;
            }
        }
    }

    #[must_use]
    pub fn set_num_reg(mut self, num_reg: usize) -> Self {
        self.num_reg = num_reg;
        self
    }

    #[must_use]
    pub fn set_max_code(mut self, max_code: usize) -> Self {
        self.max_code = max_code;
        self
    }


    #[must_use]
    pub fn set_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    #[must_use]
    pub fn set_max_label(mut self, max_label: usize) -> Self {
        self.max_label = max_label;
        self
    }

    #[must_use]
    pub fn set_imm_sf(mut self, imm_sf: usize) -> Self {
        self.imm_sf = imm_sf;
        self
    }

    #[must_use]
    pub fn set_opcodes(mut self, opcodes: EnumSet<Opcode>) -> Self {
        self.opcodes = opcodes;
        self
    }

    #[must_use]
    pub fn num_reg(&self) -> usize {
        self.num_reg
    }

    #[must_use]
    pub fn max_code(&self) -> usize {
        self.max_code
    }

    #[must_use]
    pub fn max_iter(&self) -> usize {
        self.max_iter
    }

    #[must_use]
    pub fn max_label(&self) -> usize {
        self.max_label
    }

    #[must_use]
    pub fn imm_sf(&self) -> usize {
        self.imm_sf
    }

    #[must_use]
    pub fn opcodes(&self) -> EnumSet<Opcode> {
        self.opcodes
    }
}

impl Default for LgpCfg {
    fn default() -> Self {
        Self::new()
    }
}
