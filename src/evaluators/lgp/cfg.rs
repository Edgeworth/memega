use enumset::EnumSet;
use rand::prelude::IteratorRandom;
use rand::Rng;
use smallvec::{smallvec, SmallVec};
use strum::IntoEnumIterator;

use crate::evaluators::lgp::vm::op::Op;
use crate::evaluators::lgp::vm::opcode::{Opcode, Operands};
use crate::ops::mutation::mutate_normal;

#[must_use]
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct LgpEvaluatorCfg {
    num_reg: usize,
    num_const: usize,
    output_regs: SmallVec<[u8; 8]>,
    max_code: usize,
    /// Number of significant figures the immediate value can have. This is
    /// useful to control how much precision loaded float values can be.
    imm_sf: usize,
    /// Range randomly generated floating point numbers can be in.
    imm_range: (f64, f64),
    opcodes: EnumSet<Opcode>,
}

impl LgpEvaluatorCfg {
    pub fn new() -> Self {
        Self {
            num_reg: 4,
            num_const: 0,
            output_regs: smallvec![0],
            max_code: 100,
            imm_sf: 2,
            imm_range: (-100.0, 100.0),
            opcodes: Opcode::iter().collect(),
        }
    }

    pub fn rand_op(&self) -> Op {
        let mut r = rand::thread_rng();
        let mut op = Op::from_code(self.opcodes.iter().choose(&mut r).unwrap());

        let mem_size = self.num_reg + self.num_const;
        match op.operands_mut() {
            Operands::Reg2Cmp { ra, rb } => {
                *ra = r.gen_range(0..mem_size) as u8;
                *rb = r.gen_range(0..mem_size) as u8;
            }
            Operands::Reg2Assign { ri, ra } => {
                *ri = r.gen_range(0..self.num_reg) as u8;
                *ra = r.gen_range(0..mem_size) as u8;
            }
            Operands::Reg3Assign { ri, ra, rb } => {
                *ri = r.gen_range(0..self.num_reg) as u8;
                *ra = r.gen_range(0..mem_size) as u8;
                *rb = r.gen_range(0..mem_size) as u8;
            }
            Operands::ImmAssign { ri, imm } => {
                *ri = r.gen_range(0..self.num_reg) as u8;
                let v = r.gen_range(self.imm_range.0..=self.imm_range.1);
                *imm = Self::round_sf(v, self.imm_sf()) as f32;
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

        let mem_size = self.num_reg + self.num_const;
        match op.operands_mut() {
            Operands::Reg2Cmp { ra, rb } => {
                if r.gen::<bool>() {
                    *ra = r.gen_range(0..mem_size) as u8;
                } else {
                    *rb = r.gen_range(0..mem_size) as u8;
                }
            }
            Operands::Reg2Assign { ri, ra } => {
                if r.gen::<bool>() {
                    *ri = r.gen_range(0..self.num_reg) as u8;
                } else {
                    *ra = r.gen_range(0..mem_size) as u8;
                }
            }
            Operands::Reg3Assign { ri, ra, rb } => {
                match r.gen_range(0..3) {
                    0 => {
                        *ri = r.gen_range(0..self.num_reg) as u8;
                    }
                    1 => {
                        *ra = r.gen_range(0..mem_size) as u8;
                    }
                    2 => {
                        *rb = r.gen_range(0..mem_size) as u8;
                    }
                    _ => unreachable!(),
                };
            }
            Operands::ImmAssign { ri, imm } => {
                if r.gen::<bool>() {
                    *ri = r.gen_range(0..self.num_reg) as u8;
                } else {
                    // Large/small mutation.
                    let range = self.imm_range.1 - self.imm_range.0;
                    let stddev = if r.gen::<bool>() { range.sqrt() } else { range.log10() };
                    let v = mutate_normal(*imm as f64, stddev);
                    *imm = Self::round_sf(v, self.imm_sf) as f32;
                }
            }
        }
    }

    pub fn set_num_reg(mut self, num_reg: usize) -> Self {
        self.num_reg = num_reg;
        self
    }

    pub fn set_num_const(mut self, num_const: usize) -> Self {
        self.num_const = num_const;
        self
    }

    pub fn set_output_regs(mut self, output_regs: &[u8]) -> Self {
        self.output_regs = output_regs.into();
        self
    }

    pub fn set_max_code(mut self, max_code: usize) -> Self {
        self.max_code = max_code;
        self
    }

    pub fn set_imm_sf(mut self, imm_sf: usize) -> Self {
        self.imm_sf = imm_sf;
        self
    }

    pub fn set_imm_range(mut self, imm_range: (f64, f64)) -> Self {
        self.imm_range = imm_range;
        self
    }

    pub fn set_opcodes(mut self, opcodes: EnumSet<Opcode>) -> Self {
        self.opcodes = opcodes;
        self
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
    pub fn output_regs(&self) -> &[u8] {
        &self.output_regs
    }

    #[must_use]
    pub fn max_code(&self) -> usize {
        self.max_code
    }

    #[must_use]
    pub fn imm_sf(&self) -> usize {
        self.imm_sf
    }

    #[must_use]
    pub fn imm_range(&self) -> (f64, f64) {
        self.imm_range
    }

    #[must_use]
    pub fn opcodes(&self) -> EnumSet<Opcode> {
        self.opcodes
    }
}

impl Default for LgpEvaluatorCfg {
    fn default() -> Self {
        Self::new()
    }
}
