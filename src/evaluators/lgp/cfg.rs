use std::num::NonZeroUsize;

use enumset::EnumSet;
use rand::Rng;
use rand::prelude::IteratorRandom;
use smallvec::{SmallVec, smallvec};
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
    imm_sf: NonZeroUsize,
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
            imm_sf: NonZeroUsize::new(2).unwrap(),
            imm_range: (-100.0, 100.0),
            opcodes: Opcode::iter().collect(),
        }
    }

    pub fn rand_op(&self) -> Op {
        let mut r = rand::rng();
        let mut op = Op::from_code(self.opcodes.iter().choose(&mut r).unwrap());

        let mem_size = self.num_reg + self.num_const;
        match op.operands_mut() {
            Operands::Reg2Cmp { ra, rb } => {
                *ra = r.random_range(0..mem_size) as u8;
                *rb = r.random_range(0..mem_size) as u8;
            }
            Operands::Reg2Assign { ri, ra } => {
                *ri = r.random_range(0..self.num_reg) as u8;
                *ra = r.random_range(0..mem_size) as u8;
            }
            Operands::Reg3Assign { ri, ra, rb } => {
                *ri = r.random_range(0..self.num_reg) as u8;
                *ra = r.random_range(0..mem_size) as u8;
                *rb = r.random_range(0..mem_size) as u8;
            }
            Operands::ImmAssign { ri, imm } => {
                *ri = r.random_range(0..self.num_reg) as u8;
                let v = r.random_range(self.imm_range.0..=self.imm_range.1);
                *imm = Self::round_sf(v, self.imm_sf()) as f32;
            }
        }
        op
    }

    fn round_sf(v: f64, sf: NonZeroUsize) -> f64 {
        if !v.is_finite() || v == 0.0 {
            return v;
        }

        let exp = v.abs().log10().floor() as i32;
        let scale_exp = (sf.get() as i32 - 1 - exp).clamp(-308, 308);
        let scale = 10f64.powi(scale_exp);
        (v * scale).round() / scale
    }

    // Micro-mutation of the instruction without changing the opcode.
    pub fn mutate(&self, op: &mut Op) {
        let mut r = rand::rng();

        let mem_size = self.num_reg + self.num_const;
        match op.operands_mut() {
            Operands::Reg2Cmp { ra, rb } => {
                if r.random::<bool>() {
                    *ra = r.random_range(0..mem_size) as u8;
                } else {
                    *rb = r.random_range(0..mem_size) as u8;
                }
            }
            Operands::Reg2Assign { ri, ra } => {
                if r.random::<bool>() {
                    *ri = r.random_range(0..self.num_reg) as u8;
                } else {
                    *ra = r.random_range(0..mem_size) as u8;
                }
            }
            Operands::Reg3Assign { ri, ra, rb } => match r.random_range(0..3) {
                0 => {
                    *ri = r.random_range(0..self.num_reg) as u8;
                }
                1 => {
                    *ra = r.random_range(0..mem_size) as u8;
                }
                2 => {
                    *rb = r.random_range(0..mem_size) as u8;
                }
                _ => unreachable!(),
            },
            Operands::ImmAssign { ri, imm } => {
                if r.random::<bool>() {
                    *ri = r.random_range(0..self.num_reg) as u8;
                } else {
                    // Large/small mutation.
                    let range = self.imm_range.1 - self.imm_range.0;
                    let stddev = if r.random::<bool>() { range.sqrt() } else { range.log10() };
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

    pub fn set_imm_sf(mut self, imm_sf: NonZeroUsize) -> Self {
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
    pub fn imm_sf(&self) -> NonZeroUsize {
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

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn round_sf_zero() {
        let sf = NonZeroUsize::new(2).unwrap();
        assert_relative_eq!(0.0, LgpEvaluatorCfg::round_sf(0.0, sf));
    }

    #[test]
    fn round_sf_basic() {
        let sf = NonZeroUsize::new(2).unwrap();
        assert_relative_eq!(1.2, LgpEvaluatorCfg::round_sf(1.2345, sf));
        assert_relative_eq!(1200.0, LgpEvaluatorCfg::round_sf(1234.5, sf));
    }

    #[test]
    fn round_sf_subnormal_no_panic() {
        let sf = NonZeroUsize::new(2).unwrap();
        let v = LgpEvaluatorCfg::round_sf(1.0e-320, sf);
        assert!(v.is_finite());
    }
}
