use std::fmt;

use rand::prelude::IteratorRandom;
use rand::Rng;
use rand_distr::{Distribution, Standard};
use strum::IntoEnumIterator;

use crate::evaluators::lgp::vm::opcode::{Opcode, Operand};

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Op {
    pub code: Opcode,
    pub data: [u8; 3], // Currently up to 3 bytes.
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rx = self.data[0];
        let ry = self.data[1];
        match self.code {
            Opcode::Add => write!(f, "add r{}, r{}", rx, ry),
            Opcode::Sub => write!(f, "sub r{}, r{}", rx, ry),
            Opcode::Mul => write!(f, "mul r{}, r{}", rx, ry),
            Opcode::Div => write!(f, "div r{}, r{}", rx, ry),
            Opcode::Abs => write!(f, "abs r{}", rx),
            Opcode::Neg => write!(f, "neg r{}", rx),
            Opcode::Pow => write!(f, "pow r{}, r{}", rx, ry),
            Opcode::Log => write!(f, "ln r{}", rx),
            Opcode::Load => write!(f, "load r{}, {}", rx, self.imm_value()),
            Opcode::Copy => write!(f, "copy r{}, r{}", rx, ry),
            Opcode::IndirectCopy => write!(f, "ind [r{}], r{}", rx, ry),
            Opcode::Jlt => write!(f, "jlt r{}, r{}, {}", rx, ry, self.label()),
            Opcode::Jle => write!(f, "jle r{}, r{}, {}", rx, ry, self.label()),
            Opcode::Jeq => write!(f, "jeq r{}, r{}, {}", rx, ry, self.label()),
            Opcode::Jmp => write!(f, "jmp {}", self.label()),
            Opcode::Label => write!(f, "lbl {}", rx),
        }
    }
}

impl Distribution<Opcode> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Opcode {
        Opcode::iter().choose(rng).unwrap()
    }
}

impl Op {
    #[must_use]
    pub fn new(op: Opcode, data: [u8; 3]) -> Self {
        Self { code: op, data }
    }

    #[must_use]
    pub fn imm_range() -> (f64, f64) {
        const RANGE: f64 = u8::MAX as f64;
        (-RANGE / 2.0, RANGE / 2.0)
    }

    #[must_use]
    pub fn imm_value(&self) -> f64 {
        assert!(
            self.code.operand(1) == Operand::Immediate
                && self.code.operand(2) == Operand::Immediate,
            "Opcode {} does not take an immediate value",
            self.code
        );

        let (lo, _) = Self::imm_range();
        let frac = self.data[1] as f64 / u8::MAX as f64;
        let int = self.data[2] as f64 + lo;
        int + frac
    }

    pub fn set_imm_f64(&mut self, v: f64) {
        assert!(
            self.code.operand(1) == Operand::Immediate
                && self.code.operand(2) == Operand::Immediate,
            "Opcode {} does not take an immediate value",
            self.code
        );

        let (lo, hi) = Self::imm_range();
        let v = v.clamp(lo, hi);
        let v = v - lo;
        self.data[1] = (v.fract() * u8::MAX as f64).trunc() as u8;
        self.data[2] = v.trunc() as u8;
    }

    #[must_use]
    pub fn label(&self) -> u8 {
        match self.code {
            Opcode::Jmp | Opcode::Label => self.data[0],
            Opcode::Jlt | Opcode::Jle | Opcode::Jeq => self.data[2],
            _ => panic!("Opcode does not have a label: {}", self.code),
        }
    }

    #[must_use]
    pub fn num_operands(&self) -> usize {
        for i in 0..self.data.len() {
            if self.code.operand(i) == Operand::None {
                return i;
            }
        }
        self.data.len()
    }

    // Computes some distance metric between operations.
    #[must_use]
    pub fn dist(a: &Op, b: &Op) -> f64 {
        let mut d = 0.0;
        if a.code != b.code {
            d += 100.0; // Kind of arbitrary, just add a constant for different opcodes.
        }
        for (&a, &b) in a.data.iter().zip(b.data.iter()) {
            d += (a as f64 - b as f64).abs();
        }
        d
    }
}
