use std::fmt;

use enumset::EnumSetType;
use rand::prelude::IteratorRandom;
use rand::Rng;
use rand_distr::{Distribution, Standard};
use strum::IntoEnumIterator;
use strum_macros::EnumIter;

// Machine consists of N registers (up to 256) that contain f64 values.
// Note that floating point comparisons are done using an epsilon.
// Opcodes are 8 bit and have variable number of operands.
// If a opcode isn't in the range of opcodes, it is mapped onto it using modulo.
// Accessing register k will access register k % N if k >= N.
#[derive(EnumSetType, Debug, PartialOrd, EnumIter)]
pub enum Opcode {
    Nop,          // no operation - 0
    Add,          // add rx, ry: rx = rx + ry
    Sub,          // sub rx, ry: rx = rx - ry
    Mul,          // mul rx, ry: rx = rx * ry
    Div,          // div rx, ry: rx = rx / ry - Div by zero => max value
    Abs,          // abs rx: rx = |rx|
    Neg,          // neg rx: rx = -rx
    Pow,          // pow rx, ry: rx = rx ^ ry - Require rx >= 0.0
    Log,          // log rx: rx = ln(rx)
    Load,         // load rx, f8:8: rx = immediate fixed point 8:8, little endian
    IndirectCopy, // copy [rx], ry: [rx] = ry - copy ry to register indicated by rx
    Jlt,          // jlt rx, ry, i8: if rx < ry pc += immediate; relative conditional
    Jle,          // jle rx, ry, i8: if rx <= ry pc += immediate; relative conditional
    Jeq,          // jeq rx, ry, i8: if rx == ry pc += immediate; relative conditional
}

impl Opcode {
    #[must_use]
    pub fn operand(&self, idx: usize) -> Operand {
        match self {
            // Zero operands
            Opcode::Nop => Operand::None,
            // Two reg operands
            Opcode::Add
            | Opcode::Sub
            | Opcode::Mul
            | Opcode::Div
            | Opcode::IndirectCopy
            | Opcode::Pow => {
                if idx <= 1 {
                    Operand::Register
                } else {
                    Operand::None
                }
            }
            // One reg operand
            Opcode::Abs | Opcode::Neg | Opcode::Log => {
                if idx == 0 {
                    Operand::Register
                } else {
                    Operand::None
                }
            }
            // One reg operand, two immediate
            Opcode::Load => {
                if idx == 0 {
                    Operand::Register
                } else {
                    Operand::Immediate
                }
            }
            // Two reg operand, one immediate
            Opcode::Jlt | Opcode::Jle | Opcode::Jeq => {
                if idx == 2 {
                    Operand::Relative
                } else {
                    Operand::Register
                }
            }
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Operand {
    None,
    Register,
    Immediate,
    Relative, // Relative jump
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Op {
    pub op: Opcode,
    pub data: [u8; 3], // Currently up to 3 bytes.
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rx = self.data[0];
        let ry = self.data[1];
        match self.op {
            Opcode::Nop => f.write_fmt(format_args!("nop")),
            Opcode::Add => f.write_fmt(format_args!("add r{}, r{}", rx, ry)),
            Opcode::Sub => f.write_fmt(format_args!("sub r{}, r{}", rx, ry)),
            Opcode::Mul => f.write_fmt(format_args!("mul r{}, r{}", rx, ry)),
            Opcode::Div => f.write_fmt(format_args!("div r{}, r{}", rx, ry)),
            Opcode::Abs => f.write_fmt(format_args!("abs r{}", rx)),
            Opcode::Neg => f.write_fmt(format_args!("neg r{}", rx)),
            Opcode::Pow => f.write_fmt(format_args!("pow r{}, r{}", rx, ry)),
            Opcode::Log => f.write_fmt(format_args!("ln r{}", rx)),
            Opcode::Load => {
                let lo = self.data[1];
                let hi = self.data[2];
                f.write_fmt(format_args!("load r{}, {}", rx, (hi as f64) + (lo as f64) / 256.0))
            }
            Opcode::IndirectCopy => f.write_fmt(format_args!("mov [r{}], r{}", rx, ry)),
            Opcode::Jlt => {
                let imm = self.data[2] as i8;
                f.write_fmt(format_args!("jlt r{}, r{}, {}", rx, ry, imm))
            }
            Opcode::Jle => {
                let imm = self.data[2] as i8;
                f.write_fmt(format_args!("jle r{}, r{}, {}", rx, ry, imm))
            }
            Opcode::Jeq => {
                let imm = self.data[2] as i8;
                f.write_fmt(format_args!("jeq r{}, r{}, {}", rx, ry, imm))
            }
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
        Self { op, data }
    }

    #[must_use]
    pub fn num_operands(&self) -> usize {
        for i in 0..self.data.len() {
            if self.op.operand(i) == Operand::None {
                return i;
            }
        }
        self.data.len()
    }

    // Computes some distance metric between operations.
    #[must_use]
    pub fn dist(a: &Op, b: &Op) -> f64 {
        let mut d = 0.0;
        if a.op != b.op {
            d += 100.0; // Kind of arbitrary, just add a constant for different opcodes.
        }
        for (&a, &b) in a.data.iter().zip(b.data.iter()) {
            d += (a as f64 - b as f64).abs();
        }
        d
    }
}
