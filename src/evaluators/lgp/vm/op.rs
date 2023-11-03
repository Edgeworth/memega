use std::fmt;
use std::mem::discriminant;

use rand::Rng;
use rand::prelude::IteratorRandom;
use rand_distr::{Distribution, StandardUniform};
use strum::IntoEnumIterator;

use crate::evaluators::lgp::vm::opcode::{Opcode, Operands};

#[must_use]
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub struct Op {
    code: Opcode,
    operands: Operands,
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mnemonic = match self.code {
            Opcode::Add => "add",
            Opcode::Sub => "sub",
            Opcode::Mul => "mul",
            Opcode::Div => "div",
            Opcode::Abs => "abs",
            Opcode::Neg => "neg",
            Opcode::Pow => "pow",
            Opcode::Ln => "ln",
            Opcode::Sin => "sin",
            Opcode::Cos => "cos",
            Opcode::Load => "load",
            Opcode::Copy => "copy",
            Opcode::IfLt => "iflt",
        };
        let operands = match self.operands {
            Operands::Reg2Cmp { ra, rb } => format!("r{ra}, r{rb}"),
            Operands::Reg2Assign { ri, ra } => format!("r{ri}, r{ra}"),
            Operands::Reg3Assign { ri, ra, rb } => format!("r{ri}, r{ra}, r{rb}"),
            Operands::ImmAssign { ri, imm } => format!("r{ri}, {imm}"),
        };
        write!(f, "{mnemonic} {operands}")
    }
}

impl Distribution<Opcode> for StandardUniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Opcode {
        Opcode::iter().choose(rng).unwrap()
    }
}

impl Op {
    pub fn new(code: Opcode, operands: Operands) -> Self {
        assert!(discriminant(&code.operands()) == discriminant(&operands), "invalid operands");
        Self { code, operands }
    }

    pub fn from_code(code: Opcode) -> Self {
        Self { code, operands: code.operands() }
    }

    // Computes some distance metric between operations.
    #[must_use]
    pub fn dist(a: &Op, b: &Op) -> f64 {
        let mut d = 0.0;
        if a.code != b.code {
            d += 10.0; // Kind of arbitrary, just add a constant for different opcodes.
        }
        match (a.operands(), b.operands()) {
            (Operands::Reg2Cmp { ra: ra1, rb: rb1 }, Operands::Reg2Cmp { ra: ra2, rb: rb2 }) => {
                if ra1 != ra2 {
                    d += 1.0;
                }
                if rb1 != rb2 {
                    d += 1.0;
                }
            }
            (
                Operands::Reg2Assign { ri: ri1, ra: ra1 },
                Operands::Reg2Assign { ri: ri2, ra: ra2 },
            ) => {
                if ri1 != ri2 {
                    d += 1.0;
                }
                if ra1 != ra2 {
                    d += 1.0;
                }
            }
            (
                Operands::Reg3Assign { ri: ri1, ra: ra1, rb: rb1 },
                Operands::Reg3Assign { ri: ri2, ra: ra2, rb: rb2 },
            ) => {
                if ri1 != ri2 {
                    d += 1.0;
                }
                if ra1 != ra2 {
                    d += 1.0;
                }
                if rb1 != rb2 {
                    d += 1.0;
                }
            }
            (
                Operands::ImmAssign { ri: ri1, imm: imm1 },
                Operands::ImmAssign { ri: ri2, imm: imm2 },
            ) => {
                if ri1 != ri2 {
                    d += 1.0;
                }
                d += (imm1 - imm2).abs() as f64;
            }
            _ => {} // Opcodes were different, we already added a penalty.
        }
        d
    }

    pub fn code(&self) -> Opcode {
        self.code
    }

    pub fn operands(&self) -> Operands {
        self.operands
    }

    pub fn operands_mut(&mut self) -> &mut Operands {
        &mut self.operands
    }
}
