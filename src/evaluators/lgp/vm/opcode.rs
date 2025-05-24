use enumset::EnumSetType;
use smallvec::{SmallVec, smallvec};
use strum_macros::{Display, EnumIter};

#[must_use]
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Operands {
    /// Compare two registers.
    Reg2Cmp { ra: u8, rb: u8 },
    /// Assign function of register to another.
    Reg2Assign { ri: u8, ra: u8 },
    /// Assign function of two registers to another.
    Reg3Assign { ri: u8, ra: u8, rb: u8 },
    /// Assign immediate value to register.
    ImmAssign { ri: u8, imm: f32 },
}

impl Operands {
    #[must_use]
    pub fn input_regs(&self) -> SmallVec<[u8; 2]> {
        match *self {
            Operands::Reg2Assign { ra, .. } => smallvec![ra],
            Operands::Reg3Assign { ra, rb, .. } | Operands::Reg2Cmp { ra, rb } => smallvec![ra, rb],
            Operands::ImmAssign { .. } => smallvec![],
        }
    }

    #[must_use]
    pub fn output_regs(&self) -> SmallVec<[u8; 1]> {
        match *self {
            Operands::Reg2Cmp { .. } => smallvec![],
            Operands::Reg2Assign { ri, .. }
            | Operands::Reg3Assign { ri, .. }
            | Operands::ImmAssign { ri, .. } => smallvec![ri],
        }
    }
}

/// Machine consists of N registers (up to 256) that contain f64 values.
/// Opcodes are 8 bit and have variable number of operands.
#[must_use]
#[derive(EnumSetType, Debug, Display, PartialOrd, EnumIter)]
pub enum Opcode {
    // Arithmetic - three register assignments:
    Add, // add ri, ra, rb: rx = rx + ry
    Sub, // sub ri, ra, rb: ri = ra - rb
    Mul, // mul ri, ra, rb: ri = ra * rb
    Div, // div ri, ra, rb: ri = ra / rb - Div by zero is ignored.
    Pow, // pow ri, ra, rb: ri = ra ^ rb - Infinite value is ignored.

    // Arithmetic - two register assignments:
    Abs, // abs ri, ra: ri = |ra|
    Neg, // neg ri, ra: ri = -ra
    Ln,  // ln ri, ra: ri = ln(ra)
    Sin, // sin ri, ra: ri = sin(ra)
    Cos, // cos ri, ra: ri = cos(ra)

    // Loading:
    Load, // load ri, f64: ri = floating point value
    Copy, // copy ri, ra: ri = ra - direct copy

    // Branching:
    IfLt, // iflt ra, rb: if ra < rb execute next instruction. Can be chained.
}

impl Opcode {
    pub fn operands(&self) -> Operands {
        match self {
            // Three reg assign
            Opcode::Add | Opcode::Sub | Opcode::Mul | Opcode::Div | Opcode::Pow => {
                Operands::Reg3Assign { ri: 0, ra: 0, rb: 0 }
            }
            // Two reg assign:
            Opcode::Abs | Opcode::Neg | Opcode::Ln | Opcode::Sin | Opcode::Cos | Opcode::Copy => {
                Operands::Reg2Assign { ri: 0, ra: 0 }
            }
            // Immediate assign
            Opcode::Load => Operands::ImmAssign { ri: 0, imm: 0.0 },
            // Two reg compare:
            Opcode::IfLt => Operands::Reg2Cmp { ra: 0, rb: 0 },
        }
    }

    #[must_use]
    pub fn is_branch(&self) -> bool {
        matches!(self, Opcode::IfLt)
    }
}
