use enumset::EnumSetType;
use strum_macros::{Display, EnumIter};

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

/// Machine consists of N registers (up to 256) that contain f64 values.
/// Opcodes are 8 bit and have variable number of operands.
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
    #[must_use]
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
}
