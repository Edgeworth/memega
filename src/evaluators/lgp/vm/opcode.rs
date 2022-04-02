use enumset::EnumSetType;
use strum_macros::{Display, EnumIter};

// Machine consists of N registers (up to 256) that contain f64 values.
// Note that floating point comparisons are done using an epsilon.
// Opcodes are 8 bit and have variable number of operands.
// If a opcode isn't in the range of opcodes, it is mapped onto it using modulo.
// Accessing register k will access register k % N if k >= N.
#[derive(EnumSetType, Debug, Display, PartialOrd, EnumIter)]
pub enum Opcode {
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
    Jlt,   // jlt rx, ry, u8: if rx < ry pc = address of first label with name u8, if it exists
    Jle,   // jle rx, ry, u8: if rx <= ry pc = address of first label with name u8, if it exists
    Jeq,   // jeq rx, ry, u8: if rx == ry pc = address of first label with name u8, if it exists
    Jmp,   // jmp u8: pc = address of first label with name u8, if it exists
    Label, // Label that jump instructions jump to
}

impl Opcode {
    #[must_use]
    pub fn operand(&self, idx: usize) -> Operand {
        match self {
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
                    Operand::Label
                } else {
                    Operand::Register
                }
            }
            // Single label operands:
            Opcode::Jmp | Opcode::Label => {
                if idx == 0 {
                    Operand::Label
                } else {
                    Operand::None
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
    Label,
}
