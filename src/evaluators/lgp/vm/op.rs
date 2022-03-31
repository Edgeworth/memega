use std::fmt;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use rand::Rng;
use rand_distr::{Distribution, Standard};
use strum::EnumCount as EnumCountTrait;
use strum_macros::{EnumCount, EnumIter};

use crate::ops::mutation::mutate_creep;

// Machine consists of N registers (up to 256) that contain f64 values.
// Note that floating point comparisons are done using an epsilon.
// Opcodes are 8 bit and have variable number of operands.
// If a opcode isn't in the range of opcodes, it is mapped onto it using modulo.
// Accessing register k will access register k % N if k >= N.
#[derive(
    Debug, Copy, Clone, PartialEq, PartialOrd, IntoPrimitive, TryFromPrimitive, EnumCount, EnumIter,
)]
#[repr(u8)]
pub enum Opcode {
    Nop = 0,           // no operation - 0
    Add = 1,           // add rx, ry: rx = rx + ry
    Sub = 2,           // sub rx, ry: rx = rx - ry
    Mul = 3,           // mul rx, ry: rx = rx * ry
    Div = 4,           // div rx, ry: rx = rx / ry - Div by zero => max value
    Abs = 5,           // abs rx: rx = |rx|
    Neg = 6,           // neg rx: rx = -rx
    Pow = 7,           // pow rx, ry: rx = rx ^ ry - Require rx >= 0.0
    Log = 8,           // log rx: rx = ln(rx)
    Load = 9,          // load rx, f8:8: rx = immediate fixed point 8:8, little endian
    IndirectCopy = 10, // copy [rx], ry: [rx] = ry - copy ry to register indicated by rx
    Jlt = 11,          // jlt rx, ry, i8: if rx < ry pc += immediate; relative conditional
    Jle = 12,          // jle rx, ry, i8: if rx <= ry pc += immediate; relative conditional
    Jeq = 13,          // jeq rx, ry, i8: if rx == ry pc += immediate; relative conditional
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
        Opcode::try_from_primitive(rng.gen_range(0..Opcode::COUNT) as u8).unwrap()
    }
}

impl Op {
    #[must_use]
    pub fn new(op: Opcode, data: [u8; 3]) -> Self {
        Self { op, data }
    }

    #[must_use]
    pub fn rand(op: Opcode, num_reg: usize, code_size: usize) -> Self {
        let mut r = rand::thread_rng();
        let mut data = [0, 0, 0];
        let code_size = code_size.clamp(0, i8::MAX as usize) as i32;
        for (i, v) in data.iter_mut().enumerate() {
            match op.operand(i) {
                Operand::None => {}
                Operand::Register => *v = r.gen_range(0..num_reg) as u8,
                Operand::Immediate => *v = r.gen::<u8>(),
                Operand::Relative => *v = r.gen_range(-code_size..=code_size) as u8,
            }
        }
        Op::new(op, data)
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

    // Micro-mutation of the instruction without changing the opcode.
    pub fn mutate(&mut self, num_reg: usize, code_size: usize) {
        let mut r = rand::thread_rng();
        let num_operands = self.num_operands();
        if num_operands == 0 {
            return;
        }
        let idx = r.gen_range(0..num_operands);
        let code_size = code_size.clamp(0, i8::MAX as usize) as i8;
        match self.op.operand(idx) {
            Operand::None => {}
            Operand::Register => self.data[idx] = r.gen_range(0..num_reg) as u8,
            Operand::Immediate => self.data[idx] = mutate_creep(self.data[idx], 64),
            Operand::Relative => {
                self.data[idx] = mutate_creep(self.data[idx] as i8, code_size)
                    .clamp(-code_size, code_size) as u8;
            }
        }
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
