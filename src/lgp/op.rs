use std::fmt;

use num_enum::{IntoPrimitive, TryFromPrimitive};
use rand::Rng;
use rand_distr::{Distribution, Standard};
use strum::EnumCount;
use strum_macros::EnumCount;

use crate::ops::mutation::mutate_creep;

// Machine consists of N registers (up to 256) that contain f64 values.
// Note that floating point comparisons are done using an epsilon.
// Opcodes are 8 bit and have variable number of operands.
// If a opcode isn't in the range of opcodes, it is mapped onto it using modulo.
// Accessing register k will access register k % N if k >= N.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, IntoPrimitive, TryFromPrimitive, EnumCount)]
#[repr(u8)]
pub enum Opcode {
    Nop = 0,   // no operation - 0
    Add = 1,   // add rx, ry: rx = rx + ry
    Sub = 2,   // sub rx, ry: rx = rx - ry
    Mul = 3,   // mul rx, ry: rx = rx * ry
    Div = 4,   // div rx, ry: rx = rx / ry - Div by zero => max value
    Abs = 5,   // abs rx: rx = |rx|
    Neg = 6,   // neg rx: rx = -rx
    Pow = 7,   // pow rx, ry: rx = rx ^ ry - Require rx >= 0.0
    Log = 8,   // log rx: rx = ln(rx)
    Load = 9,  // load rx, f8:8: rx = immediate fixed point 8:8, little endian
    Copy = 10, // copy [rx], ry: [rx] = ry - copy ry to register indicated by rx
    Jlt = 11,  // jlt rx, ry, i8: if rx < ry pc += immediate; relative conditional
    Jle = 12,  // jle rx, ry, i8: if rx <= ry pc += immediate; relative conditional
    Jeq = 13,  // jeq rx, ry, i8: if rx == ry pc += immediate; relative conditional
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
            Opcode::Add => f.write_fmt(format_args!("r{} += r{}", rx, ry)),
            Opcode::Sub => f.write_fmt(format_args!("r{} -= r{}", rx, ry)),
            Opcode::Mul => f.write_fmt(format_args!("r{} *= r{}", rx, ry)),
            Opcode::Div => f.write_fmt(format_args!("r{} /= r{}", rx, ry)),
            Opcode::Abs => f.write_fmt(format_args!("r{} = |r{}|", rx, rx)),
            Opcode::Neg => f.write_fmt(format_args!("r{} = -r{}", rx, rx)),
            Opcode::Pow => f.write_fmt(format_args!("r{} = r{} ** r{}", rx, rx, ry)),
            Opcode::Log => f.write_fmt(format_args!("r{} = ln(r{})", rx, rx)),
            Opcode::Load => {
                let lo = self.data[1];
                let hi = self.data[2];
                f.write_fmt(format_args!("[r{}] = {}", rx, (hi as f64) + (lo as f64) / 256.0))
            }
            Opcode::Copy => f.write_fmt(format_args!("r{} = r{}", rx, ry)),
            Opcode::Jlt => {
                let imm = self.data[2] as i8;
                f.write_fmt(format_args!("if r{} < r{}: jmp {}", rx, ry, imm))
            }
            Opcode::Jle => {
                let imm = self.data[2] as i8;
                f.write_fmt(format_args!("if r{} <= r{}: jmp {}", rx, ry, imm))
            }
            Opcode::Jeq => {
                let imm = self.data[2] as i8;
                f.write_fmt(format_args!("if r{} == r{}: jmp {}", rx, ry, imm))
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
    pub fn new(op: Opcode, data: [u8; 3]) -> Self {
        Self { op, data }
    }

    pub fn rand(num_reg: usize) -> Self {
        let mut r = rand::thread_rng();
        let opcode = r.gen::<Opcode>();
        let rx = r.gen_range(0..num_reg) as u8;
        let ry = r.gen_range(0..num_reg) as u8;
        let imm0 = r.gen::<u8>();
        let imm1 = r.gen::<u8>();
        let data = match opcode {
            Opcode::Nop => [0, 0, 0],
            Opcode::Add => [rx, ry, 0],
            Opcode::Sub => [rx, ry, 0],
            Opcode::Mul => [rx, ry, 0],
            Opcode::Div => [rx, ry, 0],
            Opcode::Abs => [rx, 0, 0],
            Opcode::Neg => [rx, 0, 0],
            Opcode::Pow => [rx, ry, 0],
            Opcode::Log => [rx, 0, 0],
            Opcode::Load => [rx, imm0, imm1],
            Opcode::Copy => [rx, ry, 0],
            Opcode::Jlt => [rx, ry, imm0],
            Opcode::Jle => [rx, ry, imm0],
            Opcode::Jeq => [rx, ry, imm0],
        };
        Op::new(opcode, data)
    }

    pub fn num_operands(&self) -> usize {
        match self.op {
            Opcode::Nop => 0,
            Opcode::Add => 2,
            Opcode::Sub => 2,
            Opcode::Mul => 2,
            Opcode::Div => 2,
            Opcode::Abs => 1,
            Opcode::Neg => 1,
            Opcode::Pow => 2,
            Opcode::Log => 1,
            Opcode::Load => 3,
            Opcode::Copy => 2,
            Opcode::Jlt => 3,
            Opcode::Jle => 3,
            Opcode::Jeq => 3,
        }
    }

    // Micro-mutation of the instruction without changing the opcode.
    pub fn mutate(&mut self) {
        let mut r = rand::thread_rng();
        let num_operands = self.num_operands();
        if num_operands == 0 {
            return;
        }
        let idx = r.gen_range(0..num_operands);
        self.data[idx] = mutate_creep(self.data[idx], 64);
    }

    // Computes some distance metric between operations.
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
