use num_enum::{FromPrimitive, IntoPrimitive};

const EP: f64 = 1.0e-6;

// Machine consists of N registers (up to 256) that contain f64 values.
// Note that floating point comparisons are done using an epsilon.
// Opcodes are 8 bit and have variable number of operands.
// Accessing register k will access register k % N if k >= N.
#[derive(Debug, Clone, PartialEq, PartialOrd, IntoPrimitive, FromPrimitive)]
#[repr(u8)]
pub enum Opcode {
    #[num_enum(default)]
    Nop = 0, // no operation - 0 and all uncovered opcodes
    Add = 1,   // add rx, ry: rx = rx + ry
    Sub = 2,   // sub rx, ry: rx = rx - ry
    Mul = 3,   // mul rx, ry: rx = rx * ry
    Div = 4,   // div rx, ry: rx = rx / ry - Div by zero => max value
    Abs = 5,   // abs rx: rx = |rx|
    Neg = 6,   // neg rx: rx = -rx
    Pow = 7,   // pow rx, ry: rx = rx ^ ry
    Log = 8,   // log rx: rx = ln(rx)
    Load = 9,  // load rx, f8:8: rx = immediate fixed point 8:8, little endian
    Copy = 10, // copy [rx], ry: [rx] = ry - copy ry to register indicated by rx
    Jmp = 11,  // jmp i8: pc += immediate value; relative jump
    Jlt = 12,  // jlt rx, ry, i8: if rx < ry pc += immediate; relative conditional
    Jle = 13,  // jle rx, ry, i8: if rx <= ry pc += immediate; relative conditional
    Jeq = 14,  // jeq rx, ry, i8: if rx == ry pc += immediate; relative conditional
}

// Virtual machine for lgp code.
#[derive(Debug, Clone)]
pub struct LgpExec {
    pc: usize,
    reg: Vec<f64>,
    code: Vec<u8>,
    max_iter: usize,
}

impl LgpExec {
    pub fn new(reg: &[f64], code: &[u8], max_iter: usize) -> Self {
        if reg.len() > 256 {
            panic!("cannot use more than 256 registers");
        }
        Self { pc: 0, reg: reg.to_vec(), code: code.to_vec(), max_iter }
    }

    pub fn reg(&self, idx: u8) -> f64 {
        self.reg[(idx as usize) % self.reg.len()]
    }

    fn set_reg(&mut self, idx: u8, v: f64) {
        let idx = (idx as usize) % self.reg.len();
        self.reg[idx] = v;
    }

    fn fetch(&mut self) -> u8 {
        // Check if we finished the program. Return 0 if we overrun.
        let v = if self.pc > self.code.len() { 0 } else { self.code[self.pc] };
        self.pc += 1;
        v
    }

    fn f64_to_reg(&self, v: f64) -> u8 {
        let v = (v % (self.reg.len() as f64) + self.reg.len() as f64) as usize;
        (v % self.reg.len()) as u8
    }

    fn rel_jmp(&mut self, imm: i8) {
        let pc = self.pc as i32 + imm as i32;
        let pc = pc.clamp(0, self.code.len() as i32);
        self.pc = pc as usize;
    }

    // Returns true iff finished.
    fn step(&mut self) -> bool {
        let op: Opcode = self.fetch().into();
        match op {
            Opcode::Nop => {} // Do nothing
            Opcode::Add => {
                let rx = self.fetch();
                let ry = self.fetch();
                self.set_reg(rx, self.reg(rx) + self.reg(ry));
            }
            Opcode::Sub => {
                let rx = self.fetch();
                let ry = self.fetch();
                self.set_reg(rx, self.reg(rx) - self.reg(ry));
            }
            Opcode::Mul => {
                let rx = self.fetch();
                let ry = self.fetch();
                self.set_reg(rx, self.reg(rx) * self.reg(ry));
            }
            Opcode::Div => {
                let rx = self.fetch();
                let ry = self.fetch();
                let ryv = self.reg(ry);
                // Skip division by zero.
                if ryv != 0.0 {
                    self.set_reg(rx, self.reg(rx) / ryv);
                }
            }
            Opcode::Abs => {
                let rx = self.fetch();
                self.set_reg(rx, self.reg(rx).abs());
            }
            Opcode::Neg => {
                let rx = self.fetch();
                self.set_reg(rx, -self.reg(rx));
            }
            Opcode::Pow => {
                let rx = self.fetch();
                let ry = self.fetch();
                self.set_reg(rx, self.reg(rx).powf(self.reg(ry)));
            }
            Opcode::Log => {
                let rx = self.fetch();
                let rxv = self.reg(rx);
                // Skip if it would produce NaN.
                if rxv > 0.0 {
                    self.set_reg(rx, rxv.ln());
                }
            }
            Opcode::Load => {
                let rx = self.fetch();
                let lo = self.fetch();
                let hi = self.fetch();
                self.set_reg(rx, (hi as f64) + (lo as f64) / 256.0);
            }
            Opcode::Copy => {
                let rx = self.fetch();
                let ry = self.fetch();
                self.set_reg(self.f64_to_reg(self.reg(rx)), self.reg(ry));
            }
            Opcode::Jmp => {
                let imm = self.fetch() as i8;
                self.rel_jmp(imm);
            }
            Opcode::Jlt => {
                let rx = self.fetch();
                let ry = self.fetch();
                let imm = self.fetch() as i8;
                if self.reg(rx) < self.reg(ry) - EP {
                    self.rel_jmp(imm);
                }
            }
            Opcode::Jle => {
                let rx = self.fetch();
                let ry = self.fetch();
                let imm = self.fetch() as i8;
                if self.reg(rx) <= self.reg(ry) + EP {
                    self.rel_jmp(imm);
                }
            }
            Opcode::Jeq => {
                let rx = self.fetch();
                let ry = self.fetch();
                let imm = self.fetch() as i8;
                if (self.reg(rx) - self.reg(ry)).abs() < EP {
                    self.rel_jmp(imm);
                }
            }
        }
        self.pc >= self.code.len()
    }

    pub fn run(&mut self) {
        for _ in 0..self.max_iter {
            if self.step() {
                break;
            }
        }
    }
}
