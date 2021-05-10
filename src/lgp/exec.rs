use crate::lgp::op::{Op, Opcode};

const EP: f64 = 1.0e-6;

// Virtual machine for lgp code.
#[derive(Debug, Clone)]
pub struct LgpExec {
    pc: usize,
    reg: Vec<f64>,
    code: Vec<Op>,
    max_iter: usize,
}

impl LgpExec {
    pub fn new(reg: &[f64], code: &[Op], max_iter: usize) -> Self {
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

    fn fetch(&mut self) -> Option<Op> {
        // Check if we finished the program. Return 0 if we overrun.
        let v = if self.pc >= self.code.len() { None } else { Some(self.code[self.pc]) };
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
        if let Some(op) = self.fetch() {
            let rx = op.data[0];
            let ry = op.data[1];
            match op.op {
                Opcode::Nop => {} // Do nothing
                Opcode::Add => {
                    let v = self.reg(rx) + self.reg(ry);
                    if v.is_finite() {
                        self.set_reg(rx, v);
                    }
                }
                Opcode::Sub => {
                    let v = self.reg(rx) - self.reg(ry);
                    if v.is_finite() {
                        self.set_reg(rx, v);
                    }
                }
                Opcode::Mul => {
                    let v = self.reg(rx) * self.reg(ry);
                    if v.is_finite() {
                        self.set_reg(rx, v);
                    }
                }
                Opcode::Div => {
                    let v = self.reg(rx) / self.reg(ry);
                    if v.is_finite() {
                        self.set_reg(rx, v);
                    }
                }
                Opcode::Abs => {
                    self.set_reg(rx, self.reg(rx).abs());
                }
                Opcode::Neg => {
                    self.set_reg(rx, -self.reg(rx));
                }
                Opcode::Pow => {
                    let v = self.reg(rx).powf(self.reg(ry));
                    if v.is_finite() {
                        self.set_reg(rx, v);
                    }
                }
                Opcode::Log => {
                    let v = self.reg(rx).ln();
                    if v.is_finite() {
                        self.set_reg(rx, v);
                    }
                }
                Opcode::Load => {
                    let lo = op.data[1];
                    let hi = op.data[2];
                    self.set_reg(rx, (hi as f64) + (lo as f64) / 256.0);
                }
                Opcode::Copy => {
                    self.set_reg(self.f64_to_reg(self.reg(rx)), self.reg(ry));
                }
                Opcode::Jlt => {
                    let imm = op.data[2] as i8;
                    if self.reg(rx) < self.reg(ry) - EP {
                        self.rel_jmp(imm);
                    }
                }
                Opcode::Jle => {
                    let imm = op.data[2] as i8;
                    if self.reg(rx) <= self.reg(ry) + EP {
                        self.rel_jmp(imm);
                    }
                }
                Opcode::Jeq => {
                    let imm = op.data[2] as i8;
                    if (self.reg(rx) - self.reg(ry)).abs() < EP {
                        self.rel_jmp(imm);
                    }
                }
            }
            false
        } else {
            true
        }
    }

    pub fn run(&mut self) {
        for _ in 0..self.max_iter {
            if self.step() {
                break;
            }
        }
    }
}
