use crate::evaluators::lgp::vm::op::Op;
use crate::evaluators::lgp::vm::opcode::Opcode;

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
    #[must_use]
    pub fn new(reg: &[f64], code: &[Op], max_iter: usize) -> Self {
        assert!(reg.len() <= 256, "cannot use more than 256 registers");
        Self { pc: 0, reg: reg.to_vec(), code: code.to_vec(), max_iter }
    }

    #[must_use]
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

    /// Looks up a particular register given the index.
    fn f64_to_reg(&self, v: f64) -> u8 {
        let v = (v % (self.reg.len() as f64) + self.reg.len() as f64) as usize;
        (v % self.reg.len()) as u8
    }

    /// Jumps to the first label with the given label id.
    fn label_jmp(&mut self, label: u8) {
        for (i, op) in self.code.iter().enumerate() {
            if op.code == Opcode::Label && op.label() == label {
                self.pc = i + 1;
                return;
            }
        }
    }

    // Returns true iff finished.
    fn step(&mut self) -> bool {
        if let Some(op) = self.fetch() {
            let rx = op.data[0];
            let ry = op.data[1];
            match op.code {
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
                    self.set_reg(rx, op.imm_value());
                }
                Opcode::IndirectCopy => {
                    self.set_reg(self.f64_to_reg(self.reg(rx)), self.reg(ry));
                }
                Opcode::Jlt => {
                    if self.reg(rx) < self.reg(ry) - EP {
                        self.label_jmp(op.label());
                    }
                }
                Opcode::Jle => {
                    if self.reg(rx) <= self.reg(ry) + EP {
                        self.label_jmp(op.label());
                    }
                }
                Opcode::Jeq => {
                    if (self.reg(rx) - self.reg(ry)).abs() < EP {
                        self.label_jmp(op.label());
                    }
                }
                Opcode::Jmp => {
                    self.label_jmp(op.label());
                }
                Opcode::Label => {}
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
