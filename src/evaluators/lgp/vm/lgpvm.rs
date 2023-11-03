use crate::evaluators::lgp::vm::cfg::LgpVmCfg;
use crate::evaluators::lgp::vm::op::Op;
use crate::evaluators::lgp::vm::opcode::{Opcode, Operands};

/// Virtual machine for lgp code. Programs should not be able to run forever,
/// and have acyclic control flow graphs.
#[must_use]
#[derive(Debug, Clone)]
pub struct LgpVm {
    pc: usize,
    mem: Vec<f64>,
    code: Vec<Op>,
    /// Number of non-constant memory locations.
    num_reg: usize,
}

impl LgpVm {
    pub fn new(cfg: &LgpVmCfg) -> Self {
        let num_reg = cfg.regs().len();
        let mem_size = cfg.regs().len() + cfg.constants().len();
        let mut mem = vec![0.0; mem_size];
        mem[..num_reg].copy_from_slice(cfg.regs());
        mem[num_reg..].copy_from_slice(cfg.constants());
        Self { pc: 0, mem, code: cfg.code().to_vec(), num_reg }
    }

    fn is_constant(&self, idx: u8) -> bool {
        idx as usize >= self.num_reg
    }

    #[must_use]
    pub fn mem_slice(&self) -> &[f64] {
        &self.mem
    }

    #[must_use]
    pub fn mem(&self, idx: u8) -> f64 {
        self.mem[idx as usize]
    }

    fn set_mem(&mut self, idx: u8, v: f64) {
        self.mem[idx as usize] = v;
    }

    fn peek(&mut self) -> Option<Op> {
        if self.pc >= self.code.len() {
            None
        } else {
            Some(self.code[self.pc])
        }
    }

    fn fetch(&mut self) -> Option<Op> {
        if let Some(v) = self.peek() {
            self.pc += 1;
            Some(v)
        } else {
            None
        }
    }

    // Returns true iff finished.
    fn step(&mut self) -> bool {
        if let Some(op) = self.fetch() {
            match (op.code(), op.operands()) {
                (Opcode::Add, Operands::Reg3Assign { ri, ra, rb }) => {
                    let v = self.mem(ra) + self.mem(rb);
                    if v.is_finite() && !self.is_constant(ri) {
                        self.set_mem(ri, v);
                    }
                }
                (Opcode::Sub, Operands::Reg3Assign { ri, ra, rb }) => {
                    let v = self.mem(ra) - self.mem(rb);
                    if v.is_finite() && !self.is_constant(ri) {
                        self.set_mem(ri, v);
                    }
                }
                (Opcode::Mul, Operands::Reg3Assign { ri, ra, rb }) => {
                    let v = self.mem(ra) * self.mem(rb);
                    if v.is_finite() && !self.is_constant(ri) {
                        self.set_mem(ri, v);
                    }
                }
                (Opcode::Div, Operands::Reg3Assign { ri, ra, rb }) => {
                    let v = self.mem(ra) / self.mem(rb);
                    if v.is_finite() && !self.is_constant(ri) {
                        self.set_mem(ri, v);
                    }
                }
                (Opcode::Pow, Operands::Reg3Assign { ri, ra, rb }) => {
                    let v = self.mem(ra).powf(self.mem(rb));
                    if v.is_finite() && !self.is_constant(ri) {
                        self.set_mem(ri, v);
                    }
                }
                (Opcode::Abs, Operands::Reg2Assign { ri, ra }) => {
                    if !self.is_constant(ri) {
                        self.set_mem(ri, self.mem(ra).abs());
                    }
                }
                (Opcode::Neg, Operands::Reg2Assign { ri, ra }) => {
                    if !self.is_constant(ri) {
                        self.set_mem(ri, -self.mem(ra));
                    }
                }
                (Opcode::Ln, Operands::Reg2Assign { ri, ra }) => {
                    let v = self.mem(ra).ln();
                    if v.is_finite() && !self.is_constant(ri) {
                        self.set_mem(ri, v);
                    }
                }
                (Opcode::Sin, Operands::Reg2Assign { ri, ra }) => {
                    let v = self.mem(ra).sin();
                    if v.is_finite() && !self.is_constant(ri) {
                        self.set_mem(ri, v);
                    }
                }
                (Opcode::Cos, Operands::Reg2Assign { ri, ra }) => {
                    let v = self.mem(ra).cos();
                    if v.is_finite() && !self.is_constant(ri) {
                        self.set_mem(ri, v);
                    }
                }
                (Opcode::Load, Operands::ImmAssign { ri, imm }) => {
                    if !self.is_constant(ri) {
                        self.set_mem(ri, imm as f64);
                    }
                }
                (Opcode::Copy, Operands::Reg2Assign { ri, ra }) => {
                    if !self.is_constant(ri) {
                        self.set_mem(ri, self.mem(ra));
                    }
                }
                (Opcode::IfLt, Operands::Reg2Cmp { ra, rb }) => {
                    if self.mem(ra) >= self.mem(rb) {
                        // Find first non if instruction and skip it (last fetch will skip).
                        while let Some(op) = self.fetch()
                            && op.code().is_branch()
                        {}
                    }
                }
                _ => panic!("incorrect or unimplemented opcode: {op:?}"),
            }
            false
        } else {
            true
        }
    }

    pub fn run(&mut self) {
        while !self.step() {}
    }
}
