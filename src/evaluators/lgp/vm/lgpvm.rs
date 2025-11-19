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
        if self.pc >= self.code.len() { None } else { Some(self.code[self.pc]) }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluators::lgp::vm::op::Op;
    use crate::evaluators::lgp::vm::opcode::{Opcode, Operands};
    use pretty_assertions::assert_eq;

    #[test]
    fn test_vm_creation() {
        let cfg = LgpVmCfg::new()
            .set_regs(&[1.0, 2.0, 3.0])
            .set_constants(&[10.0, 20.0])
            .set_code(&[]);

        let vm = LgpVm::new(&cfg);
        assert_eq!(vm.mem(0), 1.0);
        assert_eq!(vm.mem(1), 2.0);
        assert_eq!(vm.mem(2), 3.0);
        assert_eq!(vm.mem(3), 10.0); // Constants after registers
        assert_eq!(vm.mem(4), 20.0);
    }

    #[test]
    fn test_add_operation() {
        let add_op = Op::new(Opcode::Add, Operands::Reg3Assign { ri: 0, ra: 1, rb: 2 });
        let cfg = LgpVmCfg::new()
            .set_regs(&[0.0, 5.0, 3.0])
            .set_constants(&[])
            .set_code(&[add_op]);

        let mut vm = LgpVm::new(&cfg);
        vm.run();
        assert_eq!(vm.mem(0), 8.0); // 5 + 3 = 8
    }

    #[test]
    fn test_sub_operation() {
        let sub_op = Op::new(Opcode::Sub, Operands::Reg3Assign { ri: 0, ra: 1, rb: 2 });
        let cfg = LgpVmCfg::new()
            .set_regs(&[0.0, 10.0, 3.0])
            .set_constants(&[])
            .set_code(&[sub_op]);

        let mut vm = LgpVm::new(&cfg);
        vm.run();
        assert_eq!(vm.mem(0), 7.0); // 10 - 3 = 7
    }

    #[test]
    fn test_mul_operation() {
        let mul_op = Op::new(Opcode::Mul, Operands::Reg3Assign { ri: 0, ra: 1, rb: 2 });
        let cfg = LgpVmCfg::new()
            .set_regs(&[0.0, 4.0, 5.0])
            .set_constants(&[])
            .set_code(&[mul_op]);

        let mut vm = LgpVm::new(&cfg);
        vm.run();
        assert_eq!(vm.mem(0), 20.0); // 4 * 5 = 20
    }

    #[test]
    fn test_div_operation() {
        let div_op = Op::new(Opcode::Div, Operands::Reg3Assign { ri: 0, ra: 1, rb: 2 });
        let cfg = LgpVmCfg::new()
            .set_regs(&[0.0, 20.0, 4.0])
            .set_constants(&[])
            .set_code(&[div_op]);

        let mut vm = LgpVm::new(&cfg);
        vm.run();
        assert_eq!(vm.mem(0), 5.0); // 20 / 4 = 5
    }

    #[test]
    fn test_div_by_zero() {
        let div_op = Op::new(Opcode::Div, Operands::Reg3Assign { ri: 0, ra: 1, rb: 2 });
        let cfg = LgpVmCfg::new()
            .set_regs(&[0.0, 20.0, 0.0])
            .set_constants(&[])
            .set_code(&[div_op]);

        let mut vm = LgpVm::new(&cfg);
        vm.run();
        // Division by zero produces infinity, which is not finite, so r0 should remain 0.0
        assert_eq!(vm.mem(0), 0.0);
    }

    #[test]
    fn test_pow_operation() {
        let pow_op = Op::new(Opcode::Pow, Operands::Reg3Assign { ri: 0, ra: 1, rb: 2 });
        let cfg = LgpVmCfg::new()
            .set_regs(&[0.0, 2.0, 3.0])
            .set_constants(&[])
            .set_code(&[pow_op]);

        let mut vm = LgpVm::new(&cfg);
        vm.run();
        assert_eq!(vm.mem(0), 8.0); // 2^3 = 8
    }

    #[test]
    fn test_abs_operation() {
        let abs_op = Op::new(Opcode::Abs, Operands::Reg2Assign { ri: 0, ra: 1 });
        let cfg = LgpVmCfg::new()
            .set_regs(&[0.0, -5.0])
            .set_constants(&[])
            .set_code(&[abs_op]);

        let mut vm = LgpVm::new(&cfg);
        vm.run();
        assert_eq!(vm.mem(0), 5.0); // |-5| = 5
    }

    #[test]
    fn test_neg_operation() {
        let neg_op = Op::new(Opcode::Neg, Operands::Reg2Assign { ri: 0, ra: 1 });
        let cfg = LgpVmCfg::new()
            .set_regs(&[0.0, 5.0])
            .set_constants(&[])
            .set_code(&[neg_op]);

        let mut vm = LgpVm::new(&cfg);
        vm.run();
        assert_eq!(vm.mem(0), -5.0); // -(5) = -5
    }

    #[test]
    fn test_copy_operation() {
        let copy_op = Op::new(Opcode::Copy, Operands::Reg2Assign { ri: 0, ra: 1 });
        let cfg = LgpVmCfg::new()
            .set_regs(&[0.0, 42.0])
            .set_constants(&[])
            .set_code(&[copy_op]);

        let mut vm = LgpVm::new(&cfg);
        vm.run();
        assert_eq!(vm.mem(0), 42.0);
    }

    #[test]
    fn test_load_operation() {
        let load_op = Op::new(Opcode::Load, Operands::ImmAssign { ri: 0, imm: 3.14 });
        let cfg = LgpVmCfg::new()
            .set_regs(&[0.0])
            .set_constants(&[])
            .set_code(&[load_op]);

        let mut vm = LgpVm::new(&cfg);
        vm.run();
        assert!((vm.mem(0) - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_iflt_branch_taken() {
        // r0 = r1 + r2, but only if r1 < r2
        let iflt_op = Op::new(Opcode::IfLt, Operands::Reg2Cmp { ra: 1, rb: 2 });
        let add_op = Op::new(Opcode::Add, Operands::Reg3Assign { ri: 0, ra: 1, rb: 2 });

        let cfg = LgpVmCfg::new()
            .set_regs(&[0.0, 3.0, 5.0]) // 3 < 5, so branch is taken
            .set_constants(&[])
            .set_code(&[iflt_op, add_op]);

        let mut vm = LgpVm::new(&cfg);
        vm.run();
        assert_eq!(vm.mem(0), 8.0); // Add was executed
    }

    #[test]
    fn test_iflt_branch_not_taken() {
        // r0 = r1 + r2, but only if r1 < r2
        let iflt_op = Op::new(Opcode::IfLt, Operands::Reg2Cmp { ra: 1, rb: 2 });
        let add_op = Op::new(Opcode::Add, Operands::Reg3Assign { ri: 0, ra: 1, rb: 2 });

        let cfg = LgpVmCfg::new()
            .set_regs(&[0.0, 5.0, 3.0]) // 5 >= 3, so branch is not taken
            .set_constants(&[])
            .set_code(&[iflt_op, add_op]);

        let mut vm = LgpVm::new(&cfg);
        vm.run();
        assert_eq!(vm.mem(0), 0.0); // Add was skipped
    }

    #[test]
    fn test_constants_immutable() {
        // Try to write to a constant (should be ignored)
        let add_op = Op::new(Opcode::Add, Operands::Reg3Assign { ri: 2, ra: 0, rb: 1 });
        let cfg = LgpVmCfg::new()
            .set_regs(&[10.0, 20.0])
            .set_constants(&[100.0])
            .set_code(&[add_op]);

        let mut vm = LgpVm::new(&cfg);
        vm.run();
        // r2 is a constant, so it should not be modified
        assert_eq!(vm.mem(2), 100.0);
    }

    #[test]
    fn test_sequential_operations() {
        // r0 = r1 + r2, then r0 = r0 * 2
        let add_op = Op::new(Opcode::Add, Operands::Reg3Assign { ri: 0, ra: 1, rb: 2 });
        let mul_op = Op::new(Opcode::Mul, Operands::Reg3Assign { ri: 0, ra: 0, rb: 1 });

        let cfg = LgpVmCfg::new()
            .set_regs(&[0.0, 2.0, 3.0])
            .set_constants(&[])
            .set_code(&[add_op, mul_op]);

        let mut vm = LgpVm::new(&cfg);
        vm.run();
        // (2 + 3) * 2 = 10
        assert_eq!(vm.mem(0), 10.0);
    }

    #[test]
    fn test_mem_slice() {
        let cfg = LgpVmCfg::new()
            .set_regs(&[1.0, 2.0, 3.0])
            .set_constants(&[10.0, 20.0])
            .set_code(&[]);

        let vm = LgpVm::new(&cfg);
        let mem = vm.mem_slice();
        assert_eq!(mem.len(), 5);
        assert_eq!(mem, &[1.0, 2.0, 3.0, 10.0, 20.0]);
    }

}
