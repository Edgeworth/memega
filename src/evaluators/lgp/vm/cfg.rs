use crate::evaluators::lgp::vm::op::Op;

/// Virtual machine for lgp code.
#[must_use]
#[derive(Debug, Clone)]
pub struct LgpVmCfg {
    /// Initial registers (readwrite memory).
    regs: Vec<f64>,
    /// Values of constants to be copied to the end of memory as read only values.
    constants: Vec<f64>,
    /// Code to execute.
    code: Vec<Op>,
}

impl Default for LgpVmCfg {
    fn default() -> Self {
        Self::new()
    }
}

impl LgpVmCfg {
    pub fn new() -> Self {
        Self { regs: vec![], constants: vec![], code: vec![] }
    }

    pub fn set_regs(mut self, regs: &[f64]) -> Self {
        self.regs = regs.to_vec();
        assert!(
            self.regs.len() + self.constants.len() <= 256,
            "cannot use more than 256 memory locations"
        );
        self
    }

    pub fn set_constants(mut self, constants: &[f64]) -> Self {
        self.constants = constants.to_vec();
        assert!(
            self.regs.len() + self.constants.len() <= 256,
            "cannot use more than 256 memory locations"
        );
        self
    }

    pub fn set_code(mut self, code: &[Op]) -> Self {
        self.code = code.to_vec();
        self
    }

    #[must_use]
    pub fn regs(&self) -> &[f64] {
        &self.regs
    }

    #[must_use]
    pub fn constants(&self) -> &[f64] {
        &self.constants
    }

    pub fn code(&self) -> &[Op] {
        &self.code
    }
}
