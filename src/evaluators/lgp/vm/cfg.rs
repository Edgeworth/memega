use crate::evaluators::lgp::vm::op::Op;

// Virtual machine for lgp code.
#[derive(Debug, Clone)]
pub struct LgpVmCfg {
    /// The size of the memory not including the constants.
    num_reg: usize,
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
    #[must_use]
    pub fn new() -> Self {
        Self { num_reg: 0, constants: vec![], code: vec![] }
    }

    #[must_use]
    pub fn set_num_reg(mut self, num_reg: usize) -> Self {
        self.num_reg = num_reg;
        assert!(
            self.num_reg + self.constants.len() <= 256,
            "cannot use more than 256 memory locations"
        );
        self
    }

    #[must_use]
    pub fn set_constants(mut self, constants: &[f64]) -> Self {
        self.constants = constants.to_vec();
        assert!(
            self.num_reg + self.constants.len() <= 256,
            "cannot use more than 256 memory locations"
        );
        self
    }

    #[must_use]
    pub fn set_code(mut self, code: &[Op]) -> Self {
        self.code = code.to_vec();
        self
    }

    #[must_use]
    pub fn num_reg(&self) -> usize {
        self.num_reg
    }

    #[must_use]
    pub fn constants(&self) -> &[f64] {
        &self.constants
    }

    #[must_use]
    pub fn code(&self) -> &[Op] {
        &self.code
    }
}
