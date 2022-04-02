use enumset::EnumSet;
use rand::prelude::IteratorRandom;
use rand::Rng;
use strum::IntoEnumIterator;

use crate::evaluators::lgp::vm::op::{Op, Opcode, Operand};
use crate::ops::mutation::mutate_creep;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd)]
pub struct LgpCfg {
    num_reg: usize,
    max_code: usize,
    imm_sf: usize, // Number of significant figures in immediate values.
    opcodes: EnumSet<Opcode>,
}

impl LgpCfg {
    #[must_use]
    pub fn new() -> Self {
        Self { num_reg: 4, max_code: 4, imm_sf: 2, opcodes: Opcode::iter().collect() }
    }

    fn clamped_code_size(&self, code_size: Option<usize>) -> i8 {
        let code_size = match code_size {
            Some(n) => n,
            None => self.max_code,
        };
        code_size.clamp(0, i8::MAX as usize) as i8
    }

    #[must_use]
    pub fn rand_op(&self, code_size: Option<usize>) -> Op {
        let mut r = rand::thread_rng();
        let op = self.opcodes.iter().choose(&mut r).unwrap();
        let code_size = self.clamped_code_size(code_size);
        let mut data = [0, 0, 0];
        for (i, v) in data.iter_mut().enumerate() {
            match op.operand(i) {
                Operand::None => {}
                Operand::Register => *v = r.gen_range(0..self.num_reg) as u8,
                Operand::Immediate => *v = r.gen::<u8>(),
                Operand::Relative => *v = r.gen_range(-code_size..=code_size) as u8,
            }
        }
        Op::new(op, data)
    }

    // Micro-mutation of the instruction without changing the opcode.
    pub fn mutate(&self, code_size: Option<usize>, op: &mut Op) {
        let mut r = rand::thread_rng();
        let num_operands = op.num_operands();
        if num_operands == 0 {
            return;
        }
        let idx = r.gen_range(0..num_operands);
        let code_size = self.clamped_code_size(code_size);
        match op.op.operand(idx) {
            Operand::None => {}
            Operand::Register => op.data[idx] = r.gen_range(0..self.num_reg) as u8,
            Operand::Immediate => op.data[idx] = mutate_creep(op.data[idx], 64),
            Operand::Relative => {
                op.data[idx] =
                    mutate_creep(op.data[idx] as i8, code_size).clamp(-code_size, code_size) as u8;
            }
        }
    }

    #[must_use]
    pub fn set_num_reg(mut self, num_reg: usize) -> Self {
        self.num_reg = num_reg;
        self
    }

    #[must_use]
    pub fn set_max_code(mut self, max_code: usize) -> Self {
        self.max_code = max_code;
        self
    }

    #[must_use]
    pub fn set_imm_sf(mut self, imm_sf: usize) -> Self {
        self.imm_sf = imm_sf;
        self
    }

    #[must_use]
    pub fn set_opcodes(mut self, opcodes: EnumSet<Opcode>) -> Self {
        self.opcodes = opcodes;
        self
    }

    #[must_use]
    pub fn num_reg(&self) -> usize {
        self.num_reg
    }

    #[must_use]
    pub fn max_code(&self) -> usize {
        self.max_code
    }

    #[must_use]
    pub fn imm_sf(&self) -> usize {
        self.imm_sf
    }

    #[must_use]
    pub fn opcodes(&self) -> EnumSet<Opcode> {
        self.opcodes
    }
}

impl Default for LgpCfg {
    fn default() -> Self {
        Self::new()
    }
}
