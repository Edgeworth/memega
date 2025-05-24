use smallvec::{SmallVec, smallvec};

use crate::evaluators::lgp::vm::op::Op;

/// Virtual machine for lgp code. Programs should not be able to run forever,
/// and have acyclic control flow graphs.
#[must_use]
#[derive(Debug, Clone)]
pub struct LgpOptimizer {
    code: Vec<Op>,
    output_regs: SmallVec<[u8; 4]>,
}

impl LgpOptimizer {
    pub fn new(code: &[Op], output_regs: &[u8]) -> Self {
        Self { code: code.to_vec(), output_regs: output_regs.into() }
    }

    #[must_use]
    pub fn optimize(&self) -> Vec<Op> {
        let mut eff_regs = [false; u8::MAX as usize];
        for reg in &self.output_regs {
            eff_regs[*reg as usize] = true;
        }

        let mut eff_code = vec![];
        let mut next_effective = false;
        let mut next_output_regs: SmallVec<[u8; 1]> = smallvec![];
        for op in self.code.iter().rev() {
            // Check to see if this op affects an effective register.
            let mut effective = false;
            for output in op.operands().output_regs() {
                if eff_regs[output as usize] {
                    effective = true;
                    // Changes to this register earlier in the program no longer
                    // affect the final output registers (unless it is part of a
                    // branch).
                    eff_regs[output as usize] = false;
                }
            }

            // If this op is a branch, and the next instruction in the program
            // is effective, then this op is also effective.
            // Also, re-add the next ops output registers to the effective
            // register set, since the branch might not execute, and those
            // output registers that we previously removed could still be
            // effective.
            if next_effective && op.code().is_branch() {
                effective = true;
                for reg in next_output_regs {
                    eff_regs[reg as usize] = true;
                }
            }

            // If this op is reachable, add it to the reachable code and append
            // its inputs to the reachable registers.
            if effective {
                eff_code.push(*op);
                for input in op.operands().input_regs() {
                    eff_regs[input as usize] = true;
                }
            }
            next_effective = effective;
            next_output_regs = op.operands().output_regs();
        }

        eff_code.reverse();
        eff_code
    }
}

#[cfg(test)]
mod tests {
    use eyre::Result;
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::evaluators::lgp::vm::asm::lgp_asm;
    use crate::evaluators::lgp::vm::disasm::lgp_disasm;

    #[test]
    fn optimize_branches() -> Result<()> {
        let code = lgp_asm(
            "neg r1, r2\n\
            iflt r1, r2\n\
            mul r0, r1, r3\n\
            iflt r2, r3\n\
            mul r0, r2, r3\n\
            add r0, r1, r2\n",
        )?;
        let expected = "neg r1, r2\n\
            add r0, r1, r2\n";
        assert_eq!(expected, lgp_disasm(&LgpOptimizer::new(&code, &[0]).optimize()));
        Ok(())
    }

    #[test]
    fn optimize_basic() -> Result<()> {
        let code = lgp_asm(
            "add r1, r1, r2\n\
            iflt r1, r2\n\
            mul r1, r1, r3\n\
            add r0, r2, r3\n",
        )?;
        let expected = "add r0, r2, r3\n";
        assert_eq!(expected, lgp_disasm(&LgpOptimizer::new(&code, &[0]).optimize()));
        Ok(())
    }

    #[test]
    fn optimize_branch_off() -> Result<()> {
        let code = lgp_asm(
            "add r4, r1, r2\n\
            add r3, r1, r2\n\
            add r1, r1, r2\n\
            iflt r2, r3\n\
            mul r1, r2, r3\n\
            add r0, r2, r2\n",
        )?;
        let expected = "add r0, r2, r2\n";
        assert_eq!(expected, lgp_disasm(&LgpOptimizer::new(&code, &[0]).optimize()));
        Ok(())
    }

    #[test]
    fn optimize_branch_on() -> Result<()> {
        let code = lgp_asm(
            "add r4, r1, r2\n\
            add r3, r1, r2\n\
            add r1, r1, r2\n\
            iflt r2, r3\n\
            mul r1, r2, r3\n\
            add r0, r1, r1\n",
        )?;
        let expected = "add r3, r1, r2\n\
            add r1, r1, r2\n\
            iflt r2, r3\n\
            mul r1, r2, r3\n\
            add r0, r1, r1\n";
        assert_eq!(expected, lgp_disasm(&LgpOptimizer::new(&code, &[0]).optimize()));
        Ok(())
    }

    #[test]
    fn optimize_two_branch_off() -> Result<()> {
        let code = lgp_asm(
            "add r4, r1, r2\n\
            add r3, r1, r2\n\
            add r1, r1, r2\n\
            iflt r2, r4\n\
            iflt r2, r3\n\
            mul r1, r2, r3\n\
            add r0, r2, r2\n",
        )?;
        let expected = "add r0, r2, r2\n";
        assert_eq!(expected, lgp_disasm(&LgpOptimizer::new(&code, &[0]).optimize()));
        Ok(())
    }

    #[test]
    fn optimize_two_branch_on() -> Result<()> {
        let code = lgp_asm(
            "add r4, r1, r2\n\
            add r3, r1, r2\n\
            add r1, r1, r2\n\
            iflt r2, r4\n\
            iflt r2, r3\n\
            mul r1, r2, r3\n\
            add r0, r1, r1\n",
        )?;
        let expected = "add r4, r1, r2\n\
            add r3, r1, r2\n\
            add r1, r1, r2\n\
            iflt r2, r4\n\
            iflt r2, r3\n\
            mul r1, r2, r3\n\
            add r0, r1, r1\n";
        assert_eq!(expected, lgp_disasm(&LgpOptimizer::new(&code, &[0]).optimize()));
        Ok(())
    }

    #[test]
    fn optimize_remove_last_branch() -> Result<()> {
        let code = lgp_asm(
            "iflt r1, r2\n\
            mul r1, r1, r3\n",
        )?;
        let expected = "";
        assert_eq!(expected, lgp_disasm(&LgpOptimizer::new(&code, &[0]).optimize()));
        Ok(())
    }

    #[test]
    fn optimize_keep_last_branch() -> Result<()> {
        let code = lgp_asm(
            "iflt r1, r2\n\
            mul r0, r1, r3\n",
        )?;
        let expected = "iflt r1, r2\n\
            mul r0, r1, r3\n";
        assert_eq!(expected, lgp_disasm(&LgpOptimizer::new(&code, &[0]).optimize()));
        Ok(())
    }
}
