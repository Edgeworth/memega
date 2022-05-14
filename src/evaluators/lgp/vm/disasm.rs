use std::fmt::Write;

use crate::evaluators::lgp::vm::op::Op;
#[must_use]
pub fn lgp_disasm(code: &[Op]) -> String {
    let mut prog = String::new();
    for ins in code.iter() {
        let _ = writeln!(prog, "{}", ins);
    }
    prog
}

#[cfg(test)]
mod tests {
    use eyre::Result;

    use super::*;
    use crate::evaluators::lgp::vm::asm::lgp_asm;
    use crate::evaluators::lgp::vm::op::Op;
    use crate::evaluators::lgp::vm::opcode::{Opcode, Operands};

    #[test]
    fn basic_disasm() -> Result<()> {
        let code = vec![
            Op::new(Opcode::Add, Operands::Reg3Assign { ri: 0, ra: 1, rb: 2 }),
            Op::new(Opcode::Sub, Operands::Reg3Assign { ri: 2, ra: 1, rb: 0 }),
        ];
        let text = "add r0, r1, r2\nsub r2, r1, r0\n";
        assert_eq!(text, lgp_disasm(&code));
        assert_eq!(code, lgp_asm(text)?);
        Ok(())
    }
}
