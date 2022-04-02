use crate::evaluators::lgp::vm::op::Op;

#[must_use]
pub fn lgp_disasm(code: &[Op]) -> String {
    let mut prog = String::new();
    for ins in code.iter() {
        prog += &format!("{}\n", ins);
    }
    prog
}

#[cfg(test)]
mod tests {
    use eyre::Result;

    use super::*;
    use crate::evaluators::lgp::vm::asm::lgp_asm;
    use crate::evaluators::lgp::vm::op::Op;
    use crate::evaluators::lgp::vm::opcode::Opcode;

    #[test]
    fn basic_disasm() -> Result<()> {
        let code = vec![Op::new(Opcode::Label, [1, 0, 0]), Op::new(Opcode::Add, [2, 3, 0])];
        let text = "lbl 1\nadd r2, r3\n";
        assert_eq!(text, lgp_disasm(&code));
        assert_eq!(code, lgp_asm(text)?);
        Ok(())
    }
}
