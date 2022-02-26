use crate::lgp::op::Op;

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
    use super::*;
    use crate::lgp::op::Opcode;

    #[test]
    fn basic_disasm() {
        let code = &[Op::new(Opcode::Nop, [0, 0, 0]), Op::new(Opcode::Add, [2, 3, 0])];
        assert_eq!("nop\nr2 += r3\n", lgp_disasm(code));
    }
}
