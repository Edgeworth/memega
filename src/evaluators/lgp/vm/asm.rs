use eyre::{eyre, Result};

use crate::evaluators::lgp::vm::op::Op;
use crate::evaluators::lgp::vm::opcode::{Opcode, Operand};

fn lgp_asm_op(s: &str) -> Result<Op> {
    let mut tokens = s.split_whitespace();
    let mut data = [0, 0, 0];
    let op = match tokens.next().ok_or_else(|| eyre!("missing token"))? {
        "add" => Opcode::Add,
        "sub" => Opcode::Sub,
        "mul" => Opcode::Mul,
        "div" => Opcode::Div,
        "abs" => Opcode::Abs,
        "neg" => Opcode::Neg,
        "pow" => Opcode::Pow,
        "ln" => Opcode::Log,
        "load" => Opcode::Load,
        "mov" => Opcode::IndirectCopy,
        "jlt" => Opcode::Jlt,
        "jle" => Opcode::Jle,
        "jeq" => Opcode::Jeq,
        "lbl" => Opcode::Label,
        _ => return Err(eyre!("unknown instruction format")),
    };
    let mut idx = 0;
    for i in 0..data.len() {
        match op.operand(i) {
            Operand::None => {}
            Operand::Register => {
                let tok = tokens.next().ok_or_else(|| eyre!("missing register"))?;
                data[idx] = tok.replace(&[',', '[', ']'][..], "")[1..].parse()?;
                idx += 1;
            }
            Operand::Immediate => {
                let tok = tokens.next().ok_or_else(|| eyre!("missing immediate for {:?}", op))?;
                let mut op = Op::new(op, data);
                op.set_imm_f64(tok.parse::<f64>()?);
                return Ok(op);
            }
            Operand::Label => {
                let tok = tokens.next().ok_or_else(|| eyre!("missing label"))?;
                data[idx] = tok.parse::<u8>()?;
                idx += 1;
            }
        }
    }
    Ok(Op::new(op, data))
}

pub fn lgp_asm(s: &str) -> Result<Vec<Op>> {
    let mut ops = Vec::new();
    for line in s.lines() {
        ops.push(lgp_asm_op(line)?);
    }
    Ok(ops)
}
