use eyre::{Result, eyre};

use crate::evaluators::lgp::vm::op::Op;
use crate::evaluators::lgp::vm::opcode::{Opcode, Operands};

fn lgp_asm_op(s: &str) -> Result<Op> {
    let mut tokens = s.split_whitespace();
    let op = match tokens.next().ok_or_else(|| eyre!("missing token"))? {
        "add" => Opcode::Add,
        "sub" => Opcode::Sub,
        "mul" => Opcode::Mul,
        "div" => Opcode::Div,
        "abs" => Opcode::Abs,
        "neg" => Opcode::Neg,
        "pow" => Opcode::Pow,
        "ln" => Opcode::Ln,
        "sin" => Opcode::Sin,
        "cos" => Opcode::Cos,
        "load" => Opcode::Load,
        "copy" => Opcode::Copy,
        "iflt" => Opcode::IfLt,
        _ => return Err(eyre!("unknown instruction format")),
    };
    let mut op = Op::from_code(op);
    match op.operands_mut() {
        Operands::Reg2Cmp { ra, rb } => {
            let tok = tokens.next().ok_or_else(|| eyre!("missing register"))?;
            *ra = tok.replace(',', "")[1..].parse()?;

            let tok = tokens.next().ok_or_else(|| eyre!("missing register"))?;
            *rb = tok.replace(',', "")[1..].parse()?;
        }
        Operands::Reg2Assign { ri, ra } => {
            let tok = tokens.next().ok_or_else(|| eyre!("missing register"))?;
            *ri = tok.replace(',', "")[1..].parse()?;

            let tok = tokens.next().ok_or_else(|| eyre!("missing register"))?;
            *ra = tok.replace(',', "")[1..].parse()?;
        }
        Operands::Reg3Assign { ri, ra, rb } => {
            let tok = tokens.next().ok_or_else(|| eyre!("missing register"))?;
            *ri = tok.replace(',', "")[1..].parse()?;

            let tok = tokens.next().ok_or_else(|| eyre!("missing register"))?;
            *ra = tok.replace(',', "")[1..].parse()?;

            let tok = tokens.next().ok_or_else(|| eyre!("missing register"))?;
            *rb = tok.replace(',', "")[1..].parse()?;
        }
        Operands::ImmAssign { ri, imm } => {
            let tok = tokens.next().ok_or_else(|| eyre!("missing register"))?;
            *ri = tok.replace(',', "")[1..].parse()?;

            let tok = tokens.next().ok_or_else(|| eyre!("missing register"))?;
            *imm = tok.parse::<f32>()?;
        }
    }
    Ok(op)
}

pub fn lgp_asm(s: &str) -> Result<Vec<Op>> {
    let mut ops = Vec::new();
    for line in s.lines() {
        ops.push(lgp_asm_op(line)?);
    }
    Ok(ops)
}
