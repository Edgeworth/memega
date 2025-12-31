use crate::error::{Error, Result};
use crate::evaluators::lgp::vm::op::Op;
use crate::evaluators::lgp::vm::opcode::{Opcode, Operands};

fn parse_reg(tok: &str) -> Result<u8> {
    let tok = tok.trim_end_matches(',');
    let num = tok.strip_prefix('r').ok_or_else(|| {
        Error::InvalidLgp(format!("register must be formatted like r0, got {tok:?}"))
    })?;
    Ok(num.parse::<u8>()?)
}

fn parse_imm(tok: &str) -> Result<f32> {
    let tok = tok.trim_end_matches(',');
    Ok(tok.parse::<f32>()?)
}

fn lgp_asm_op(s: &str) -> Result<Op> {
    let mut tokens = s.split_whitespace();
    let op = match tokens.next().ok_or_else(|| Error::InvalidLgp("missing token".to_string()))? {
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
        _ => return Err(Error::InvalidLgp("unknown instruction".to_string())),
    };
    let mut op = Op::from_code(op);
    match op.operands_mut() {
        Operands::Reg2Cmp { ra, rb } => {
            let tok = tokens.next().ok_or_else(|| {
                Error::InvalidLgp("missing register for 2-reg compare".to_string())
            })?;
            *ra = parse_reg(tok)?;

            let tok = tokens.next().ok_or_else(|| {
                Error::InvalidLgp("missing register for 2-reg compare".to_string())
            })?;
            *rb = parse_reg(tok)?;
        }
        Operands::Reg2Assign { ri, ra } => {
            let tok = tokens.next().ok_or_else(|| {
                Error::InvalidLgp("missing destination register for 2-reg assign".to_string())
            })?;
            *ri = parse_reg(tok)?;

            let tok = tokens.next().ok_or_else(|| {
                Error::InvalidLgp("missing input register for 2-reg assign".to_string())
            })?;
            *ra = parse_reg(tok)?;
        }
        Operands::Reg3Assign { ri, ra, rb } => {
            let tok = tokens.next().ok_or_else(|| {
                Error::InvalidLgp("missing destination register for 3-reg assign".to_string())
            })?;
            *ri = parse_reg(tok)?;

            let tok = tokens.next().ok_or_else(|| {
                Error::InvalidLgp("missing first input register for 3-reg assign".to_string())
            })?;
            *ra = parse_reg(tok)?;

            let tok = tokens.next().ok_or_else(|| {
                Error::InvalidLgp("missing second input register for 3-reg assign".to_string())
            })?;
            *rb = parse_reg(tok)?;
        }
        Operands::ImmAssign { ri, imm } => {
            let tok = tokens.next().ok_or_else(|| {
                Error::InvalidLgp("missing destination register for imm assign".to_string())
            })?;
            *ri = parse_reg(tok)?;

            let tok = tokens
                .next()
                .ok_or_else(|| Error::InvalidLgp("missing immediate for imm assign".to_string()))?;
            *imm = parse_imm(tok)?;
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

#[cfg(test)]
mod tests {
    use super::lgp_asm;

    #[test]
    fn lgp_asm_invalid_register_is_error() {
        let r = lgp_asm("add , r1, r2\n");
        assert!(r.is_err());
    }
}
