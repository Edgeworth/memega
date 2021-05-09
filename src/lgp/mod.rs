use std::convert::TryFrom;

use strum::EnumCount;

use crate::lgp::exec::Opcode;

pub mod disasm;
pub mod exec;
pub mod state;

fn u8_to_opcode(v: u8) -> Opcode {
    let v = v % (Opcode::COUNT as u8);
    Opcode::try_from(v).unwrap()
}
