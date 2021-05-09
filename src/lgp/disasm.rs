use crate::lgp::exec::Opcode;

// Disassembler for lgp code.
#[derive(Debug, Clone)]
pub struct LgpDisasm {
    pc: usize,
    code: Vec<u8>,
    prog: String,
}

impl LgpDisasm {
    pub fn new(code: &[u8]) -> Self {
        Self { pc: 0, code: code.to_vec(), prog: String::new() }
    }

    fn fetch(&mut self) -> u8 {
        // Check if we finished the program. Return 0 if we overrun.
        let v = if self.pc > self.code.len() { 0 } else { self.code[self.pc] };
        self.pc += 1;
        v
    }

    // Returns true iff finished.
    fn step(&mut self) -> bool {
        let op: Opcode = self.fetch().into();
        match op {
            Opcode::Nop => {
                self.prog += "nop";
            }
            Opcode::Add => {
                let rx = self.fetch();
                let ry = self.fetch();
                self.prog += &format!("r{} += r{}", rx, ry);
            }
            Opcode::Sub => {
                let rx = self.fetch();
                let ry = self.fetch();
                self.prog += &format!("r{} -= r{}", rx, ry);
            }
            Opcode::Mul => {
                let rx = self.fetch();
                let ry = self.fetch();
                self.prog += &format!("r{} *= r{}", rx, ry);
            }
            Opcode::Div => {
                let rx = self.fetch();
                let ry = self.fetch();
                self.prog += &format!("r{} /= r{}", rx, ry);
            }
            Opcode::Abs => {
                let rx = self.fetch();
                self.prog += &format!("r{} = |r{}|", rx, rx);
            }
            Opcode::Neg => {
                let rx = self.fetch();
                self.prog += &format!("r{} = -r{}", rx, rx);
            }
            Opcode::Pow => {
                let rx = self.fetch();
                let ry = self.fetch();
                self.prog += &format!("r{} = r{} ** r{}", rx, rx, ry);
            }
            Opcode::Log => {
                let rx = self.fetch();
                self.prog += &format!("r{} = ln(r{})", rx, rx);
            }
            Opcode::Load => {
                let rx = self.fetch();
                let lo = self.fetch();
                let hi = self.fetch();
                self.prog += &format!("[r{}] = {}", rx, (hi as f64) + (lo as f64) / 256.0);
            }
            Opcode::Copy => {
                let rx = self.fetch();
                let ry = self.fetch();
                self.prog += &format!("r{} = r{}", rx, ry);
            }
            Opcode::Jmp => {
                let imm = self.fetch() as i8;
                self.prog += &format!("jmp {}", imm);
            }
            Opcode::Jlt => {
                let rx = self.fetch();
                let ry = self.fetch();
                let imm = self.fetch() as i8;
                self.prog += &format!("if r{} < r{}: jmp {}", rx, ry, imm);
            }
            Opcode::Jle => {
                let rx = self.fetch();
                let ry = self.fetch();
                let imm = self.fetch() as i8;
                self.prog += &format!("if r{} <= r{}: jmp {}", rx, ry, imm);
            }
            Opcode::Jeq => {
                let rx = self.fetch();
                let ry = self.fetch();
                let imm = self.fetch() as i8;
                self.prog += &format!("if r{} == r{}: jmp {}", rx, ry, imm);
            }
        }
        self.prog += "\n";
        self.pc >= self.code.len()
    }


    pub fn disasm(&mut self) -> String {
        while !self.step() {}
        self.prog.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_disasm() {
        let mut ds = LgpDisasm::new(&[0, 1, 2, 3]);
        assert_eq!("nop\nr2 += r3\n", ds.disasm())
    }
}
