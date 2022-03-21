use rand::prelude::Distribution;
use rand::Rng;

#[derive(Debug)]
pub struct PrintableAscii;

impl Distribution<u8> for PrintableAscii {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> u8 {
        r.gen_range(32..=126)
    }
}

impl Distribution<char> for PrintableAscii {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> char {
        Distribution::<u8>::sample(self, r) as char
    }
}
