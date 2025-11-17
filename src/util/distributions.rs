use rand::Rng;
use rand::prelude::Distribution;

#[must_use]
#[derive(Debug)]
pub struct PrintableAscii;

impl Distribution<u8> for PrintableAscii {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> u8 {
        r.random_range(32..=126)
    }
}

impl Distribution<char> for PrintableAscii {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> char {
        Distribution::<u8>::sample(self, r) as char
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_printable_ascii_u8() {
        let dist = PrintableAscii;
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..100 {
            let byte: u8 = dist.sample(&mut rng);
            assert!(byte >= 32 && byte <= 126, "Byte {} is not printable ASCII", byte);
        }
    }

    #[test]
    fn test_printable_ascii_char() {
        let dist = PrintableAscii;
        let mut rng = StdRng::seed_from_u64(123);

        for _ in 0..100 {
            let ch: char = dist.sample(&mut rng);
            assert!(ch.is_ascii(), "Char {} is not ASCII", ch);
            assert!(!ch.is_ascii_control(), "Char {} is a control character", ch);
            assert!(ch >= ' ' && ch <= '~', "Char {} is not printable ASCII", ch);
        }
    }

    #[test]
    fn test_printable_ascii_range() {
        let dist = PrintableAscii;
        let mut rng = StdRng::seed_from_u64(456);

        // Sample many times and ensure we get variety
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            let byte: u8 = dist.sample(&mut rng);
            seen.insert(byte);
        }

        // Should see at least 50 different characters
        assert!(seen.len() >= 50, "Not enough variety in samples");
    }
}
