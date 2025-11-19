use std::f64::consts::E;

use num_traits::{Num, Saturating};
use rand::Rng;
use rand::prelude::{IteratorRandom, SliceRandom};
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, StandardNormal, StandardUniform};

// Permutation mutation operators ////////////////////////////////////////////////

// Mutate by swapping
pub fn mutate_swap<T: Copy>(s: &mut [T]) {
    let mut r = rand::rng();
    s.swap(r.random_range(0..s.len()), r.random_range(0..s.len()));
}

// Mutate by making two random elements next to each-other, shuffling the
// elements in between. E.g. AbcdEfg => bcdAEfg
pub fn mutate_insert<T: Copy>(s: &mut [T]) {
    let mut r = rand::rng();
    let st = r.random_range(0..s.len());
    let en = r.random_range(st..s.len());
    for i in st..en {
        s.swap(i, i + 1);
    }
}

// Mutate by scrambling a random substring of the input. e.g. aBCDefg => aCDBefg
pub fn mutate_scramble<T: Copy>(s: &mut [T]) {
    let mut r = rand::rng();
    let st = r.random_range(0..s.len());
    let en = r.random_range(st..s.len());
    s[st..=en].shuffle(&mut r);
}

// Mutate by inverting a random substring of the input, e.g. aBCDefg => aDCBefg.
// For adjacency-based problems this is the smallest mutation - it only affects
// two edges (the ends where the inversion happens).
pub fn mutate_inversion<T: Copy>(s: &mut [T]) {
    let mut r = rand::rng();
    let st = r.random_range(0..s.len());
    let en = r.random_range(st..s.len());
    s[st..=en].reverse();
}

// Discrete mutation operators ////////////////////////////////////////////////

// Generates a random value.
#[must_use]
pub fn mutate_gen<T>() -> T
where
    StandardUniform: Distribution<T>,
{
    let mut r = rand::rng();
    r.random::<T>()
}

// Replaces a random value in |s| with |v|.
pub fn mutate_reset<T>(s: &mut [T], v: T) {
    let mut r = rand::rng();
    if let Some(ov) = s.iter_mut().choose(&mut r) {
        *ov = v;
    }
}

// Mutates using the given function for each element, using |rate| to decide to mutate or not.
pub fn mutate_rate<T: Copy>(s: &mut [T], rate: f64, mut f: impl FnMut(T) -> T) {
    let mut r = rand::rng();
    for v in s {
        if r.random::<f64>() < rate {
            *v = f(*v);
        }
    }
}

// Real mutation operators  ////////////////////////////////////////////////

// Random value taken from the uniform distribution on |range|.
#[must_use]
pub fn mutate_uniform(st: f64, en: f64) -> f64 {
    let mut r = rand::rng();
    r.random_range(st..=en)
}

// Mutate |v| by a value from N(0, std). It's usual to use the mutation rate as |std|.
// May want to clamp the value to a range afterwards.
#[must_use]
pub fn mutate_normal(v: f64, std: f64) -> f64 {
    let mut r = rand::rng();
    v + std * r.sample::<f64, _>(StandardNormal)
}

// Mutate s.t. v' = v * e^(std * N(0, 1)).
// May want to clamp the value to a range afterwards.
#[must_use]
pub fn mutate_lognorm(v: f64, std: f64) -> f64 {
    let mut r = rand::rng();
    v * E.powf(std * r.sample::<f64, _>(StandardNormal))
}

// Number mutation operators:
pub fn mutate_creep<T: Num + Saturating + SampleUniform + PartialOrd>(v: T, max_diff: T) -> T {
    let mut r = rand::rng();
    let diff = r.random_range(T::zero()..max_diff);
    if r.random::<bool>() { v.saturating_sub(diff) } else { v.saturating_add(diff) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn test_mutate_swap() {
        let mut v = vec![1, 2, 3, 4, 5];
        mutate_swap(&mut v);
        // Should still contain same elements
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_mutate_insert() {
        let mut v = vec![1, 2, 3, 4, 5];
        mutate_insert(&mut v);
        // Should still contain same elements
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_mutate_scramble() {
        let mut v = vec![1, 2, 3, 4, 5];
        mutate_scramble(&mut v);
        // Should still contain same elements
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_mutate_inversion() {
        let mut v = vec![1, 2, 3, 4, 5];
        mutate_inversion(&mut v);
        // Should still contain same elements
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_mutate_reset() {
        let mut v = vec![1, 2, 3, 4, 5];
        mutate_reset(&mut v, 99);
        // Should contain exactly one 99
        assert_eq!(v.iter().filter(|&&x| x == 99).count(), 1);
        // Other values should be from original
        assert_eq!(v.len(), 5);
    }

    #[test]
    fn test_mutate_rate_zero() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let original = v.clone();
        mutate_rate(&mut v, 0.0, |x| x + 10.0);
        // With rate 0, nothing should change
        assert_eq!(v, original);
    }

    #[test]
    fn test_mutate_rate_one() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        mutate_rate(&mut v, 1.0, |x| x + 10.0);
        // With rate 1, all should change
        assert_eq!(v, vec![11.0, 12.0, 13.0, 14.0, 15.0]);
    }

    #[test]
    fn test_mutate_rate_half() {
        let mut v = vec![1.0; 100];
        mutate_rate(&mut v, 0.5, |x| x + 1.0);
        // With rate 0.5, approximately half should change
        let changed = v.iter().filter(|&&x| x == 2.0).count();
        assert!(changed > 30 && changed < 70); // Roughly half with some variance
    }

    #[test]
    fn test_mutate_uniform() {
        let result = mutate_uniform(0.0, 10.0);
        assert!(result >= 0.0 && result <= 10.0);
    }

    #[test]
    fn test_mutate_normal() {
        let v = 5.0;
        let result = mutate_normal(v, 1.0);
        // Result should be near v, but this is probabilistic
        assert!(result.is_finite());
    }

    #[test]
    fn test_mutate_lognorm() {
        let v = 5.0;
        let result = mutate_lognorm(v, 0.5);
        // Result should be positive
        assert!(result > 0.0 && result.is_finite());
    }

    #[test]
    fn test_mutate_lognorm_preserves_sign() {
        let v = 10.0;
        let result = mutate_lognorm(v, 0.1);
        // Should remain positive
        assert!(result > 0.0);
    }

    #[test]
    fn test_mutate_creep_i32() {
        let v = 50i32;
        let result = mutate_creep(v, 10i32);
        // Should be within 10 of original
        assert!((result - v).abs() <= 10);
    }

    #[test]
    fn test_mutate_creep_saturating() {
        let v: u8 = 250;
        let result = mutate_creep(v, 10);
        // Should stay within bounds (can go up or down)
        assert!(result >= 240 && result <= 255);
    }

    #[test]
    fn test_mutate_creep_saturating_low() {
        let v: u8 = 5;
        let result = mutate_creep(v, 10);
        // Should not underflow (saturates at 0) or overflow
        assert!(result <= 15);
    }

    #[test]
    fn test_mutate_normal_zero_std() {
        let v = 5.0;
        let result = mutate_normal(v, 0.0);
        // With std=0, should be very close to original
        assert!((result - v).abs() < 0.1); // Small tolerance for floating point
    }

    #[test]
    fn test_mutate_scramble_preserves_elements() {
        let mut v = vec!['a', 'b', 'c', 'd', 'e'];
        mutate_scramble(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec!['a', 'b', 'c', 'd', 'e']);
    }

    #[test]
    fn test_mutate_inversion_preserves_elements() {
        let mut v = vec!['a', 'b', 'c', 'd', 'e'];
        mutate_inversion(&mut v);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec!['a', 'b', 'c', 'd', 'e']);
    }
}
