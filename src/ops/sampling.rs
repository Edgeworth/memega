use rand::prelude::IteratorRandom;
use rand::Rng;
use std::iter::Iterator;

// Roulette wheel selection:
pub fn rws(w: &[f64]) -> Option<usize> {
    multi_rws(w, 1).get(0).copied()
}

pub fn rws_rng<R: Rng + ?Sized>(w: &[f64], r: &mut R) -> Option<usize> {
    multi_rws_rng(w, 1, r).get(0).copied()
}

pub fn multi_rws(w: &[f64], k: usize) -> Vec<usize> {
    let mut r = rand::thread_rng();
    multi_rws_rng(w, k, &mut r)
}

pub fn multi_rws_rng<R: Rng + ?Sized>(w: &[f64], k: usize, r: &mut R) -> Vec<usize> {
    let sum = w.iter().sum();
    if sum == 0.0 {
        return (0..w.len()).choose_multiple(r, k);
    }

    let mut idxs = Vec::new();
    for _ in 0..k {
        let cursor = r.gen_range(0.0..=sum);
        let mut cursum = 0.0;
        for (i, v) in w.iter().enumerate() {
            cursum += v;
            if cursum >= cursor {
                idxs.push(i);
                break;
            }
        }
    }
    idxs
}

// Stochastic universal sampling:
pub fn sus(w: &[f64], k: usize) -> Vec<usize> {
    let mut r = rand::thread_rng();
    sus_rng(w, k, &mut r)
}

pub fn sus_rng<R: Rng + ?Sized>(w: &[f64], k: usize, r: &mut R) -> Vec<usize> {
    let sum: f64 = w.iter().sum();
    if k == 0 {
        return vec![];
    }
    if sum == 0.0 {
        return (0..w.len()).choose_multiple(r, k);
    }
    let step = sum / k as f64;
    let mut idxs = Vec::new();
    let mut idx = 0;
    let mut cursum = 0.0;
    let mut cursor = r.gen_range(0.0..=step);
    for _ in 0..k {
        while cursum + w[idx] < cursor {
            cursum += w[idx];
            idx += 1;
        }
        idxs.push(idx);
        cursor += step;
    }
    idxs
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::mock::StepRng;

    #[test]
    fn test_rws() {
        let mut r = StepRng::new(1 << 31, 1 << 31);
        assert_eq!(rws_rng(&[], &mut r), None);
        assert_eq!(rws_rng(&[1.0], &mut r), Some(0));
        assert_eq!(rws_rng(&[0.0, 1.0], &mut r), Some(1));
    }

    #[test]
    fn test_multi_rws() {
        let mut r = StepRng::new(1 << 31, 1 << 31);
        assert_eq!(multi_rws_rng(&[], 0, &mut r), []);
        assert_eq!(multi_rws_rng(&[], 1, &mut r), []);
        assert_eq!(multi_rws_rng(&[1.0], 0, &mut r), []);
        assert_eq!(multi_rws_rng(&[1.0], 1, &mut r), [0]);
        assert_eq!(multi_rws_rng(&[1.0], 1, &mut r), [0]);
        assert_eq!(multi_rws_rng(&[0.0, 1.0], 1, &mut r), [1]);
    }

    #[test]
    fn test_sus() {
        let mut r = StepRng::new(1 << 31, 1 << 31);
        assert_eq!(sus_rng(&[], 0, &mut r), []);
        assert_eq!(sus_rng(&[], 1, &mut r), []);
        assert_eq!(sus_rng(&[1.0], 0, &mut r), []);
        assert_eq!(sus_rng(&[1.0], 1, &mut r), [0]);
        assert_eq!(sus_rng(&[1.0], 1, &mut r), [0]);
        assert_eq!(sus_rng(&[1.0, 1.0], 1, &mut r), [0]);
        assert_eq!(sus_rng(&[0.0, 1.0], 1, &mut r), [1]);
        assert_eq!(sus_rng(&[1.0, 1.0], 2, &mut r), [0, 1]);
        assert_eq!(sus_rng(&[1.0, 2.0], 3, &mut r), [0, 1, 1]);
    }
}
