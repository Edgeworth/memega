use rand::Rng;
use rand::prelude::IteratorRandom;

// Roulette wheel selection:
#[must_use]
pub fn rws(w: &[f64]) -> Option<usize> {
    multi_rws(w, 1).first().copied()
}

pub fn rws_rng<R: Rng + ?Sized>(w: &[f64], r: &mut R) -> Option<usize> {
    multi_rws_rng(w, 1, r).first().copied()
}

#[must_use]
pub fn multi_rws(w: &[f64], k: usize) -> Vec<usize> {
    let mut r = rand::rng();
    multi_rws_rng(w, k, &mut r)
}

pub fn multi_rws_rng<R: Rng + ?Sized>(w: &[f64], k: usize, r: &mut R) -> Vec<usize> {
    let sum: f64 = w.iter().sum();
    if sum == 0.0 {
        return (0..w.len()).choose_multiple(r, k);
    }

    let mut idxs = Vec::new();
    for _ in 0..k {
        let cursor = r.random_range(0.0..=sum);
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
#[must_use]
pub fn sus(w: &[f64], k: usize) -> Vec<usize> {
    let mut r = rand::rng();
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
    let mut cursor = r.random_range(0.0..=step);
    for _ in 0..k {
        while idx < w.len() && cursum + w[idx] < cursor {
            cursum += w[idx];
            idx += 1;
        }
        idxs.push(idx.min(w.len() - 1));
        cursor += step;
    }
    idxs
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;

    #[test]
    fn test_rws() {
        let mut r = StdRng::seed_from_u64(0);
        assert_eq!(rws_rng(&[], &mut r), None);
        assert_eq!(rws_rng(&[1.0], &mut r), Some(0));
        assert_eq!(rws_rng(&[0.0, 1.0], &mut r), Some(1));
    }

    #[test]
    fn test_multi_rws() {
        let mut r = StdRng::seed_from_u64(0);
        assert_eq!(multi_rws_rng(&[], 0, &mut r), []);
        assert_eq!(multi_rws_rng(&[], 1, &mut r), []);
        assert_eq!(multi_rws_rng(&[1.0], 0, &mut r), []);
        assert_eq!(multi_rws_rng(&[1.0], 1, &mut r), [0]);
        assert_eq!(multi_rws_rng(&[1.0], 1, &mut r), [0]);
        assert_eq!(multi_rws_rng(&[0.0, 1.0], 1, &mut r), [1]);
    }

    #[test]
    fn test_sus() {
        let mut r = StdRng::seed_from_u64(0);
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

    #[test]
    fn test_sus_many_samples() {
        let mut r = StdRng::seed_from_u64(42);
        // Test with k > weights.len()
        let result = sus_rng(&[1.0, 2.0, 3.0], 10, &mut r);
        assert_eq!(result.len(), 10);
        // All indices should be valid
        assert!(result.iter().all(|&idx| idx < 3));
    }

    #[test]
    fn test_sus_large_weights() {
        let mut r = StdRng::seed_from_u64(123);
        let weights = vec![100.0, 200.0, 300.0, 400.0];
        let result = sus_rng(&weights, 20, &mut r);
        assert_eq!(result.len(), 20);
        // Check distribution is reasonable
        let count_3 = result.iter().filter(|&&x| x == 3).count();
        assert!(count_3 > 5); // Highest weight should appear most
    }

    #[test]
    fn test_sus_small_weights() {
        let mut r = StdRng::seed_from_u64(456);
        let weights = vec![0.001, 0.002, 0.003];
        let result = sus_rng(&weights, 5, &mut r);
        assert_eq!(result.len(), 5);
        assert!(result.iter().all(|&idx| idx < 3));
    }

    #[test]
    fn test_rws_many_samples() {
        let mut r = StdRng::seed_from_u64(789);
        let result = multi_rws_rng(&[1.0, 2.0, 3.0], 10, &mut r);
        assert_eq!(result.len(), 10);
        assert!(result.iter().all(|&idx| idx < 3));
    }

    #[test]
    fn test_rws_all_zeros() {
        let mut r = StdRng::seed_from_u64(999);
        let result = multi_rws_rng(&[0.0, 0.0, 0.0], 5, &mut r);
        // When all weights are 0, falls back to random choice
        // Can only return up to the number of elements
        assert!(result.len() <= 5);
        assert!(result.iter().all(|&idx| idx < 3));
    }

    #[test]
    fn test_sus_all_zeros() {
        let mut r = StdRng::seed_from_u64(111);
        let result = sus_rng(&[0.0, 0.0, 0.0], 5, &mut r);
        // When all weights are 0, falls back to random choice
        // Can only return up to the number of elements
        assert!(result.len() <= 5);
        assert!(result.iter().all(|&idx| idx < 3));
    }

    #[test]
    fn test_rws_single_large_weight() {
        let mut r = StdRng::seed_from_u64(222);
        let result = multi_rws_rng(&[1000.0, 1.0, 1.0], 10, &mut r);
        assert_eq!(result.len(), 10);
        // Should mostly select index 0
        let count_0 = result.iter().filter(|&&x| x == 0).count();
        assert!(count_0 >= 8);
    }

    #[test]
    fn test_sus_uniform_weights() {
        let mut r = StdRng::seed_from_u64(333);
        let result = sus_rng(&[1.0, 1.0, 1.0, 1.0], 8, &mut r);
        assert_eq!(result.len(), 8);
        // Each index should appear approximately equally
        for i in 0..4 {
            let count = result.iter().filter(|&&x| x == i).count();
            assert!(count >= 1 && count <= 3);
        }
    }
}
