use std::mem::swap;

use eyre::{Result, eyre};
use num_traits::{Num, NumAssign};

// Generalised distance - add missing * difference in lengths distance if the
// arrays are different distances.
pub fn dist_fn<T>(s1: &[T], s2: &[T], missing: f64, mut f: impl FnMut(&T, &T) -> f64) -> f64 {
    let min = s1.len().min(s2.len());
    let mut dist = (s1.len() as f64 - s2.len() as f64).abs() * missing;
    for i in 0..min {
        dist += f(&s1[i], &s2[i]);
    }
    dist
}

// Norm 1 distance
pub fn dist_abs<T: Num + NumAssign + Copy + PartialOrd>(mut a: T, mut b: T) -> T {
    if a < b {
        swap(&mut a, &mut b);
    }
    a - b
}

// Norm 1 distance - manhattan distance.
pub fn dist1<T: Num + NumAssign + Copy + PartialOrd>(s1: &[T], s2: &[T]) -> T {
    let max = s1.len().max(s2.len());
    let mut dist = T::zero();
    for i in 0..max {
        let zero = T::zero();
        let a = s1.get(i).unwrap_or(&zero);
        let b = s2.get(i).unwrap_or(&zero);
        dist += dist_abs(*a, *b);
    }
    dist
}

// Norm 2 distance - euclidean distance.
#[must_use]
pub fn dist2(s1: &[f64], s2: &[f64]) -> f64 {
    let max = s1.len().max(s2.len());
    let mut dist = 0.0;
    for i in 0..max {
        let a = s1.get(i).unwrap_or(&0.0);
        let b = s2.get(i).unwrap_or(&0.0);
        dist += (a - b) * (a - b);
    }
    dist.sqrt()
}

// Number of different pairs
pub fn count_different<T: PartialEq>(s1: &[T], s2: &[T]) -> usize {
    let min = s1.len().min(s2.len());
    let max = s1.len().max(s2.len());
    let mut count = 0;
    for i in 0..min {
        if s1[i] != s2[i] {
            count += 1;
        }
    }
    count + max - min
}

// Kendall tau distance: https://en.wikipedia.org/wiki/Kendall_tau_distance
pub fn kendall_tau<T: PartialOrd>(s1: &[T], s2: &[T]) -> Result<usize> {
    if s1.len() != s2.len() {
        return Err(eyre!("must be same length"));
    }
    let mut count = 0;
    for i in 0..s1.len() {
        for j in (i + 1)..s2.len() {
            if (s1[i] < s1[j]) != (s2[i] < s2[j]) {
                count += 1;
            }
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_count_different() {
        assert_eq!(count_different(&[1], &[1]), 0);
        assert_eq!(count_different(&[1], &[2]), 1);
        assert_eq!(count_different(&[1], &[1, 2]), 1);
        assert_eq!(count_different(&[1, 2], &[1]), 1);
    }

    #[test]
    fn test_kendall_tau() -> Result<()> {
        assert_eq!(kendall_tau(&[1], &[1])?, 0);
        assert_eq!(kendall_tau(&[1], &[2])?, 0);
        assert_eq!(kendall_tau(&[1, 2], &[1, 2])?, 0);
        assert_eq!(kendall_tau(&[1, 2], &[2, 1])?, 1);
        assert_eq!(kendall_tau(&[1, 2, 3, 4, 5], &[3, 4, 1, 2, 5])?, 4);
        Ok(())
    }

    #[test]
    fn test_kendall_tau_different_lengths() {
        let result = kendall_tau(&[1, 2, 3], &[1, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dist_abs() {
        assert_eq!(dist_abs(5, 3), 2);
        assert_eq!(dist_abs(3, 5), 2);
        assert_eq!(dist_abs(10, 10), 0);
    }

    #[test]
    fn test_dist1() {
        assert_eq!(dist1(&[1, 2, 3], &[1, 2, 3]), 0);
        assert_eq!(dist1(&[0, 0, 0], &[1, 1, 1]), 3);
        assert_eq!(dist1(&[1, 2, 3], &[4, 5, 6]), 9);
    }

    #[test]
    fn test_dist1_different_lengths() {
        assert_eq!(dist1(&[1, 2, 3], &[1, 2]), 3);
        assert_eq!(dist1(&[1, 2], &[1, 2, 3]), 3);
    }

    #[test]
    fn test_dist2() {
        assert_eq!(dist2(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
        assert!((dist2(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-10); // 3-4-5 triangle
    }

    #[test]
    fn test_dist2_different_lengths() {
        let result = dist2(&[1.0, 2.0, 3.0], &[1.0, 2.0]);
        assert!((result - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dist_fn() {
        let s1 = vec![1i32, 2, 3];
        let s2 = vec![1i32, 2, 3];
        let result = dist_fn(&s1, &s2, 10.0, |a, b| (*a - *b).abs() as f64);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_dist_fn_with_missing() {
        let s1 = vec![1i32, 2, 3];
        let s2 = vec![1i32, 2];
        let result = dist_fn(&s1, &s2, 10.0, |a, b| (*a - *b).abs() as f64);
        assert_eq!(result, 10.0); // 1 missing element * 10.0
    }

    #[test]
    fn test_dist_fn_custom_function() {
        let s1 = vec![1.0f64, 2.0, 3.0];
        let s2 = vec![2.0f64, 4.0, 6.0];
        let result = dist_fn(&s1, &s2, 0.0, |a, b| (*a - *b).powi(2));
        // (1-2)^2 + (2-4)^2 + (3-6)^2 = 1 + 4 + 9 = 14
        assert!((result - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_count_different_all_same() {
        assert_eq!(count_different(&[1, 2, 3], &[1, 2, 3]), 0);
    }

    #[test]
    fn test_count_different_all_different() {
        assert_eq!(count_different(&[1, 2, 3], &[4, 5, 6]), 3);
    }

}
