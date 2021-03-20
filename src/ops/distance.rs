use num_traits::{Num, NumAssign};
use std::mem::swap;

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
