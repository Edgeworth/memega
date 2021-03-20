use rand::prelude::IteratorRandom;
use rand::Rng;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::hash::Hash;
use std::mem::swap;

// Permutation crossover operators ////////////////////////////////////////////

// Partially mapped crossover.
// 1 2 3 | 4 5 6 7 | 8 9  =>  . . . 4 5 6 7 . . => . . . 4 5 6 7 . 8
// 9 3 7 | 8 2 6 5 | 1 4
// Take a random substring from s1 to create child c1. We then want to place the
// rest of the elements from s2 into c1. But we need to find new places for
// elements in s2 that correspond to locations in c1 that we placed, if they
// aren't already in c1.
//
// Since there is already an element placed in c1, the location of the
// corresponding element in s2 will become free for putting something (since we
// already used it in c1), unless it is also in the already copied substring.
// Follow the links until we find a free spot.
//
// Finally, copy elements from s2 into any remaining empty spots in c1.
//
// It's easier to think about this in reverse - for all positions outside of the
// initially copied substring, follow the links in reverse to see if there is an
// element from s2 that was displaced that we should place here.
//
// Then do it with swapped s1 and s2 swapped for c2.
//
// Non-common elements:
//   If there are non-common elements between s1 and s2, they effectively end up
//   undergoing 2 point crossover (not touched by the partially mapped crossover,
//   except for the initial 2-point crossover).
//
// s1 and s2 must have the same length.
pub fn crossover_pmx<T: Copy + Hash + Default + Eq>(s1: &mut [T], s2: &mut [T]) {
    let mut r = rand::thread_rng();
    let mut i0 = r.gen_range(0..s1.len());
    let mut i1 = r.gen_range(0..s1.len());
    if i0 > i1 {
        swap(&mut i0, &mut i1);
    }
    let c1 = crossover_pmx_single(s1, s2, i0, i1);
    let c2 = crossover_pmx_single(s2, s1, i0, i1);
    s1.copy_from_slice(&c1);
    s2.copy_from_slice(&c2);
}

pub fn crossover_pmx_single<T: Copy + Hash + Default + Eq>(
    s1: &[T],
    s2: &[T],
    i0: usize,
    i1: usize,
) -> Vec<T> {
    let mut c1 = vec![Default::default(); s1.len()];
    let mut m: HashMap<T, T> = HashMap::new();
    for i in i0..=i1 {
        c1[i] = s1[i]; // Copy substring from s1 into c1.
        m.entry(s1[i]).or_insert(s2[i]); // Map from s1 => s2.
    }

    // Find new locations for items in s2 that were displaced by stuff copied
    // into c1.
    for i in (0..i0).chain((i1 + 1)..s1.len()) {
        let mut ins = s2[i];
        // Try looking up
        let mut count = 0;
        while let Some(&next) = m.get(&ins) {
            // println!("{} => {}", ins, next);
            ins = next;
            count += 1;
            if count > s1.len() {
                ins = s2[i];
                break; // Cycle detected, break out.
            }
        }
        c1[i] = ins;
    }

    c1
}

// Non-common elements:
//
// s1 and s2 must have the same length.
pub fn crossover_edge<T>(s1: &mut [T], s2: &mut [T]) {
    let mut r = rand::thread_rng();
}

// Non-common elements:
//
// s1 and s2 must have the same length.
pub fn crossover_order<T>(s1: &mut [T], s2: &mut [T]) {
    let mut r = rand::thread_rng();
}

// Non-common elements:
//
// Duplicate elements:
//
// s1 and s2 must have the same length.
pub fn crossover_cycle<T>(s1: &mut [T], s2: &mut [T]) {
    let mut r = rand::thread_rng();
}

// Discrete crossover operators  //////////////////////////////////////////////

// Random point K-point crossover.
pub fn crossover_kpx<T>(s1: &mut [T], s2: &mut [T], k: usize) {
    let mut r = rand::thread_rng();
    let xpoints = (0..s1.len()).choose_multiple(&mut r, k);
    crossover_kpx_pts(s1, s2, &xpoints)
}

// K-point crossover.
pub fn crossover_kpx_pts<T>(s1: &mut [T], s2: &mut [T], xpoints: &[usize]) {
    let mut xpoints: SmallVec<[usize; 4]> = SmallVec::from_slice(xpoints);
    let min = s1.len().min(s2.len());
    xpoints.push(min);
    xpoints.sort_unstable();
    for &[st, en] in xpoints.array_chunks::<2>() {
        for i in st..en {
            std::mem::swap(&mut s1[i], &mut s2[i]);
        }
    }
}

// Uniform crossover.
pub fn crossover_ux<T>(s1: &mut [T], s2: &mut [T]) {
    let mut r = rand::thread_rng();
    crossover_ux_rng(s1, s2, &mut r);
}

pub fn crossover_ux_rng<T, R: Rng + ?Sized>(s1: &mut [T], s2: &mut [T], r: &mut R) {
    let min = s1.len().min(s2.len());
    for i in 0..min {
        if r.gen::<bool>() {
            std::mem::swap(&mut s1[i], &mut s2[i]);
        }
    }
}

// Real crossover operators  ////////////////////////////////////////////////

// Whole arithemtic recombination. This takes the linear combination between
// the two states for each element.
pub fn crossover_arith_alpha(s1: &mut [f64], s2: &mut [f64], alpha: f64) {
    let min = s1.len().min(s2.len());
    for i in 0..min {
        let c1 = alpha * s1[i] + (1.0 - alpha) * s2[i];
        let c2 = alpha * s2[i] + (1.0 - alpha) * s1[i];
        (s1[i], s2[i]) = (c1, c2);
    }
}

// Whole arithmetic recombination with a random combination multiplier.
pub fn crossover_arith(s1: &mut [f64], s2: &mut [f64]) {
    let mut r = rand::thread_rng();
    crossover_arith_alpha(s1, s2, r.gen())
}

// Blend crossover. For each element x < y, randomly generate a value in
// [x - |y - x| * alpha, y + |y - x| * alpha]. A good choice for alpha is 0.5.
pub fn crossover_blx(s1: &mut [f64], s2: &mut [f64], alpha: f64) {
    let mut r = rand::thread_rng();
    let min = s1.len().min(s2.len());
    for i in 0..min {
        let x = s1[i].min(s2[i]);
        let y = s1[i].max(s2[i]);
        let dist = y - x;
        let left = x - dist * alpha;
        let right = y + dist * alpha;
        s1[i] = r.gen_range(left..=right);
        s2[i] = r.gen_range(left..=right);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::util::{str_to_vec, vec_to_str};
    use rand::rngs::mock::StepRng;

    #[test]
    fn test_crossover_pmx_int() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let b = [9, 3, 7, 8, 2, 6, 5, 1, 4];
        assert_eq!(
            crossover_pmx_single(&a, &b, 3, 6),
            [9, 3, 2, 4, 5, 6, 7, 1, 8]
        );
    }

    #[test]
    fn test_crossover_pmx_str() {
        let a = str_to_vec("abcdefghi");
        let b = str_to_vec("icghbfead");
        assert_eq!(vec_to_str(&crossover_pmx_single(&a, &b, 3, 6)), "icbdefgah");
    }

    #[test]
    fn test_crossover_pmx_dupes() {
        let a = [1, 1, 1, 1, 1];
        let b = [1, 1, 1, 1, 1];
        assert_eq!(crossover_pmx_single(&a, &b, 1, 3), [1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_crossover_pmx_dups_non_common() {
        let a = [1, 2, 3, 1, 1];
        let b = [1, 1, 4, 5, 6];
        assert_eq!(crossover_pmx_single(&a, &b, 1, 3), [5, 2, 3, 1, 6]);
    }

    #[test]
    fn test_crossover_1px() {
        let mut a = str_to_vec("abcd");
        let mut b = str_to_vec("wxyz");
        crossover_kpx_pts(&mut a, &mut b, &[3]);
        assert_eq!(vec_to_str(&a), "abcz");
        assert_eq!(vec_to_str(&b), "wxyd");
    }

    #[test]
    fn test_crossover_2px() {
        let mut a = str_to_vec("abcd");
        let mut b = str_to_vec("wxyz");
        crossover_kpx_pts(&mut a, &mut b, &[1, 2]);
        assert_eq!(vec_to_str(&a), "axcd");
        assert_eq!(vec_to_str(&b), "wbyz");
    }

    #[test]
    fn test_crossover_ux() {
        let mut r = StepRng::new(1 << 31, 1 << 31);
        let mut a = str_to_vec("abcd");
        let mut b = str_to_vec("wxyz");
        crossover_ux_rng(&mut a, &mut b, &mut r);
        assert_eq!(vec_to_str(&a), "wbyd");
        assert_eq!(vec_to_str(&b), "axcz");
    }
}
