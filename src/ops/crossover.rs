use rand::prelude::IteratorRandom;
use rand::Rng;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::mem::swap;

// Permutation crossover operators ////////////////////////////////////////////

// Partially mapped crossover. Good for permutations where adjacency is important.
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
// s1 and s2 must have the same length.
pub fn crossover_pmx<T: Copy + Hash + Default + Eq>(s1: &mut [T], s2: &mut [T]) {
    let mut r = rand::thread_rng();
    let mut st = r.gen_range(0..s1.len());
    let mut en = r.gen_range(0..s1.len());
    if st > en {
        swap(&mut st, &mut en);
    }
    let c1 = crossover_pmx_single(s1, s2, st, en);
    let c2 = crossover_pmx_single(s2, s1, st, en);
    s1.copy_from_slice(&c1);
    s2.copy_from_slice(&c2);
}

// en is inclusive.
pub fn crossover_pmx_single<T: Copy + Hash + Default + Eq>(
    s1: &[T],
    s2: &[T],
    st: usize,
    en: usize,
) -> Vec<T> {
    if s1.is_empty() {
        return vec![];
    }

    let mut c1 = vec![Default::default(); s1.len()];
    let mut m: HashMap<T, T> = HashMap::new();
    for i in st..=en {
        c1[i] = s1[i]; // Copy substring from s1 into c1.
        m.entry(s1[i]).or_insert(s2[i]); // Map from s1 => s2.
    }

    // Find new locations for items in s2 that were displaced by stuff copied
    // into c1.
    for i in (0..st).chain((en + 1)..s1.len()) {
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

// Order type crossover. Useful for permutations where relative order
// information is more important.
//
// This copies a random substring from s1 into c1. Then fills the remaining
// places starting after the substring ends in c1 and wrapping around with
// unused values from s2.
//
// s1 and s2 must have the same length.
pub fn crossover_order<T: Copy + Hash + Default + Eq>(s1: &mut [T], s2: &mut [T]) {
    let mut r = rand::thread_rng();
    let mut st = r.gen_range(0..s1.len());
    let mut en = r.gen_range(0..s1.len());
    if st > en {
        swap(&mut st, &mut en);
    }
    let c1 = crossover_order_single(s1, s2, st, en);
    let c2 = crossover_order_single(s2, s1, st, en);
    s1.copy_from_slice(&c1);
    s2.copy_from_slice(&c2);
}

// en is inclusive.
pub fn crossover_order_single<T: Copy + Hash + Default + Eq>(
    s1: &[T],
    s2: &[T],
    st: usize,
    en: usize,
) -> Vec<T> {
    if s1.is_empty() {
        return vec![];
    }

    let mut c1 = vec![Default::default(); s1.len()];
    let mut m: HashSet<T> = HashSet::new();
    for i in st..=en {
        c1[i] = s1[i]; // Copy substring from s1 into c1.
        m.insert(s1[i]); // Record stuff already in c1.
    }

    // Add elements from s2 in order after en to c1.
    let mut cur_idx = (en + 1) % c1.len();
    for i in 0..s2.len() {
        // If there are non-common elements in s2, we might have too many
        // elements and need to break early.
        if cur_idx == st {
            break;
        }

        // Ignore elements already in c1.
        let v = s2[(en + 1 + i) % c1.len()];
        if m.contains(&v) {
            continue;
        }
        c1[cur_idx] = v;
        cur_idx = (cur_idx + 1) % c1.len();
    }

    // If there were duplicates, we might not be done. This time don't
    // check for duplication from s2.
    for _ in 0..s2.len() {
        if cur_idx == st {
            break;
        }
        c1[cur_idx] = s2[cur_idx];
        cur_idx = (cur_idx + 1) % c1.len();
    }
    c1
}

// Cycle type crossover. Good for when absolute position of elements is
// important.
//
// Works by finding all cycles between the two strings and alternately
// assigning the contents of each cycle to the two children.
//
// s1 and s2 must have the same length.
pub fn crossover_cycle<T: Copy + Hash + Default + Eq>(s1: &mut [T], s2: &mut [T]) {
    let mut c1: Vec<T> = vec![Default::default(); s1.len()];
    let mut c2: Vec<T> = vec![Default::default(); s1.len()];
    // Build map from values in s1 to positions.
    let mut m: HashMap<T, usize> = HashMap::new();
    for i in 0..s2.len() {
        m.entry(s1[i]).or_insert(i);
    }
    let mut seen = vec![false; s1.len()];
    for i in 0..s1.len() {
        // Already placed into c1 and c2 as part of a cycle.
        if seen[i] {
            continue;
        }
        let mut idx = i;
        // Avoid infinite loop with duplicates by checking seen.
        while !seen[idx] {
            c1[idx] = s1[idx];
            c2[idx] = s2[idx];
            seen[idx] = true;
            if let Some(&next) = m.get(&s2[idx]) {
                idx = next; // Follow cycle link.
            }
        }
        swap(&mut c1, &mut c2); // Alternate cycle data between children.
    }
    s1.copy_from_slice(&c2);
    s2.copy_from_slice(&c1);
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
    fn test_crossover_pmx() {
        let a: [i32; 0] = [];
        let b: [i32; 0] = [];
        assert_eq!(crossover_pmx_single(&a, &b, 0, 0), []);

        let a = [1];
        let b = [1];
        assert_eq!(crossover_pmx_single(&a, &b, 0, 0), [1]);

        let a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let b = [9, 3, 7, 8, 2, 6, 5, 1, 4];
        assert_eq!(
            crossover_pmx_single(&a, &b, 3, 6),
            [9, 3, 2, 4, 5, 6, 7, 1, 8]
        );

        let a = str_to_vec("abcdefghi");
        let b = str_to_vec("icghbfead");
        assert_eq!(vec_to_str(&crossover_pmx_single(&a, &b, 3, 6)), "icbdefgah");

        let a = [1, 1, 1, 1, 1];
        let b = [1, 1, 1, 1, 1];
        assert_eq!(crossover_pmx_single(&a, &b, 1, 3), [1, 1, 1, 1, 1]);

        let a = [1, 2, 3, 1, 1];
        let b = [1, 1, 4, 5, 6];
        assert_eq!(crossover_pmx_single(&a, &b, 1, 3), [5, 2, 3, 1, 6]);
    }

    #[test]
    fn test_crossover_order() {
        let a: [i32; 0] = [];
        let b: [i32; 0] = [];
        assert_eq!(crossover_order_single(&a, &b, 0, 0), []);

        let a = [1];
        let b = [1];
        assert_eq!(crossover_order_single(&a, &b, 0, 0), [1]);

        let a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let b = [9, 3, 7, 8, 2, 6, 5, 1, 4];
        assert_eq!(
            crossover_order_single(&a, &b, 3, 6),
            [3, 8, 2, 4, 5, 6, 7, 1, 9]
        );

        let a = str_to_vec("abcdefghi");
        let b = str_to_vec("icghbfead");
        assert_eq!(
            vec_to_str(&crossover_order_single(&a, &b, 3, 6)),
            "chbdefgai"
        );

        let a = [1, 1, 1, 1, 1];
        let b = [1, 1, 1, 1, 1];
        assert_eq!(crossover_order_single(&a, &b, 1, 3), [1, 1, 1, 1, 1]);

        let a = [1, 2, 3, 1, 1];
        let b = [1, 1, 4, 5, 6];
        assert_eq!(crossover_order_single(&a, &b, 1, 3), [4, 2, 3, 1, 6]);
    }

    #[test]
    fn test_crossover_cycle() {
        let mut a: [i32; 0] = [];
        let mut b: [i32; 0] = [];
        crossover_cycle(&mut a, &mut b);
        assert_eq!(a, []);
        assert_eq!(b, []);

        let mut a = [1];
        let mut b = [1];
        crossover_cycle(&mut a, &mut b);
        assert_eq!(a, [1]);
        assert_eq!(b, [1]);

        let mut a = [1];
        let mut b = [2];
        crossover_cycle(&mut a, &mut b);
        assert_eq!(a, [1]);
        assert_eq!(b, [2]);

        let mut a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut b = [9, 3, 7, 8, 2, 6, 5, 1, 4];
        crossover_cycle(&mut a, &mut b);
        assert_eq!(a, [1, 3, 7, 4, 2, 6, 5, 8, 9]);
        assert_eq!(b, [9, 2, 3, 8, 5, 6, 7, 1, 4]);

        let mut a = str_to_vec("abcdefghi");
        let mut b = str_to_vec("icghbfead");
        crossover_cycle(&mut a, &mut b);
        assert_eq!(vec_to_str(&a), "acgdbfehi");
        assert_eq!(vec_to_str(&b), "ibchefgad");

        let mut a = [1, 1, 1, 1, 1];
        let mut b = [1, 1, 1, 1, 1];
        crossover_cycle(&mut a, &mut b);
        assert_eq!(a, [1, 1, 1, 1, 1]);
        assert_eq!(b, [1, 1, 1, 1, 1]);

        let mut a = [1, 2, 3, 1, 1];
        let mut b = [1, 1, 4, 5, 6];
        crossover_cycle(&mut a, &mut b);
        assert_eq!(a, [1, 1, 3, 5, 1]);
        assert_eq!(b, [1, 2, 4, 1, 6]);
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
