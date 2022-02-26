use std::f64::consts::E;

use num_traits::{Num, Saturating};
use rand::prelude::{IteratorRandom, SliceRandom};
use rand::Rng;
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, Standard, StandardNormal};

// Permutation mutation operators ////////////////////////////////////////////////

// Mutate by swapping
pub fn mutate_swap<T: Copy>(s: &mut [T]) {
    let mut r = rand::thread_rng();
    s.swap(r.gen_range(0..s.len()), r.gen_range(0..s.len()));
}

// Mutate by making two random elements next to each-other, shuffling the
// elements in between. E.g. AbcdEfg => bcdAEfg
pub fn mutate_insert<T: Copy>(s: &mut [T]) {
    let mut r = rand::thread_rng();
    let st = r.gen_range(0..s.len());
    let en = r.gen_range(st..s.len());
    for i in st..en {
        s.swap(i, i + 1);
    }
}

// Mutate by scrambling a random substring of the input. e.g. aBCDefg => aCDBefg
pub fn mutate_scramble<T: Copy>(s: &mut [T]) {
    let mut r = rand::thread_rng();
    let st = r.gen_range(0..s.len());
    let en = r.gen_range(st..s.len());
    s[st..=en].shuffle(&mut r);
}

// Mutate by inverting a random substring of the input, e.g. aBCDefg => aDCBefg.
// For adjacency-based problems this is the smallest mutation - it only affects
// two edges (the ends where the inversion happens).
pub fn mutate_inversion<T: Copy>(s: &mut [T]) {
    let mut r = rand::thread_rng();
    let st = r.gen_range(0..s.len());
    let en = r.gen_range(st..s.len());
    s[st..=en].reverse();
}

// Discrete mutation operators ////////////////////////////////////////////////

// Generates a random value.
#[must_use]
pub fn mutate_gen<T>() -> T
where
    Standard: Distribution<T>,
{
    let mut r = rand::thread_rng();
    r.gen::<T>()
}

// Replaces a random value in |s| with |v|.
pub fn mutate_reset<T>(s: &mut [T], v: T) {
    let mut r = rand::thread_rng();
    if let Some(ov) = s.iter_mut().choose(&mut r) {
        *ov = v;
    }
}

// Mutates using the given function for each element, using |rate| to decide to mutate or not.
pub fn mutate_rate<T: Copy>(s: &mut [T], rate: f64, mut f: impl FnMut(T) -> T) {
    let mut r = rand::thread_rng();
    for v in s {
        if r.gen::<f64>() < rate {
            *v = f(*v);
        }
    }
}

// Real mutation operators  ////////////////////////////////////////////////

// Random value taken from the uniform distribution on |range|.
#[must_use]
pub fn mutate_uniform(st: f64, en: f64) -> f64 {
    let mut r = rand::thread_rng();
    r.gen_range(st..=en)
}

// Mutate |v| by a value from N(0, std). It's usual to use the mutation rate as |std|.
// May want to clamp the value to a range afterwards.
#[must_use]
pub fn mutate_normal(v: f64, std: f64) -> f64 {
    let mut r = rand::thread_rng();
    v + std * r.sample::<f64, _>(StandardNormal)
}

// Mutate s.t. v' = v * e^(std * N(0, 1)).
// May want to clamp the value to a range afterwards.
#[must_use]
pub fn mutate_lognorm(v: f64, std: f64) -> f64 {
    let mut r = rand::thread_rng();
    v * E.powf(std * r.sample::<f64, _>(StandardNormal))
}

// Number mutation operators:
pub fn mutate_creep<T: Num + Saturating + SampleUniform + PartialOrd>(v: T, max_diff: T) -> T {
    let mut r = rand::thread_rng();
    let diff = r.gen_range(T::zero()..max_diff);
    if r.gen::<bool>() { v.saturating_sub(diff) } else { v.saturating_add(diff) }
}
