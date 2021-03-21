#![warn(rust_2018_idioms, clippy::all)]
#![feature(
    array_chunks,
    array_windows,
    bool_to_option,
    const_fn,
    destructuring_assignment,
    map_first_last,
    option_result_contains,
    option_unwrap_none,
    stmt_expr_attributes,
    trait_alias,
    type_alias_impl_trait
)]

use crate::gen::Params;
use concurrent_lru::sharded::LruCache;
use std::fmt;
use std::hash::Hash;

pub mod cfg;
pub mod distributions;
pub mod examples;
pub mod gen;
pub mod hyper;
pub mod ops;
pub mod runner;

pub trait Genome = Clone + Send + Sync + PartialOrd + PartialEq + fmt::Debug;
pub trait FitnessFn<G: Genome> = Fn(&G) -> f64 + Sync + Send + Clone;

pub type State<T> = (T, Params);

pub trait Evaluator: Send + Sync {
    type Genome: Genome;
    const NUM_CROSSOVER: usize = 2; // Specify the number of crossover operators.
    const NUM_MUTATION: usize = 1; // Specify the number of mutation operators.

    // |idx| specifies which crossover function to use. 0 is conventionally do nothing,
    // with actual crossover starting from index 1.
    fn crossover(&self, s1: &mut Self::Genome, s2: &mut Self::Genome, idx: usize);

    // Unlike crossover, mutation is called for every mutation operator. No need for a nop operator.
    fn mutate(&self, s: &mut Self::Genome, rate: f64, idx: usize);
    fn fitness(&self, s: &Self::Genome) -> f64;
    fn distance(&self, s1: &Self::Genome, s2: &Self::Genome) -> f64;
}

// Evaluator which uses an LRU cache to cache fitness and distance values.
pub struct CachedEvaluator<E: Evaluator>
where
    E::Genome: Hash + Eq,
{
    eval: E,
    fitness_cache: LruCache<E::Genome, f64>,
}

impl<E: Evaluator> CachedEvaluator<E>
where
    E::Genome: Hash + Eq,
{
    pub fn new(eval: E, cap: usize) -> Self {
        Self {
            eval,
            fitness_cache: LruCache::new(cap as u64),
        }
    }
}

impl<E: Evaluator> Evaluator for CachedEvaluator<E>
where
    E::Genome: Hash + Eq,
{
    type Genome = E::Genome;
    const NUM_CROSSOVER: usize = E::NUM_CROSSOVER;
    const NUM_MUTATION: usize = E::NUM_MUTATION;

    fn crossover(&self, s1: &mut Self::Genome, s2: &mut Self::Genome, idx: usize) {
        self.eval.crossover(s1, s2, idx)
    }

    fn mutate(&self, s: &mut Self::Genome, rate: f64, idx: usize) {
        self.eval.mutate(s, rate, idx)
    }

    fn fitness(&self, s: &Self::Genome) -> f64 {
        *self
            .fitness_cache
            .get_or_init(s.clone(), 1, |s| self.eval.fitness(s))
            .value()
    }

    fn distance(&self, s1: &Self::Genome, s2: &Self::Genome) -> f64 {
        self.eval.distance(s1, s2)
    }
}
