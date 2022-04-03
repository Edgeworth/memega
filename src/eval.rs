use std::fmt;
use std::hash::Hash;

use concurrent_lru::sharded::LruCache;

pub trait State = Clone + Send + Sync + PartialOrd + PartialEq + fmt::Display;
pub trait FitnessFn<S: State> = Fn(&S, usize) -> f64 + Sync + Send + Clone;

/// Evaluates, mutates, etc a State.
pub trait Evaluator: Send + Sync {
    type State: State;
    /// Specify the number of crossover operators.
    const NUM_CROSSOVER: usize = 2;
    /// Specify the number of mutation operators.
    const NUM_MUTATION: usize = 1;

    /// |idx| specifies which crossover function to use. 0 is conventionally do nothing,
    /// with actual crossover starting from index 1.
    fn crossover(&self, s1: &mut Self::State, s2: &mut Self::State, idx: usize);

    /// Unlike crossover, mutation is called for every mutation operator. No need for a nop operator.
    fn mutate(&self, s: &mut Self::State, rate: f64, idx: usize);
    fn fitness(&self, s: &Self::State, gen: usize) -> f64;
    fn distance(&self, s1: &Self::State, s2: &Self::State) -> f64;
}

/// Evaluator which uses an LRU cache to cache fitness and distance values.
pub struct CachedEvaluator<E: Evaluator>
where
    E::State: Hash + Eq,
{
    eval: E,
    fitness_cache: LruCache<E::State, f64>,
}

impl<E: Evaluator> CachedEvaluator<E>
where
    E::State: Hash + Eq,
{
    pub fn new(eval: E, cap: usize) -> Self {
        Self { eval, fitness_cache: LruCache::new(cap as u64) }
    }
}

impl<E: Evaluator> Evaluator for CachedEvaluator<E>
where
    E::State: Hash + Eq,
{
    type State = E::State;
    const NUM_CROSSOVER: usize = E::NUM_CROSSOVER;
    const NUM_MUTATION: usize = E::NUM_MUTATION;

    fn crossover(&self, s1: &mut Self::State, s2: &mut Self::State, idx: usize) {
        self.eval.crossover(s1, s2, idx);
    }

    fn mutate(&self, s: &mut Self::State, rate: f64, idx: usize) {
        self.eval.mutate(s, rate, idx);
    }

    fn fitness(&self, s: &Self::State, gen: usize) -> f64 {
        *self.fitness_cache.get_or_init(s.clone(), 1, |s| self.eval.fitness(s, gen)).value()
    }

    fn distance(&self, s1: &Self::State, s2: &Self::State) -> f64 {
        self.eval.distance(s1, s2)
    }
}
