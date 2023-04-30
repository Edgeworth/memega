use std::fmt;
use std::hash::Hash;

use eyre::Result;
use stretto::Cache;

use crate::evolve::cfg::FitnessReduction;

pub trait State = Clone + Send + Sync + PartialOrd + PartialEq + fmt::Display;
pub trait Data = Clone + Send + Sync;
pub trait FitnessFn<S: State, D: Data = ()> = Fn(&S, &D) -> Result<f64> + Sync + Send + Clone;

/// Evaluates, mutates, etc a State.
pub trait Evaluator: Send + Sync {
    type State: State;
    /// For data that should be passed into the fitness function - e.g. if
    /// training on a subset of data e.g. to improve overfitting or because
    /// the fitness function is not the exact goal.
    type Data: Data = ();
    /// Specify the number of crossover operators.
    const NUM_CROSSOVER: usize = 2;
    /// Specify the number of mutation operators.
    const NUM_MUTATION: usize = 1;

    /// |idx| specifies which crossover function to use. 0 is conventionally do nothing,
    /// with actual crossover starting from index 1.
    fn crossover(&self, s1: &mut Self::State, s2: &mut Self::State, idx: usize);

    /// Unlike crossover, mutation is called for every mutation operator. No need for a nop operator.
    fn mutate(&self, s: &mut Self::State, rate: f64, idx: usize);

    fn fitness(&self, s: &Self::State, data: &Self::Data) -> Result<f64>;

    /// Computes fitness over multiple inputs with the given reduction.
    fn multi_fitness(
        &self,
        s: &Self::State,
        inputs: &[Self::Data],
        reduction: FitnessReduction,
    ) -> Result<f64> {
        let mut cumulative = match reduction {
            FitnessReduction::ArithmeticMean => 0.0,
            FitnessReduction::GeometricMean => 1.0,
        };
        for data in inputs {
            let fitness = self.fitness(s, data)?;
            match reduction {
                FitnessReduction::ArithmeticMean => cumulative += fitness,
                FitnessReduction::GeometricMean => cumulative *= fitness,
            }
        }
        let fitness = match reduction {
            FitnessReduction::ArithmeticMean => cumulative / inputs.len() as f64,
            FitnessReduction::GeometricMean => cumulative.powf(1.0 / inputs.len() as f64),
        };
        Ok(fitness)
    }

    fn distance(&self, s1: &Self::State, s2: &Self::State) -> Result<f64>;
}

/// Evaluator which uses an LRU cache to cache fitness and distance values.
#[must_use]
pub struct CachedEvaluator<E: Evaluator>
where
    E::State: Hash + Eq,
    E::Data: Hash + Eq,
{
    eval: E,
    fitness_cache: Cache<(E::State, E::Data), f64>,
}

impl<E: Evaluator> CachedEvaluator<E>
where
    E::State: Hash + Eq + 'static,
    E::Data: Hash + Eq + 'static,
{
    pub fn new(eval: E, cap: usize) -> Self {
        Self { eval, fitness_cache: Cache::new(cap * 10, cap as i64).unwrap() }
    }
}

impl<E: Evaluator> Evaluator for CachedEvaluator<E>
where
    E::State: Hash + Eq + 'static,
    E::Data: Hash + Eq + 'static,
{
    type State = E::State;
    type Data = E::Data;
    const NUM_CROSSOVER: usize = E::NUM_CROSSOVER;
    const NUM_MUTATION: usize = E::NUM_MUTATION;

    fn crossover(&self, s1: &mut Self::State, s2: &mut Self::State, idx: usize) {
        self.eval.crossover(s1, s2, idx);
    }

    fn mutate(&self, s: &mut Self::State, rate: f64, idx: usize) {
        self.eval.mutate(s, rate, idx);
    }

    fn fitness(&self, s: &Self::State, data: &Self::Data) -> Result<f64> {
        let key = (Self::State::clone(s), Self::Data::clone(data));
        if let Some(value) = self.fitness_cache.get(&key) {
            Ok(*value.value())
        } else {
            let value = self.eval.fitness(s, data)?;
            self.fitness_cache.insert(key, value, 1);
            Ok(value)
        }
    }

    fn distance(&self, s1: &Self::State, s2: &Self::State) -> Result<f64> {
        self.eval.distance(s1, s2)
    }
}
