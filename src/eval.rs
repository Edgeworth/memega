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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt;

    #[derive(Debug, Clone, PartialEq, PartialOrd)]
    struct TestState(f64);

    impl fmt::Display for TestState {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    struct TestEvaluator;

    impl Evaluator for TestEvaluator {
        type State = TestState;
        type Data = f64;

        fn crossover(&self, _s1: &mut Self::State, _s2: &mut Self::State, _idx: usize) {}

        fn mutate(&self, _s: &mut Self::State, _rate: f64, _idx: usize) {}

        fn fitness(&self, s: &Self::State, data: &Self::Data) -> Result<f64> {
            Ok(s.0 * data)
        }

        fn distance(&self, s1: &Self::State, s2: &Self::State) -> Result<f64> {
            Ok((s1.0 - s2.0).abs())
        }
    }

    #[test]
    fn test_multi_fitness_arithmetic_mean() {
        let eval = TestEvaluator;
        let state = TestState(2.0);
        let inputs = vec![1.0, 2.0, 3.0, 4.0];
        let result = eval.multi_fitness(&state, &inputs, FitnessReduction::ArithmeticMean).unwrap();
        // (2*1 + 2*2 + 2*3 + 2*4) / 4 = (2 + 4 + 6 + 8) / 4 = 20 / 4 = 5.0
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_multi_fitness_geometric_mean() {
        let eval = TestEvaluator;
        let state = TestState(2.0);
        let inputs = vec![1.0, 2.0, 4.0, 8.0];
        let result = eval.multi_fitness(&state, &inputs, FitnessReduction::GeometricMean).unwrap();
        // (2*1 * 2*2 * 2*4 * 2*8)^(1/4) = (2 * 4 * 8 * 16)^(1/4) = 1024^(1/4) = 5.656...
        let expected = (2.0_f64 * 4.0 * 8.0 * 16.0).powf(0.25);
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_multi_fitness_empty_inputs() {
        let eval = TestEvaluator;
        let state = TestState(2.0);
        let inputs: Vec<f64> = vec![];

        // Arithmetic mean with empty inputs
        let result = eval.multi_fitness(&state, &inputs, FitnessReduction::ArithmeticMean);
        assert!(result.is_ok()); // Will produce 0/0 = NaN, but doesn't error

        // Geometric mean with empty inputs
        let result = eval.multi_fitness(&state, &inputs, FitnessReduction::GeometricMean);
        assert!(result.is_ok()); // Will produce 1^inf = 1, but doesn't error
    }

    #[test]
    fn test_multi_fitness_single_input() {
        let eval = TestEvaluator;
        let state = TestState(3.0);

        let result = eval.multi_fitness(&state, &[5.0], FitnessReduction::ArithmeticMean).unwrap();
        assert!((result - 15.0).abs() < 1e-10);

        let result = eval.multi_fitness(&state, &[5.0], FitnessReduction::GeometricMean).unwrap();
        assert!((result - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_cached_evaluator() {
        use std::sync::Mutex;

        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        struct HashableState(i32);

        impl fmt::Display for HashableState {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        struct CountingEvaluator {
            count: Mutex<usize>,
        }

        impl Evaluator for CountingEvaluator {
            type State = HashableState;
            type Data = i32;

            fn crossover(&self, _s1: &mut Self::State, _s2: &mut Self::State, _idx: usize) {}
            fn mutate(&self, _s: &mut Self::State, _rate: f64, _idx: usize) {}

            fn fitness(&self, s: &Self::State, data: &Self::Data) -> Result<f64> {
                *self.count.lock().unwrap() += 1;
                Ok((s.0 * data) as f64)
            }

            fn distance(&self, s1: &Self::State, s2: &Self::State) -> Result<f64> {
                Ok((s1.0 - s2.0).abs() as f64)
            }
        }

        let eval = CountingEvaluator { count: Mutex::new(0) };
        let cached = CachedEvaluator::new(eval, 100);

        let state = HashableState(5);
        let data = 3;

        // First call - should compute
        let result1 = cached.fitness(&state, &data).unwrap();
        assert_eq!(result1, 15.0);
        let count1 = *cached.eval.count.lock().unwrap();
        assert!(count1 >= 1);

        // Different data - should compute again
        let result2 = cached.fitness(&state, &4).unwrap();
        assert_eq!(result2, 20.0);
        let count2 = *cached.eval.count.lock().unwrap();
        assert!(count2 > count1); // Should have computed again
    }

    #[test]
    fn test_cached_evaluator_delegates() {
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        struct SimpleState(i32);

        impl fmt::Display for SimpleState {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        struct SimpleEvaluator;

        impl Evaluator for SimpleEvaluator {
            type State = SimpleState;
            type Data = ();
            const NUM_CROSSOVER: usize = 3;
            const NUM_MUTATION: usize = 5;

            fn crossover(&self, s1: &mut Self::State, s2: &mut Self::State, _idx: usize) {
                std::mem::swap(&mut s1.0, &mut s2.0);
            }

            fn mutate(&self, s: &mut Self::State, _rate: f64, _idx: usize) {
                s.0 += 1;
            }

            fn fitness(&self, s: &Self::State, _data: &Self::Data) -> Result<f64> {
                Ok(s.0 as f64)
            }

            fn distance(&self, s1: &Self::State, s2: &Self::State) -> Result<f64> {
                Ok((s1.0 - s2.0).abs() as f64)
            }
        }

        let cached = CachedEvaluator::new(SimpleEvaluator, 10);

        // Test constants are delegated
        assert_eq!(CachedEvaluator::<SimpleEvaluator>::NUM_CROSSOVER, 3);
        assert_eq!(CachedEvaluator::<SimpleEvaluator>::NUM_MUTATION, 5);

        // Test crossover is delegated
        let mut s1 = SimpleState(10);
        let mut s2 = SimpleState(20);
        cached.crossover(&mut s1, &mut s2, 0);
        assert_eq!(s1.0, 20);
        assert_eq!(s2.0, 10);

        // Test mutate is delegated
        let mut s = SimpleState(5);
        cached.mutate(&mut s, 1.0, 0);
        assert_eq!(s.0, 6);

        // Test distance is delegated
        let d = cached.distance(&SimpleState(10), &SimpleState(15)).unwrap();
        assert!((d - 5.0).abs() < 1e-10);
    }
}
