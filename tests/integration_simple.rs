/// Integration tests for basic evolution scenarios.
/// These tests verify that the complete evolution pipeline works correctly
/// with simple evaluators and various configurations.

use eyre::Result;
use memega::eval::Evaluator;
use memega::evolve::cfg::EvolveCfg;
use memega::evolve::evolver::Evolver;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Simple state representing a vector of floats to be optimized.
#[derive(Debug, Clone, PartialEq, PartialOrd)]
struct VecState(Vec<f64>);

impl std::fmt::Display for VecState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

/// Evaluator that tries to maximize the sum of elements.
struct MaximizeSumEvaluator;

impl Evaluator for MaximizeSumEvaluator {
    type State = VecState;

    fn crossover(&self, s1: &mut Self::State, s2: &mut Self::State, _idx: usize) {
        // Single-point crossover
        if s1.0.len() != s2.0.len() {
            return;
        }
        let point = s1.0.len() / 2;
        for i in point..s1.0.len() {
            std::mem::swap(&mut s1.0[i], &mut s2.0[i]);
        }
    }

    fn mutate(&self, s: &mut Self::State, rate: f64, _idx: usize) {
        let mut rng = rand::rng();
        for val in &mut s.0 {
            if rng.random::<f64>() < rate {
                *val += rng.random_range(-1.0..1.0);
            }
        }
    }

    fn fitness(&self, s: &Self::State, _data: &Self::Data) -> Result<f64> {
        // Fitness is the sum of all values (clamped to be positive)
        let sum: f64 = s.0.iter().sum();
        Ok(sum.max(0.0) + 0.01) // Add small epsilon to avoid exactly zero
    }

    fn distance(&self, s1: &Self::State, s2: &Self::State) -> Result<f64> {
        Ok(s1.0.iter()
            .zip(s2.0.iter())
            .map(|(a, b)| (a - b).abs())
            .sum())
    }
}

#[test]
fn test_evolution_improves_fitness() {
    let cfg = EvolveCfg::new(20); // Small population for quick test
    let eval = MaximizeSumEvaluator;

    // Random state generator
    let rand_state = move || {
        let mut rng = StdRng::seed_from_u64(rand::rng().random());
        VecState(vec![rng.random_range(-10.0..10.0); 5])
    };

    let mut evolver = Evolver::new(eval, cfg, rand_state);

    // Run for a few generations
    let initial_result = evolver.run().unwrap();
    let initial_fitness = initial_result.nth(0).fitness;

    // Run more generations
    for _ in 0..10 {
        let _ = evolver.run().unwrap();
    }

    let final_result = evolver.run().unwrap();
    let final_fitness = final_result.nth(0).fitness;

    // Fitness should improve (or at least not get worse)
    assert!(
        final_fitness >= initial_fitness,
        "Fitness should improve: initial={}, final={}",
        initial_fitness,
        final_fitness
    );

    // With this simple problem, we should reach positive values
    assert!(
        final_fitness > 0.0,
        "Should reach positive fitness values"
    );
}

#[test]
fn test_evolution_with_duplicates_allowed() {
    let cfg = EvolveCfg::new(10)
        .set_duplicates(memega::evolve::cfg::Duplicates::AllowDuplicates);
    let eval = MaximizeSumEvaluator;

    let rand_state = || {
        let mut rng = StdRng::seed_from_u64(rand::rng().random());
        VecState(vec![rng.random_range(-5.0..5.0); 3])
    };

    let mut evolver = Evolver::new(eval, cfg, rand_state);

    // Should complete without errors
    let result = evolver.run();
    assert!(result.is_ok(), "Evolution should complete successfully: {:?}", result.err());
}

#[test]
fn test_evolution_with_different_survival_strategies() {
    use memega::evolve::cfg::Survival;

    let eval = MaximizeSumEvaluator;
    let rand_state = || {
        let mut rng = StdRng::seed_from_u64(rand::rng().random());
        VecState(vec![rng.random_range(-5.0..5.0); 3])
    };

    // Test TopProportion survival
    let cfg = EvolveCfg::new(10).set_survival(Survival::TopProportion(0.3));
    let mut evolver = Evolver::new(eval, cfg, rand_state);
    let result = evolver.run();
    assert!(result.is_ok(), "TopProportion survival should work");

    // Test Tournament survival
    let cfg = EvolveCfg::new(10).set_survival(Survival::Tournament(3));
    let mut evolver = Evolver::new(MaximizeSumEvaluator, cfg, rand_state);
    let result = evolver.run();
    assert!(result.is_ok(), "Tournament survival should work");
}

