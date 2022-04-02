use derive_more::{Deref, DerefMut, Display};
use memega::cfg::Cfg;
use memega::eval::{Evaluator, FitnessFn};
use memega::evolve::evolver::Evolver;
use memega::ops::crossover::crossover_arith;
use memega::ops::distance::dist2;
use memega::ops::mutation::{mutate_normal, mutate_rate, mutate_uniform};
use memega::ops::util::rand_vec;

#[derive(Debug, Display, Deref, DerefMut, Clone, PartialEq, PartialOrd)]
#[display(fmt = "{:?}", _0)]
pub struct State(pub Vec<f64>);

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct FuncEvaluator<F: FitnessFn<State>> {
    dim: usize,
    st: f64,
    en: f64,
    f: F,
}

impl<F: FitnessFn<State>> FuncEvaluator<F> {
    fn new(dim: usize, st: f64, en: f64, f: F) -> Self {
        Self { dim, st, en, f }
    }
}

impl<F: FitnessFn<State>> Evaluator for FuncEvaluator<F> {
    type Genome = State;

    fn crossover(&self, s1: &mut State, s2: &mut State, idx: usize) {
        match idx {
            0 => {}
            1 => crossover_arith(s1, s2),
            _ => panic!("bug"),
        };
    }

    fn mutate(&self, s: &mut State, rate: f64, idx: usize) {
        match idx {
            0 => mutate_rate(s, 1.0, |v| mutate_normal(v, rate).clamp(self.st, self.en)),
            _ => panic!("bug"),
        };
    }

    fn fitness(&self, s: &State, gen: usize) -> f64 {
        (self.f)(s, gen)
    }

    fn distance(&self, s1: &State, s2: &State) -> f64 {
        dist2(s1, s2)
    }
}

pub fn func_evolver<F: FitnessFn<State>>(
    dim: usize,
    st: f64,
    en: f64,
    f: F,
    cfg: Cfg,
) -> Evolver<impl Evaluator> {
    Evolver::new(FuncEvaluator::new(dim, st, en, f), cfg, move || {
        State(rand_vec(dim, || mutate_uniform(st, en)))
    })
}
