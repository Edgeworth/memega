use derive_more::{Deref, DerefMut, Display};
use eyre::Result;
use memega::eval::{Evaluator, FitnessFn};
use memega::evolve::cfg::EvolveCfg;
use memega::evolve::evolver::Evolver;
use memega::ops::crossover::crossover_arith;
use memega::ops::distance::dist2;
use memega::ops::mutation::{mutate_normal, mutate_rate, mutate_uniform};
use memega::ops::util::rand_vec;

#[derive(Debug, Display, Deref, DerefMut, Clone, PartialEq, PartialOrd)]
#[display(fmt = "{:?}", _0)]
pub struct FuncState(pub Vec<f64>);

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct FuncEvaluator<F: FitnessFn<FuncState>> {
    dim: usize,
    st: f64,
    en: f64,
    f: F,
}

impl<F: FitnessFn<FuncState>> FuncEvaluator<F> {
    fn new(dim: usize, st: f64, en: f64, f: F) -> Self {
        Self { dim, st, en, f }
    }
}

impl<F: FitnessFn<FuncState>> Evaluator for FuncEvaluator<F> {
    type State = FuncState;

    fn crossover(&self, s1: &mut Self::State, s2: &mut Self::State, idx: usize) {
        match idx {
            0 => {}
            1 => crossover_arith(s1, s2),
            _ => panic!("bug"),
        };
    }

    fn mutate(&self, s: &mut Self::State, rate: f64, idx: usize) {
        match idx {
            0 => mutate_rate(s, 1.0, |v| mutate_normal(v, rate).clamp(self.st, self.en)),
            _ => panic!("bug"),
        };
    }

    fn fitness(&self, s: &Self::State, data: &Self::Data) -> Result<f64> {
        (self.f)(s, data)
    }

    fn distance(&self, s1: &Self::State, s2: &Self::State) -> Result<f64> {
        Ok(dist2(s1, s2))
    }
}

pub fn func_evolver<F: FitnessFn<FuncState>>(
    dim: usize,
    st: f64,
    en: f64,
    f: F,
    cfg: EvolveCfg,
) -> Evolver<impl Evaluator<Data = ()>> {
    Evolver::new(FuncEvaluator::new(dim, st, en, f), cfg, move || {
        FuncState(rand_vec(dim, || mutate_uniform(st, en)))
    })
}
