use derive_more::{Deref, DerefMut, Display};
use eyre::Result;
use memega::eval::Evaluator;
use memega::evolve::cfg::EvolveCfg;
use memega::evolve::evolver::Evolver;
use memega::ops::crossover::crossover_kpx;
use memega::ops::distance::count_different;
use memega::ops::mutation::mutate_rate;
use memega::ops::util::{rand_vec, str_to_vec};
use memega::util::distributions::PrintableAscii;
use rand::Rng;

#[must_use]
#[derive(Debug, Display, Deref, DerefMut, Clone, PartialEq, Eq, PartialOrd)]
#[display("{}", self.0.iter().collect::<String>())]
pub struct TargetStringState(pub Vec<char>);

#[must_use]
#[derive(Debug, Clone)]
pub struct TargetStringEvaluator {
    target: TargetStringState,
}

impl TargetStringEvaluator {
    fn new(target: &str) -> Self {
        Self { target: TargetStringState(str_to_vec(target)) }
    }
}

impl Evaluator for TargetStringEvaluator {
    type State = TargetStringState;

    fn crossover(&self, s1: &mut Self::State, s2: &mut Self::State, idx: usize) {
        match idx {
            0 => {}
            1 => crossover_kpx(s1, s2, 2),
            _ => panic!("bug"),
        }
    }

    fn mutate(&self, s: &mut Self::State, rate: f64, idx: usize) {
        let mut r = rand::rng();
        match idx {
            0 => mutate_rate(s, rate, |_| r.sample(PrintableAscii)),
            _ => panic!("bug"),
        }
    }

    fn fitness(&self, s: &Self::State, _data: &Self::Data) -> Result<f64> {
        Ok((self.target.len() - count_different(s, &self.target)) as f64 + 1.0)
    }

    fn distance(&self, s1: &Self::State, s2: &Self::State) -> Result<f64> {
        Ok(count_different(s1, s2) as f64)
    }
}

pub fn target_string_evolver(cfg: EvolveCfg) -> Evolver<TargetStringEvaluator> {
    const TARGET: &str = "Hello world!";
    Evolver::new(TargetStringEvaluator::new(TARGET), cfg, move || {
        let mut r = rand::rng();
        TargetStringState(rand_vec(TARGET.len(), || r.sample::<char, _>(PrintableAscii)))
    })
}
