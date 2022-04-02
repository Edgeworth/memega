use derive_more::{Deref, DerefMut, Display};
use memega::cfg::Cfg;
use memega::eval::Evaluator;
use memega::evolve::evolver::Evolver;
use memega::ops::crossover::crossover_kpx;
use memega::ops::distance::count_different;
use memega::ops::mutation::mutate_rate;
use memega::ops::util::{rand_vec, str_to_vec};
use memega::util::distributions::PrintableAscii;
use rand::Rng;

#[derive(Debug, Display, Deref, DerefMut, Clone, PartialEq, PartialOrd)]
#[display(fmt = "{}", "self.0.iter().collect::<String>()")]
pub struct State(pub Vec<char>);

#[derive(Debug, Clone)]
pub struct TargetString {
    target: State,
}

impl TargetString {
    fn new(target: &str) -> Self {
        Self { target: State(str_to_vec(target)) }
    }
}

impl Evaluator for TargetString {
    type Genome = State;

    fn crossover(&self, s1: &mut State, s2: &mut State, idx: usize) {
        match idx {
            0 => {}
            1 => crossover_kpx(s1, s2, 2),
            _ => panic!("bug"),
        };
    }

    fn mutate(&self, s: &mut State, rate: f64, idx: usize) {
        let mut r = rand::thread_rng();
        match idx {
            0 => mutate_rate(s, rate, |_| r.sample(PrintableAscii)),
            _ => panic!("bug"),
        };
    }

    fn fitness(&self, s: &State, _gen: usize) -> f64 {
        (self.target.len() - count_different(s, &self.target)) as f64 + 1.0
    }

    fn distance(&self, s1: &State, s2: &State) -> f64 {
        count_different(s1, s2) as f64
    }
}

#[must_use]
pub fn target_string_evolver(cfg: Cfg) -> Evolver<TargetString> {
    const TARGET: &str = "Hello world!";
    Evolver::new(TargetString::new(TARGET), cfg, move || {
        let mut r = rand::thread_rng();
        State(rand_vec(TARGET.len(), || r.sample::<char, _>(PrintableAscii)))
    })
}
