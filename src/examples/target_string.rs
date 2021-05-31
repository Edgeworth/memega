use rand::Rng;

use crate::cfg::Cfg;
use crate::distributions::PrintableAscii;
use crate::eval::Evaluator;
use crate::ops::crossover::crossover_kpx;
use crate::ops::distance::count_different;
use crate::ops::mutation::mutate_rate;
use crate::ops::util::{rand_vec, str_to_vec};
use crate::runner::Runner;

type State = Vec<char>;

#[derive(Debug, Clone)]
pub struct TargetString {
    target: State,
}

impl TargetString {
    fn new(target: &str) -> Self {
        Self { target: str_to_vec(target) }
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

pub fn target_string_runner(cfg: Cfg) -> Runner<TargetString> {
    const TARGET: &str = "Hello world!";
    Runner::new(TargetString::new(TARGET), cfg, move || {
        let mut r = rand::thread_rng();
        rand_vec(TARGET.len(), || r.sample::<char, _>(PrintableAscii))
    })
}
