use derive_more::{Deref, DerefMut, Display};
use memega::cfg::Cfg;
use memega::eval::Evaluator;
use memega::evolve::evolver::Evolver;
use memega::ops::crossover::crossover_kpx;
use memega::ops::distance::count_different;
use memega::ops::mutation::mutate_rate;
use memega::ops::util::rand_vec;
use rand::Rng;

#[derive(Debug, Display, Deref, DerefMut, Clone, PartialEq, PartialOrd)]
#[display(fmt = "{:?}", _0)]
pub struct State(pub Vec<bool>);

#[derive(Debug, Clone)]
pub struct Knapsack {
    max_w: f64,
    items: Vec<(f64, f64)>, // weight and value
}

impl Knapsack {
    fn new(max_w: f64, items: Vec<(f64, f64)>) -> Self {
        Self { max_w, items }
    }
}

impl Evaluator for Knapsack {
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
            0 => mutate_rate(s, rate, |_| r.gen::<bool>()),
            _ => panic!("bug"),
        };
    }

    fn fitness(&self, s: &State, _gen: usize) -> f64 {
        let mut cur_w = 0.0;
        let mut cur_v = 0.0;
        for (i, &kept) in s.iter().enumerate() {
            let (w, v) = self.items[i];
            if kept && cur_w + w <= self.max_w {
                cur_w += w;
                cur_v += v;
            }
        }
        cur_v
    }

    fn distance(&self, s1: &State, s2: &State) -> f64 {
        count_different(s1, s2) as f64
    }
}

#[must_use]
pub fn knapsack_evolver(cfg: Cfg) -> Evolver<Knapsack> {
    const NUM_ITEMS: usize = 100;
    const MAX_W: f64 = 100.0;

    let mut r = rand::thread_rng();
    let items = rand_vec(NUM_ITEMS, || {
        let w = r.gen_range(0.0..MAX_W);
        let v = r.gen_range(0.1..10.0) * w;
        (w, v)
    });
    Evolver::new(Knapsack::new(MAX_W, items), cfg, move || {
        let mut r = rand::thread_rng();
        State(rand_vec(NUM_ITEMS, || r.gen::<bool>()))
    })
}
