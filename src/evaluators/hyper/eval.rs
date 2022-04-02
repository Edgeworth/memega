use std::mem::swap;

use derive_more::Display;
use rand::Rng;

use crate::cfg::{Cfg, Crossover, Mutation};
use crate::eval::Evaluator;
use crate::evolve::evolver::CreateEvolverFn;
use crate::evolve::result::Stats;
use crate::ops::crossover::crossover_blx;
use crate::ops::distance::dist2;
use crate::ops::mutation::{mutate_normal, mutate_rate};
use crate::ops::util::rand_vec;

pub trait StatFn = Fn(Cfg) -> Option<Stats> + Send + Sync;

#[derive(Debug, Display, Clone, PartialEq, PartialOrd)]
#[display(fmt = "{:?}", cfg)]
pub struct State {
    cfg: Cfg,
    crossover: Vec<f64>, // Weights for fixed crossover.
    mutation: Vec<f64>,  // Weights for fixed mutation.
}

impl State {
    #[must_use]
    pub fn rand(pop_size: usize, num_crossover: usize, num_mutation: usize) -> State {
        let mut r = rand::thread_rng();
        let crossover = rand_vec(num_crossover, || r.gen());
        let mutation = rand_vec(num_mutation, || r.gen());
        let mut cfg = Cfg::new(pop_size);
        cfg.survival = r.gen();
        cfg.selection = r.gen();
        cfg.niching = r.gen();
        cfg.species = r.gen();
        State { cfg, crossover, mutation }
    }
}

pub struct HyperAlg {
    stat_fns: Vec<Box<dyn StatFn>>,
}

impl HyperAlg {
    #[must_use]
    pub fn new(stat_fns: Vec<Box<dyn StatFn>>) -> Self {
        Self { stat_fns }
    }
}

impl Evaluator for HyperAlg {
    type Genome = State;
    const NUM_CROSSOVER: usize = 4;
    const NUM_MUTATION: usize = 10;

    fn crossover(&self, s1: &mut State, s2: &mut State, idx: usize) {
        let mut r = rand::thread_rng();
        match idx {
            0 => {}
            1 => {
                // Uniform crossover-like operation:
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.crossover, &mut s2.cfg.crossover);
                }
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.mutation, &mut s2.cfg.mutation);
                }
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.survival, &mut s2.cfg.survival);
                }
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.selection, &mut s2.cfg.selection);
                }
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.niching, &mut s2.cfg.niching);
                }
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.species, &mut s2.cfg.species);
                }
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.stagnation, &mut s2.cfg.stagnation);
                }
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.duplicates, &mut s2.cfg.duplicates);
                }
            }
            2 => crossover_blx(&mut s1.crossover, &mut s2.crossover, 0.5),
            3 => crossover_blx(&mut s1.mutation, &mut s2.mutation, 0.5),
            _ => panic!("bug"),
        }
    }

    fn mutate(&self, s: &mut State, rate: f64, idx: usize) {
        let mut r = rand::thread_rng();
        match idx {
            0 => {
                // Mutate crossover - change type
                if r.gen_bool(rate) {
                    match &s.cfg.crossover {
                        Crossover::Fixed(v) => {
                            s.crossover = v.clone();
                            s.cfg.crossover = Crossover::Adaptive;
                        }
                        Crossover::Adaptive => {
                            s.cfg.crossover = Crossover::Fixed(s.crossover.clone());
                        }
                    }
                }
            }
            1 => {
                // Mutate crossover - modify weights
                match &mut s.cfg.crossover {
                    Crossover::Fixed(v) => {
                        mutate_rate(v, 1.0, |v| mutate_normal(v, rate).max(0.0));
                    }
                    Crossover::Adaptive => {
                        mutate_rate(&mut s.crossover, 1.0, |v| mutate_normal(v, rate).max(0.0));
                    }
                }
            }
            2 => {
                // Mutate mutation - change type
                if r.gen_bool(rate) {
                    match &s.cfg.mutation {
                        Mutation::Fixed(v) => {
                            s.mutation = v.clone();
                            s.cfg.mutation = Mutation::Adaptive;
                        }
                        Mutation::Adaptive => {
                            s.cfg.mutation = Mutation::Fixed(s.mutation.clone());
                        }
                    }
                }
            }
            3 => {
                // Mutate mutation - modify weights
                match &mut s.cfg.mutation {
                    Mutation::Fixed(v) => {
                        mutate_rate(v, 1.0, |v| mutate_normal(v, rate).max(0.0));
                    }
                    Mutation::Adaptive => {
                        mutate_rate(&mut s.mutation, 1.0, |v| mutate_normal(v, rate).max(0.0));
                    }
                }
            }
            4 => {
                if r.gen_bool(rate) {
                    s.cfg.survival = r.gen();
                }
            }
            5 => {
                if r.gen_bool(rate) {
                    s.cfg.selection = r.gen();
                }
            }
            6 => {
                if r.gen_bool(rate) {
                    s.cfg.niching = r.gen();
                }
            }
            7 => {
                if r.gen_bool(rate) {
                    s.cfg.species = r.gen();
                }
            }
            8 => {
                if r.gen_bool(rate) {
                    s.cfg.stagnation = r.gen();
                }
            }
            9 => {
                if r.gen_bool(rate) {
                    s.cfg.duplicates = r.gen();
                }
            }
            _ => panic!("bug"),
        }
    }

    fn fitness(&self, s: &State, _gen: usize) -> f64 {
        const SAMPLES: usize = 30;
        let mut score = 0.0;
        for _ in 0..SAMPLES {
            for f in &self.stat_fns {
                if let Some(r) = f(s.cfg.clone()) {
                    // TODO: Need multi-objective GA here. Or at least configure
                    // what to optimise.
                    score += r.best_fitness;
                }
            }
        }
        score / SAMPLES as f64
    }

    fn distance(&self, s1: &State, s2: &State) -> f64 {
        let mut dist = 0.0;

        let s1_cross = if let Crossover::Fixed(v) = &s1.cfg.crossover { v } else { &s1.crossover };
        let s2_cross = if let Crossover::Fixed(v) = &s2.cfg.crossover { v } else { &s2.crossover };
        dist += dist2(s1_cross, s2_cross);

        let s1_mutation = if let Mutation::Fixed(v) = &s1.cfg.mutation { v } else { &s1.mutation };
        let s2_mutation = if let Mutation::Fixed(v) = &s2.cfg.mutation { v } else { &s2.mutation };
        dist += dist2(s1_mutation, s2_mutation);

        dist
    }
}
