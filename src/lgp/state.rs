use std::fmt;

use rand::prelude::SliceRandom;
use rand::Rng;

use crate::cfg::Cfg;
use crate::eval::{Evaluator, FitnessFn};
use crate::lgp::disasm::lgp_disasm;
use crate::lgp::op::Op;
use crate::ops::crossover::crossover_kpx;
use crate::ops::distance::dist_fn;
use crate::ops::mutation::{mutate_insert, mutate_reset, mutate_scramble, mutate_swap};
use crate::ops::util::rand_vec;
use crate::runner::Runner;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct State {
    pub ops: Vec<Op>, // Contains program code for linear genetic programming.
    pub num_reg: usize,
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&lgp_disasm(&self.ops))
    }
}

impl State {
    pub fn new(ops: Vec<Op>, num_reg: usize) -> Self {
        Self { ops, num_reg }
    }
}

pub struct LgpGenome<F: FitnessFn<State>> {
    max_code: usize,
    f: F,
}

impl<F: FitnessFn<State>> LgpGenome<F> {
    pub fn new(max_code: usize, f: F) -> Self {
        Self { max_code, f }
    }
}

impl<F: FitnessFn<State>> Evaluator for LgpGenome<F> {
    type Genome = State;
    const NUM_CROSSOVER: usize = 2;
    const NUM_MUTATION: usize = 7;

    fn crossover(&self, s1: &mut State, s2: &mut State, idx: usize) {
        match idx {
            0 => {} // Do nothing.
            1 => {
                crossover_kpx(&mut s1.ops, &mut s2.ops, 2);
            }
            _ => panic!("unknown crossover strategy"),
        };
    }

    fn mutate(&self, s: &mut State, rate: f64, idx: usize) {
        let mut r = rand::thread_rng();
        if r.gen::<f64>() > rate {
            return;
        }
        let code_size = s.ops.len();
        match idx {
            0 => mutate_swap(&mut s.ops),
            1 => mutate_insert(&mut s.ops),
            2 => mutate_reset(&mut s.ops, Op::rand(s.num_reg, code_size)),
            3 => mutate_scramble(&mut s.ops),
            4 => {
                // Add new random instruction.
                if code_size < self.max_code {
                    s.ops.insert(r.gen_range(0..code_size), Op::rand(s.num_reg, code_size));
                }
            }
            5 => {
                // Remove random instruction.
                if code_size > 1 {
                    s.ops.remove(r.gen_range(0..code_size));
                }
            }
            6 => {
                // Micro-mutation
                s.ops.choose_mut(&mut r).unwrap().mutate(s.num_reg, code_size);
            }
            _ => panic!("unknown mutation strategy"),
        }
    }

    fn fitness(&self, s: &State) -> f64 {
        (self.f)(s)
    }

    fn distance(&self, s1: &State, s2: &State) -> f64 {
        dist_fn(&s1.ops, &s2.ops, 1.0, |a, b| Op::dist(a, b))
    }
}

pub fn lgp_runner<F: FitnessFn<State>>(
    num_reg: usize,
    max_code: usize,
    cfg: Cfg,
    f: F,
) -> Runner<LgpGenome<F>> {
    Runner::new(LgpGenome::new(max_code, f), cfg, move || {
        State::new(rand_vec(max_code, || Op::rand(num_reg, max_code)), num_reg)
    })
}
