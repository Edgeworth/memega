use rand::Rng;

use crate::cfg::Cfg;
use crate::eval::Evaluator;
use crate::lgp::exec::LgpExec;
use crate::ops::crossover::{crossover_cycle, crossover_kpx, crossover_order, crossover_pmx};
use crate::ops::distance::dist1;
use crate::ops::mutation::{mutate_gen, mutate_insert, mutate_reset, mutate_swap};
use crate::ops::util::rand_vec;
use crate::runner::Runner;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct State {
    ops: Vec<u8>, // Contains program code for linear genetic programming.
}

impl State {
    pub fn new(ops: Vec<u8>) -> Self {
        Self { ops }
    }
}

pub struct LgpGenome {
    initial_reg: Vec<f64>,
}

impl LgpGenome {
    pub fn new(initial_reg: &[f64]) -> Self {
        Self { initial_reg: initial_reg.to_vec() }
    }
}

impl Evaluator for LgpGenome {
    type Genome = State;
    const NUM_CROSSOVER: usize = 5;
    const NUM_MUTATION: usize = 4;

    fn crossover(&self, s1: &mut State, s2: &mut State, idx: usize) {
        match idx {
            0 => {} // Do nothing.
            1 => {
                crossover_pmx(&mut s1.ops, &mut s2.ops);
            }
            2 => {
                crossover_order(&mut s1.ops, &mut s2.ops);
            }
            3 => {
                crossover_cycle(&mut s1.ops, &mut s2.ops);
            }
            4 => {
                crossover_kpx(&mut s1.ops, &mut s2.ops, 2);
            }
            _ => panic!("unknown crossover strategy"),
        };
    }

    fn mutate(&self, s: &mut State, rate: f64, idx: usize) {
        let mut r = rand::thread_rng();
        let mutate = r.gen::<f64>() < rate;
        match idx {
            0 => {
                if mutate {
                    mutate_swap(&mut s.ops);
                }
            }
            1 => {
                if mutate {
                    mutate_insert(&mut s.ops);
                }
            }
            2 => {
                if mutate {
                    mutate_reset(&mut s.ops, mutate_gen());
                }
            }
            _ => panic!("unknown mutation strategy"),
        }
    }

    fn fitness(&self, s: &State) -> f64 {
        let mut r = rand::thread_rng();
        let reg = rand_vec(8, move || r.gen::<f64>());
        let mut exec = LgpExec::new(&reg, &s.ops, 200);
        exec.run();
        // Test: Just compute sum of 1/ri for i in 0 to 8
        let ans: f64 = reg.iter().map(|v| 1.0 / v).sum();
        (ans - exec.reg(8)).abs()
    }

    fn distance(&self, s1: &State, s2: &State) -> f64 {
        // TODO: Treat different instructions differently to different constants?
        dist1(&s1.ops, &s2.ops) as f64
    }
}

pub fn lgp_runner(code_len: usize, cfg: Cfg) -> Runner<LgpGenome> {
    Runner::new(LgpGenome::new(&[]), cfg, move || State::new(rand_vec(code_len, mutate_gen::<u8>)))
}
