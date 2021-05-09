use std::fmt;

use rand::Rng;

use crate::cfg::Cfg;
use crate::eval::Evaluator;
use crate::lgp::disasm::LgpDisasm;
use crate::lgp::exec::LgpExec;
use crate::ops::crossover::{crossover_cycle, crossover_kpx, crossover_order, crossover_pmx};
use crate::ops::distance::dist1;
use crate::ops::mutation::{mutate_gen, mutate_insert, mutate_reset, mutate_swap};
use crate::ops::util::rand_vec;
use crate::runner::Runner;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct State {
    ops: Vec<u8>, // Contains program code for linear genetic programming.
    num_reg: usize,
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut ds = LgpDisasm::new(&self.ops, self.num_reg);
        f.write_str(&ds.disasm())
    }
}

impl State {
    pub fn new(ops: Vec<u8>, num_reg: usize) -> Self {
        Self { ops, num_reg }
    }
}

pub struct LgpGenome {
    max_code: usize,
}

impl LgpGenome {
    pub fn new(max_code: usize) -> Self {
        Self { max_code }
    }
}

impl Evaluator for LgpGenome {
    type Genome = State;
    const NUM_CROSSOVER: usize = 2;
    const NUM_MUTATION: usize = 5;

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
            3 => {
                if mutate && s.ops.len() < self.max_code {
                    s.ops.push(mutate_gen());
                }
            }
            4 => {
                if mutate {
                    s.ops.pop();
                }
            }
            _ => panic!("unknown mutation strategy"),
        }
    }

    fn fitness(&self, s: &State) -> f64 {
        let mut fitness = 0.0;
        for _ in 0..1000 {
            let mut r = rand::thread_rng();
            let mut reg = vec![0.0, 0.0, 0.0, 0.0, 0.0]; // Space for work and answer.
            let vals = rand_vec(5, move || r.gen_range(0.05..5.0));
            reg.extend(&vals);
            if reg.len() != s.num_reg {
                panic!("ASDF");
            }
            let mut exec = LgpExec::new(&reg, &s.ops, 200);
            exec.run();
            let ans: f64 = vals.iter().sum::<f64>();
            fitness += 1.0 / (1.0 + (ans - exec.reg(0)).abs())
        }
        fitness + 1.0 / (1.0 + s.ops.len() as f64)
    }

    fn distance(&self, s1: &State, s2: &State) -> f64 {
        // TODO: Treat different instructions differently to different
        // constants?
        let s1ops: Vec<f64> = s1.ops.iter().map(|&v| v as f64).collect();
        let s2ops: Vec<f64> = s2.ops.iter().map(|&v| v as f64).collect();
        dist1(&s1ops, &s2ops)
    }
}

pub fn lgp_runner(max_code: usize, cfg: Cfg) -> Runner<LgpGenome> {
    // TODO: num reg here.
    Runner::new(LgpGenome::new(max_code), cfg, move || {
        State::new(rand_vec(max_code, mutate_gen::<u8>), 10)
    })
}
