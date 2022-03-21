use std::fmt;

use rand::prelude::SliceRandom;
use rand::Rng;
use strum::IntoEnumIterator;

use crate::cfg::Cfg;
use crate::eval::{Evaluator, FitnessFn};
use crate::lgp::disasm::lgp_disasm;
use crate::lgp::op::{Op, Opcode};
use crate::ops::crossover::crossover_kpx;
use crate::ops::distance::dist_fn;
use crate::ops::mutation::{mutate_insert, mutate_reset, mutate_scramble, mutate_swap};
use crate::ops::util::rand_vec;
use crate::run::runner::Runner;

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
    #[must_use]
    pub fn new(ops: Vec<Op>, num_reg: usize) -> Self {
        Self { ops, num_reg }
    }
}

#[derive(Clone)]
pub struct LgpGenomeConfig {
    max_reg: usize,
    max_code: usize,
    opcodes: Vec<Opcode>,
}

impl LgpGenomeConfig {
    #[must_use]
    pub fn new(max_reg: usize, max_code: usize) -> Self {
        Self { max_reg, max_code, opcodes: Opcode::iter().collect() }
    }

    #[must_use]
    pub fn with_opcodes(mut self, opcodes: &[Opcode]) -> Self {
        self.opcodes = opcodes.to_vec();
        self
    }
}

pub struct LgpGenome {
    cfg: LgpGenomeConfig,
}

impl LgpGenome {
    #[must_use]
    pub fn new(cfg: LgpGenomeConfig) -> Self {
        Self { cfg }
    }
}

impl Evaluator for LgpGenome {
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
        let opcode = *self.cfg.opcodes.choose(&mut r).unwrap();
        let op = Op::rand(opcode, s.num_reg, code_size);
        match idx {
            0 => mutate_swap(&mut s.ops),
            1 => mutate_insert(&mut s.ops),
            2 => mutate_reset(&mut s.ops, op),
            3 => mutate_scramble(&mut s.ops),
            4 => {
                // Add new random instruction.
                if code_size < self.cfg.max_code {
                    s.ops.insert(r.gen_range(0..code_size), op);
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

    fn fitness(&self, _: &State, _gen: usize) -> f64 {
        unimplemented!()
    }

    fn distance(&self, s1: &State, s2: &State) -> f64 {
        dist_fn(&s1.ops, &s2.ops, 1.0, Op::dist)
    }
}

pub struct LgpGenomeFn<F: FitnessFn<State>> {
    genome: LgpGenome,
    f: F,
}

impl<F: FitnessFn<State>> LgpGenomeFn<F> {
    pub fn new(cfg: LgpGenomeConfig, f: F) -> Self {
        Self { genome: LgpGenome::new(cfg), f }
    }
}

impl<F: FitnessFn<State>> Evaluator for LgpGenomeFn<F> {
    type Genome = <LgpGenome as Evaluator>::Genome;
    const NUM_CROSSOVER: usize = LgpGenome::NUM_CROSSOVER;
    const NUM_MUTATION: usize = LgpGenome::NUM_MUTATION;

    fn crossover(&self, s1: &mut State, s2: &mut State, idx: usize) {
        self.genome.crossover(s1, s2, idx);
    }

    fn mutate(&self, s: &mut State, rate: f64, idx: usize) {
        self.genome.mutate(s, rate, idx);
    }

    fn fitness(&self, s: &State, gen: usize) -> f64 {
        (self.f)(s, gen)
    }

    fn distance(&self, s1: &State, s2: &State) -> f64 {
        self.genome.distance(s1, s2)
    }
}

fn rand_op(cfg: &LgpGenomeConfig) -> Op {
    let mut r = rand::thread_rng();
    Op::rand(*cfg.opcodes.choose(&mut r).unwrap(), cfg.max_reg, cfg.max_code)
}

pub fn lgp_runner<E: Evaluator<Genome = State>, F: FnOnce(LgpGenome) -> E>(
    lgpcfg: LgpGenomeConfig,
    cfg: Cfg,
    f: F,
) -> Runner<E> {
    Runner::new(f(LgpGenome::new(lgpcfg.clone())), cfg, move || {
        State::new(rand_vec(lgpcfg.max_code, || rand_op(&lgpcfg.clone())), lgpcfg.max_reg)
    })
}

pub fn lgp_runner_fn<F: FitnessFn<State>>(
    lgpcfg: LgpGenomeConfig,
    cfg: Cfg,
    f: F,
) -> Runner<LgpGenomeFn<F>> {
    Runner::new(LgpGenomeFn::new(lgpcfg.clone(), f), cfg, move || {
        State::new(rand_vec(lgpcfg.max_code, || rand_op(&lgpcfg.clone())), lgpcfg.max_reg)
    })
}
