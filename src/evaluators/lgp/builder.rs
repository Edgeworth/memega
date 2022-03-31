use rand::prelude::SliceRandom;
use strum::IntoEnumIterator;

use crate::cfg::Cfg;
use crate::eval::{Evaluator, FitnessFn};
use crate::evaluators::lgp::eval::{LgpGenome, State};
use crate::evaluators::lgp::vm::op::{Op, Opcode};
use crate::ops::util::rand_vec;
use crate::run::runner::Runner;

fn rand_op(cfg: &LgpGenomeConfig) -> Op {
    let mut r = rand::thread_rng();
    Op::rand(*cfg.opcodes.choose(&mut r).unwrap(), cfg.max_reg, cfg.max_code)
}

#[derive(Clone)]
pub struct LgpGenomeConfig {
    max_reg: usize,
    max_code: usize,
    opcodes: Vec<Opcode>,
}

impl LgpGenomeConfig {
    #[must_use]
    pub fn new() -> Self {
        Self { max_reg: 4, max_code: 4, opcodes: Opcode::iter().collect() }
    }

    #[must_use]
    pub fn set_max_reg(mut self, max_reg: usize) -> Self {
        self.max_reg = max_reg;
        self
    }

    #[must_use]
    pub fn set_max_code(mut self, max_code: usize) -> Self {
        self.max_code = max_code;
        self
    }

    #[must_use]
    pub fn set_opcodes(mut self, opcodes: &[Opcode]) -> Self {
        self.opcodes = opcodes.to_vec();
        self
    }

    #[must_use]
    pub fn max_reg(&self) -> usize {
        self.max_reg
    }

    #[must_use]
    pub fn max_code(&self) -> usize {
        self.max_code
    }

    #[must_use]
    pub fn opcodes(&self) -> &[Opcode] {
        self.opcodes.as_ref()
    }
}

impl Default for LgpGenomeConfig {
    fn default() -> Self {
        Self::new()
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
