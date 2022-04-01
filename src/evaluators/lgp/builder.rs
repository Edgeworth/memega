use crate::cfg::Cfg;
use crate::eval::{Evaluator, FitnessFn};
use crate::evaluators::lgp::cfg::LgpCfg;
use crate::evaluators::lgp::eval::{LgpGenome, State};
use crate::ops::util::rand_vec;
use crate::run::runner::Runner;

pub struct LgpGenomeFn<F: FitnessFn<State>> {
    genome: LgpGenome,
    f: F,
}

impl<F: FitnessFn<State>> LgpGenomeFn<F> {
    pub fn new(cfg: LgpCfg, f: F) -> Self {
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
    lgpcfg: LgpCfg,
    cfg: Cfg,
    f: F,
) -> Runner<E> {
    Runner::new(f(LgpGenome::new(lgpcfg)), cfg, move || {
        State::new(rand_vec(lgpcfg.max_code(), || lgpcfg.rand_op(None)), lgpcfg)
    })
}

pub fn lgp_runner_fn<F: FitnessFn<State>>(
    lgpcfg: LgpCfg,
    cfg: Cfg,
    f: F,
) -> Runner<LgpGenomeFn<F>> {
    Runner::new(LgpGenomeFn::new(lgpcfg, f), cfg, move || {
        State::new(rand_vec(lgpcfg.max_code(), || lgpcfg.rand_op(None)), lgpcfg)
    })
}
