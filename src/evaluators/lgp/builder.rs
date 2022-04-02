use crate::cfg::Cfg;
use crate::eval::{Evaluator, FitnessFn};
use crate::evaluators::lgp::cfg::LgpCfg;
use crate::evaluators::lgp::eval::{LgpGenome, State};
use crate::evolve::evolver::Evolver;
use crate::ops::util::rand_vec;

pub struct LgpGenomeFitnessFn<F: FitnessFn<State>> {
    genome: LgpGenome,
    f: F,
}

impl<F: FitnessFn<State>> LgpGenomeFitnessFn<F> {
    pub fn new(genome: LgpGenome, f: F) -> Self {
        Self { genome, f }
    }
}

impl<F: FitnessFn<State>> Evaluator for LgpGenomeFitnessFn<F> {
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

pub fn lgp_evolver<E: Evaluator<Genome = State>, F: FnOnce(LgpGenome) -> E>(
    lgpcfg: LgpCfg,
    cfg: Cfg,
    f: F,
) -> Evolver<E> {
    Evolver::new(f(LgpGenome::new(lgpcfg)), cfg, move || {
        State::new(rand_vec(lgpcfg.max_code(), || lgpcfg.rand_op()), lgpcfg)
    })
}

pub fn lgp_fitness_evolver<F: FitnessFn<State>>(
    lgpcfg: LgpCfg,
    cfg: Cfg,
    f: F,
) -> Evolver<impl Evaluator> {
    lgp_evolver(lgpcfg, cfg, |genome| LgpGenomeFitnessFn::new(genome, f))
}
