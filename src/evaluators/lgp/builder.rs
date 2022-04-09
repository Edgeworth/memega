use crate::eval::{Evaluator, FitnessFn};
use crate::evaluators::lgp::cfg::LgpEvaluatorCfg;
use crate::evaluators::lgp::eval::{LgpEvaluator, LgpState};
use crate::evolve::cfg::EvolveCfg;
use crate::evolve::evolver::Evolver;
use crate::ops::mutation::mutate_normal;
use crate::ops::util::rand_vec;

pub struct LgpFitnessFnEvaluator<F: FitnessFn<LgpState>> {
    evaluator: LgpEvaluator,
    f: F,
}

impl<F: FitnessFn<LgpState>> LgpFitnessFnEvaluator<F> {
    pub fn new(evaluator: LgpEvaluator, f: F) -> Self {
        Self { evaluator, f }
    }
}

impl<F: FitnessFn<LgpState>> Evaluator for LgpFitnessFnEvaluator<F> {
    type State = <LgpEvaluator as Evaluator>::State;
    const NUM_CROSSOVER: usize = LgpEvaluator::NUM_CROSSOVER;
    const NUM_MUTATION: usize = LgpEvaluator::NUM_MUTATION;

    fn crossover(&self, s1: &mut LgpState, s2: &mut LgpState, idx: usize) {
        self.evaluator.crossover(s1, s2, idx);
    }

    fn mutate(&self, s: &mut LgpState, rate: f64, idx: usize) {
        self.evaluator.mutate(s, rate, idx);
    }

    fn fitness(&self, s: &LgpState, gen: usize) -> f64 {
        (self.f)(s, gen)
    }

    fn distance(&self, s1: &LgpState, s2: &LgpState) -> f64 {
        self.evaluator.distance(s1, s2)
    }
}

pub fn lgp_evolver<E: Evaluator<State = LgpState>, F: FnOnce(LgpEvaluator) -> E>(
    lgpcfg: LgpEvaluatorCfg,
    cfg: EvolveCfg,
    f: F,
) -> Evolver<E> {
    const INITIAL_LENGTH_MEAN: f64 = 10.0;
    const INITIAL_LENGTH_STD: f64 = 2.0;

    Evolver::new(f(LgpEvaluator::new(lgpcfg.clone())), cfg, move || {
        // Better to start with small-ish programs, even if the max code
        // length is high.
        let length = mutate_normal(INITIAL_LENGTH_MEAN, INITIAL_LENGTH_STD).round() as usize;
        let length = length.clamp(1, lgpcfg.max_code());
        let ops = rand_vec(length, || lgpcfg.rand_op());
        LgpState::new(ops, lgpcfg.num_reg(), lgpcfg.num_const(), lgpcfg.output_regs())
    })
}

pub fn lgp_fitness_evolver<F: FitnessFn<LgpState>>(
    lgpcfg: LgpEvaluatorCfg,
    cfg: EvolveCfg,
    f: F,
) -> Evolver<impl Evaluator> {
    lgp_evolver(lgpcfg, cfg, |evaluator| LgpFitnessFnEvaluator::new(evaluator, f))
}
