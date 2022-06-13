use eyre::Result;

use crate::eval::{Data, Evaluator, FitnessFn};
use crate::evaluators::lgp::cfg::LgpEvaluatorCfg;
use crate::evaluators::lgp::eval::{LgpEvaluator, LgpState};
use crate::evolve::cfg::EvolveCfg;
use crate::evolve::evolver::Evolver;
use crate::ops::mutation::mutate_normal;
use crate::ops::util::rand_vec;

#[must_use]
pub struct LgpFitnessFnEvaluator<D: Data, F: FitnessFn<LgpState, D>> {
    evaluator: LgpEvaluator<D>,
    f: F,
}

impl<D: Data, F: FitnessFn<LgpState, D>> LgpFitnessFnEvaluator<D, F> {
    pub fn new(evaluator: LgpEvaluator<D>, f: F) -> Self {
        Self { evaluator, f }
    }
}

impl<D: Data, F: FitnessFn<LgpState, D>> Evaluator for LgpFitnessFnEvaluator<D, F> {
    type State = <LgpEvaluator<D> as Evaluator>::State;
    type Data = <LgpEvaluator<D> as Evaluator>::Data;
    const NUM_CROSSOVER: usize = LgpEvaluator::<D>::NUM_CROSSOVER;
    const NUM_MUTATION: usize = LgpEvaluator::<D>::NUM_MUTATION;

    fn crossover(&self, s1: &mut Self::State, s2: &mut Self::State, idx: usize) {
        self.evaluator.crossover(s1, s2, idx);
    }

    fn mutate(&self, s: &mut Self::State, rate: f64, idx: usize) {
        self.evaluator.mutate(s, rate, idx);
    }

    fn fitness(&self, s: &Self::State, data: &Self::Data) -> Result<f64> {
        (self.f)(s, data)
    }

    fn distance(&self, s1: &Self::State, s2: &Self::State) -> Result<f64> {
        self.evaluator.distance(s1, s2)
    }
}

pub fn lgp_create_evolver<
    D: Data,
    E: Evaluator<State = LgpState, Data = D>,
    F: FnOnce(LgpEvaluator<D>) -> E,
>(
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

pub fn lgp_fitness_evolver<D: Data, F: FitnessFn<LgpState, D>>(
    lgpcfg: LgpEvaluatorCfg,
    cfg: EvolveCfg,
    f: F,
) -> Evolver<impl Evaluator<Data = D>> {
    lgp_create_evolver(lgpcfg, cfg, |evaluator| LgpFitnessFnEvaluator::new(evaluator, f))
}
