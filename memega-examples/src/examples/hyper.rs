use std::time::Duration;

use memega::cfg::Cfg;
use memega::evaluators::hyper::builder::HyperBuilder;
use memega::evaluators::hyper::eval::HyperAlg;
use memega::evolve::evolver::Evolver;

use crate::examples::ackley::ackley_evolver;
use crate::examples::griewank::griewank_evolver;
use crate::examples::knapsack::knapsack_evolver;
use crate::examples::rastrigin::rastrigin_evolver;
use crate::examples::target_string::target_string_evolver;

#[must_use]
pub fn hyper_evolver(pop_size: usize, sample_dur: Duration, cfg: Cfg) -> Evolver<HyperAlg> {
    let mut builder = HyperBuilder::new(pop_size, sample_dur);
    builder.add(1.0, &|cfg| rastrigin_evolver(2, cfg));
    builder.add(1.0, &|cfg| griewank_evolver(2, cfg));
    builder.add(1.0, &|cfg| ackley_evolver(2, cfg));
    builder.add(1000.0, &knapsack_evolver);
    builder.add(12.0, &target_string_evolver);
    builder.build(cfg)
}
