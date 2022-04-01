use std::time::Duration;

use memega::evaluators::hyper::builder::HyperBuilder;
use memega::evaluators::hyper::eval::HyperAlg;
use memega::run::runner::Runner;

use crate::examples::ackley::ackley_runner;
use crate::examples::griewank::griewank_runner;
use crate::examples::knapsack::knapsack_runner;
use crate::examples::rastrigin::rastrigin_runner;
use crate::examples::target_string::target_string_runner;

#[must_use]
pub fn hyper_runner(pop_size: usize, sample_dur: Duration) -> Runner<HyperAlg> {
    let mut builder = HyperBuilder::new(pop_size, sample_dur);
    builder.add(1.0, &|cfg| rastrigin_runner(2, cfg));
    builder.add(1.0, &|cfg| griewank_runner(2, cfg));
    builder.add(1.0, &|cfg| ackley_runner(2, cfg));
    builder.add(1000.0, &knapsack_runner);
    builder.add(12.0, &target_string_runner);
    builder.build()
}
