use std::mem::swap;
use std::time::{Duration, Instant};

use crate::cfg::{Cfg, Crossover, Mutation, Niching, Selection, Species, Survival};
use crate::eval::Evaluator;
use crate::evaluators::hyper::eval::{HyperAlg, StatFn, State};
use crate::examples::ackley::ackley_runner;
use crate::examples::griewank::griewank_runner;
use crate::examples::knapsack::knapsack_runner;
use crate::examples::rastrigin::rastrigin_runner;
use crate::examples::target_string::target_string_runner;
use crate::run::result::Stats;
use crate::run::runner::{CreateRunnerFn, Runner};

pub struct HyperBuilder {
    stat_fns: Vec<Box<dyn StatFn>>,
    pop_size: usize,
    num_crossover: usize,
    num_mutation: usize,
    sample_dur: Duration,
}

impl HyperBuilder {
    #[must_use]
    pub fn new(pop_size: usize, sample_dur: Duration) -> Self {
        Self { stat_fns: Vec::new(), pop_size, num_crossover: 0, num_mutation: 0, sample_dur }
    }

    /// Add a runner for which we should optimise the hyperparameters for.
    /// Adding multiple runners will optimise a common set of hyperparameters
    /// over all of them.
    pub fn add<F: CreateRunnerFn<E>, E: Evaluator>(&mut self, max_fitness: f64, f: F) {
        self.num_crossover = self.num_crossover.max(E::NUM_CROSSOVER);
        self.num_mutation = self.num_mutation.max(E::NUM_MUTATION);
        let sample_dur = self.sample_dur;
        self.stat_fns.push(Box::new(move |cfg| {
            let mut runner = f(cfg);
            let st = Instant::now();
            let mut r1 = None;
            let mut r2 = None;
            while (Instant::now() - st) < sample_dur {
                swap(&mut r1, &mut r2);
                r2 = Some(runner.run_iter().unwrap());
            }

            // Get the last run that ran in time.
            if let Some(mut r) = r1 {
                let mut stats = Stats::from_run(&mut r);
                stats.best_fitness /= max_fitness;
                stats.mean_fitness /= max_fitness;
                Some(stats)
            } else {
                None
            }
        }));
    }

    #[must_use]
    pub fn build(self) -> Runner<HyperAlg> {
        let cfg = Cfg::new(100)
            .set_mutation(Mutation::Adaptive)
            .set_crossover(Crossover::Adaptive)
            .set_survival(Survival::TopProportion(0.25))
            .set_selection(Selection::Sus)
            .set_species(Species::None)
            .set_niching(Niching::None)
            .set_par_dist(false)
            .set_par_fitness(true);
        let pop_size = self.pop_size;
        let num_crossover = self.num_crossover;
        let num_mutation = self.num_mutation;
        let genomefn = move || State::rand(pop_size, num_crossover, num_mutation);
        Runner::new(HyperAlg::new(self.stat_fns), cfg, genomefn)
    }
}

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
