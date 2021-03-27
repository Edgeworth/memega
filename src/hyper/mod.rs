use crate::cfg::{Cfg, Crossover, Mutation, Niching, Selection, Species, Survival};
use crate::eval::Evaluator;
use crate::examples::ackley::ackley_runner;
use crate::examples::griewank::griewank_runner;
use crate::examples::knapsack::knapsack_runner;
use crate::examples::rastrigin::rastrigin_runner;
use crate::examples::target_string::target_string_runner;
use crate::ops::crossover::crossover_blx;
use crate::ops::distance::dist2;
use crate::ops::mutation::{mutate_normal, mutate_rate};
use crate::ops::util::rand_vec;
use crate::runner::{Runner, RunnerFn, Stats};
use rand::Rng;
use std::mem::swap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct State {
    cfg: Cfg,
    crossover: Vec<f64>, // Weights for fixed crossover.
    mutation: Vec<f64>,  // Weights for fixed mutation.
}

trait StatFn = Fn(Cfg) -> Option<Stats> + Send + Sync;

pub struct HyperAlg {
    stat_fns: Vec<Box<dyn StatFn>>,
}

impl HyperAlg {
    pub fn new(stat_fns: Vec<Box<dyn StatFn>>) -> Self {
        Self { stat_fns }
    }
}

impl Evaluator for HyperAlg {
    type Genome = State;
    const NUM_CROSSOVER: usize = 4;
    const NUM_MUTATION: usize = 10;

    fn crossover(&self, s1: &mut State, s2: &mut State, idx: usize) {
        let mut r = rand::thread_rng();
        match idx {
            0 => {}
            1 => {
                // Uniform crossover-like operation:
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.crossover, &mut s2.cfg.crossover);
                }
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.mutation, &mut s2.cfg.mutation);
                }
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.survival, &mut s2.cfg.survival);
                }
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.selection, &mut s2.cfg.selection);
                }
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.niching, &mut s2.cfg.niching);
                }
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.species, &mut s2.cfg.species);
                }
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.stagnation, &mut s2.cfg.stagnation);
                }
                if r.gen::<bool>() {
                    swap(&mut s1.cfg.duplicates, &mut s2.cfg.duplicates);
                }
            }
            2 => crossover_blx(&mut s1.crossover, &mut s2.crossover, 0.5),
            3 => crossover_blx(&mut s1.mutation, &mut s2.mutation, 0.5),
            _ => panic!("bug"),
        }
    }

    fn mutate(&self, s: &mut State, rate: f64, idx: usize) {
        let mut r = rand::thread_rng();
        match idx {
            0 => {
                // Mutate crossover - change type
                if r.gen_bool(rate) {
                    match &s.cfg.crossover {
                        Crossover::Fixed(v) => {
                            s.crossover = v.clone();
                            s.cfg.crossover = Crossover::Adaptive;
                        }
                        Crossover::Adaptive => {
                            s.cfg.crossover = Crossover::Fixed(s.crossover.clone());
                        }
                    }
                }
            }
            1 => {
                // Mutate crossover - modify weights
                match &mut s.cfg.crossover {
                    Crossover::Fixed(v) => {
                        mutate_rate(v, 1.0, |v| mutate_normal(v, rate).max(0.0));
                    }
                    Crossover::Adaptive => {
                        mutate_rate(&mut s.crossover, 1.0, |v| mutate_normal(v, rate).max(0.0));
                    }
                }
            }
            2 => {
                // Mutate mutation - change type
                if r.gen_bool(rate) {
                    match &s.cfg.mutation {
                        Mutation::Fixed(v) => {
                            s.mutation = v.clone();
                            s.cfg.mutation = Mutation::Adaptive;
                        }
                        Mutation::Adaptive => {
                            s.cfg.mutation = Mutation::Fixed(s.mutation.clone());
                        }
                    }
                }
            }
            3 => {
                // Mutate mutation - modify weights
                match &mut s.cfg.mutation {
                    Mutation::Fixed(v) => {
                        mutate_rate(v, 1.0, |v| mutate_normal(v, rate).max(0.0));
                    }
                    Mutation::Adaptive => {
                        mutate_rate(&mut s.mutation, 1.0, |v| mutate_normal(v, rate).max(0.0));
                    }
                }
            }
            4 => {
                if r.gen_bool(rate) {
                    s.cfg.survival = r.gen()
                }
            }
            5 => {
                if r.gen_bool(rate) {
                    s.cfg.selection = r.gen()
                }
            }
            6 => {
                if r.gen_bool(rate) {
                    s.cfg.niching = r.gen()
                }
            }
            7 => {
                if r.gen_bool(rate) {
                    s.cfg.species = r.gen()
                }
            }
            8 => {
                if r.gen_bool(rate) {
                    s.cfg.stagnation = r.gen()
                }
            }
            9 => {
                if r.gen_bool(rate) {
                    s.cfg.duplicates = r.gen()
                }
            }
            _ => panic!("bug"),
        }
    }

    fn fitness(&self, s: &State) -> f64 {
        const SAMPLES: usize = 30;
        let mut score = 0.0;
        for _ in 0..SAMPLES {
            for f in self.stat_fns.iter() {
                if let Some(r) = f(s.cfg.clone()) {
                    // TODO: Need multi-objective GA here. Or at least configure
                    // what to optimise.
                    score += r.best_fitness;
                }
            }
        }
        score / SAMPLES as f64
    }

    fn distance(&self, s1: &State, s2: &State) -> f64 {
        let mut dist = 0.0;

        let s1_cross = if let Crossover::Fixed(v) = &s1.cfg.crossover {
            v
        } else {
            &s1.crossover
        };
        let s2_cross = if let Crossover::Fixed(v) = &s2.cfg.crossover {
            v
        } else {
            &s2.crossover
        };
        dist += dist2(s1_cross, s2_cross);

        let s1_mutation = if let Mutation::Fixed(v) = &s1.cfg.mutation {
            v
        } else {
            &s1.mutation
        };
        let s2_mutation = if let Mutation::Fixed(v) = &s2.cfg.mutation {
            v
        } else {
            &s2.mutation
        };
        dist += dist2(s1_mutation, s2_mutation);

        dist
    }
}

pub struct HyperBuilder {
    stat_fns: Vec<Box<dyn StatFn>>,
    pop_size: usize,
    num_crossover: usize,
    num_mutation: usize,
    sample_dur: Duration,
}

impl HyperBuilder {
    pub fn new(pop_size: usize, sample_dur: Duration) -> Self {
        Self {
            stat_fns: Vec::new(),
            pop_size,
            num_crossover: 0,
            num_mutation: 0,
            sample_dur,
        }
    }

    fn rand_state(pop_size: usize, num_crossover: usize, num_mutation: usize) -> State {
        let mut r = rand::thread_rng();
        let crossover = rand_vec(num_crossover, || r.gen());
        let mutation = rand_vec(num_mutation, || r.gen());
        let mut cfg = Cfg::new(pop_size);
        cfg.survival = r.gen();
        cfg.selection = r.gen();
        cfg.niching = r.gen();
        cfg.species = r.gen();
        State {
            cfg,
            crossover,
            mutation,
        }
    }

    pub fn add<F: RunnerFn<E>, E: Evaluator>(&mut self, max_fitness: f64, f: F) {
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

    pub fn build(self) -> Runner<HyperAlg> {
        let cfg = Cfg::new(100)
            .with_mutation(Mutation::Adaptive)
            .with_crossover(Crossover::Adaptive)
            .with_survival(Survival::TopProportion(0.25))
            .with_selection(Selection::Sus)
            .with_species(Species::None)
            .with_niching(Niching::None)
            .with_par_dist(false)
            .with_par_fitness(true);
        let pop_size = self.pop_size;
        let num_crossover = self.num_crossover;
        let num_mutation = self.num_mutation;
        let genomefn = move || Self::rand_state(pop_size, num_crossover, num_mutation);
        Runner::new(HyperAlg::new(self.stat_fns), cfg, genomefn)
    }
}

pub fn hyper_runner(pop_size: usize, sample_dur: Duration) -> Runner<HyperAlg> {
    let mut builder = HyperBuilder::new(pop_size, sample_dur);
    builder.add(1.0, &|cfg| rastrigin_runner(2, cfg));
    builder.add(1.0, &|cfg| griewank_runner(2, cfg));
    builder.add(1.0, &|cfg| ackley_runner(2, cfg));
    builder.add(1000.0, &knapsack_runner);
    builder.add(12.0, &target_string_runner);
    builder.build()
}
