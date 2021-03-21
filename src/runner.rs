use crate::cfg::{Cfg, Crossover, Mutation, Stagnation};
use crate::gen::evaluated::EvaluatedGen;
use crate::gen::unevaluated::UnevaluatedGen;
use crate::ops::util::rand_vec;
use crate::{Evaluator, Genome};
use derive_more::Display;
use eyre::Result;
use float_pretty_print::PrettyPrintFloat;

pub trait RunnerFn<E: Evaluator> = Fn(Cfg) -> Runner<E> + Sync + Send + Clone + 'static;

#[derive(Debug, Copy, Clone, PartialEq, Display)]
#[display(
    fmt = "best: {}, mean: {}, pop: {}, dupes: {}, dist: {}, species: {}",
    "PrettyPrintFloat(*best_fitness)",
    "PrettyPrintFloat(*mean_fitness)",
    pop_size,
    num_dup,
    "PrettyPrintFloat(*mean_distance)",
    num_species
)]
pub struct Stats {
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub pop_size: usize,
    pub num_dup: usize,
    pub mean_distance: f64,
    pub num_species: usize,
}

impl Stats {
    pub fn from_run<T: Genome, E: Evaluator<Genome = T>>(
        r: &mut RunResult<T>,
        runner: &Runner<E>,
    ) -> Self {
        Self {
            best_fitness: r.gen.best().base_fitness,
            mean_fitness: r.gen.mean_base_fitness(),
            pop_size: r.gen.size(),
            num_dup: r.gen.num_dup(),
            mean_distance: r.gen.dists(runner.cfg(), runner.eval()).mean(),
            num_species: r.gen.num_species(),
        }
    }
}

#[derive(Display, Clone, PartialEq)]
#[display(fmt = "Run({})", gen)]
pub struct RunResult<T: Genome> {
    pub gen: EvaluatedGen<T>,
}

pub trait RandGenome<G: Genome> = FnMut() -> G;

pub struct Runner<E: Evaluator> {
    eval: E,
    cfg: Cfg,
    gen: UnevaluatedGen<E::Genome>,
    rand_genome: Box<dyn RandGenome<E::Genome>>,
    stagnation_count: usize,
    stagnation_fitness: f64,
}

impl<E: Evaluator> Runner<E> {
    pub fn from_gen(
        eval: E,
        cfg: Cfg,
        gen: UnevaluatedGen<E::Genome>,
        rand_genome: impl RandGenome<E::Genome> + 'static,
    ) -> Self {
        Self {
            eval,
            cfg,
            gen,
            rand_genome: Box::new(rand_genome),
            stagnation_count: 0,
            stagnation_fitness: 0.0,
        }
    }

    pub fn new(eval: E, cfg: Cfg, mut rand_genome: impl RandGenome<E::Genome> + 'static) -> Self {
        let gen = UnevaluatedGen::initial::<E>(rand_vec(cfg.pop_size, || rand_genome()), &cfg);
        Self {
            eval,
            cfg,
            gen,
            rand_genome: Box::new(rand_genome),
            stagnation_count: 0,
            stagnation_fitness: 0.0,
        }
    }

    pub fn run_iter(&mut self) -> Result<RunResult<E::Genome>> {
        const REL_ERR: f64 = 1e-12;

        let gen = self.gen.evaluate(&self.cfg, &self.eval)?;
        if (gen.best().base_fitness - self.stagnation_fitness).abs() / self.stagnation_fitness
            < REL_ERR
        {
            self.stagnation_count += 1;
        } else {
            self.stagnation_count = 0;
        }
        self.stagnation_fitness = gen.best().base_fitness;
        let mut genfn = None;
        if let Stagnation::NumGenerations(count) = self.cfg.stagnation {
            if self.stagnation_count >= count {
                genfn = Some(self.rand_genome.as_mut());
                self.stagnation_count = 0;
            }
        }
        let mut next = gen.next_gen(genfn, &self.cfg, &self.eval)?;
        std::mem::swap(&mut next, &mut self.gen);
        Ok(RunResult { gen })
    }

    pub fn cfg(&self) -> &Cfg {
        &self.cfg
    }

    pub fn eval(&self) -> &E {
        &self.eval
    }

    pub fn summary(&self, r: &mut RunResult<E::Genome>) -> String {
        let mut s = String::new();
        s += &format!("{}\n", Stats::from_run(r, &self));
        if self.cfg.mutation == Mutation::Adaptive {
            s += "  mutation weights: ";
            for &v in r.gen.best().state.1.mutation.iter() {
                s += &format!("{}, ", PrettyPrintFloat(v));
            }
            s += "\n";
        }
        if self.cfg.crossover == Crossover::Adaptive {
            s += "  crossover weights: ";
            for &v in r.gen.best().state.1.crossover.iter() {
                s += &format!("{}, ", PrettyPrintFloat(v));
            }
            s += "\n";
        }
        s
    }
}
