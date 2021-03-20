use crate::cfg::{Cfg, Crossover, Mutation};
use crate::gen::evaluated::EvaluatedGen;
use crate::gen::unevaluated::UnevaluatedGen;
use crate::{Evaluator, Genome};
use derive_more::Display;
use eyre::Result;

pub trait RunnerFn<E: Evaluator> = Fn(Cfg) -> Runner<E> + Sync + Send + Clone + 'static;

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Stats {
    pub best_fitness: f64,
    pub mean_fitness: f64,
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

pub struct Runner<E: Evaluator> {
    eval: E,
    cfg: Cfg,
    gen: UnevaluatedGen<E::Genome>,
}

impl<E: Evaluator> Runner<E> {
    pub fn new(eval: E, cfg: Cfg, gen: UnevaluatedGen<E::Genome>) -> Self {
        Self { eval, cfg, gen }
    }

    pub fn run_iter(&mut self) -> Result<RunResult<E::Genome>> {
        let evaluated = self.gen.evaluate(&self.cfg, &self.eval)?;
        let mut gen = evaluated.next_gen(&self.cfg, &self.eval)?;
        std::mem::swap(&mut gen, &mut self.gen);
        Ok(RunResult { gen: evaluated })
    }

    pub fn cfg(&self) -> &Cfg {
        &self.cfg
    }

    pub fn eval(&self) -> &E {
        &self.eval
    }

    pub fn summary(&self, r: &mut RunResult<E::Genome>) -> String {
        let mut s = String::new();
        s += &format!("Best: {:?}", Stats::from_run(r, &self));
        if self.cfg.mutation == Mutation::Adaptive {
            s += &format!("  mutation weights: {:?}", r.gen.best().state.1.mutation)
        }
        if self.cfg.crossover == Crossover::Adaptive {
            s += &format!("  crossover weights: {:?}", r.gen.best().state.1.crossover)
        }
        s
    }
}
