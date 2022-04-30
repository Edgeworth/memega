use clap::{ArgEnum, Parser};
use eyre::Result;
use memega::eval::{Data, Evaluator};
use memega::evaluators::lgp::cfg::LgpEvaluatorCfg;
use memega::evolve::cfg::{
    Crossover, EvolveCfg, Mutation, Niching, Replacement, Species, Stagnation, StagnationCondition,
    Survival,
};
use memega::evolve::evolver::CreateEvolverFn;
use memega::evolve::result::Stats;
use memega::train::cfg::{Termination, TrainerCfg};
use memega::train::sampler::{DataSampler, EmptyDataSampler};
use memega::train::trainer::Trainer;

use crate::examples::ackley::ackley_evolver;
use crate::examples::expr::{expr_evolver, ExprDataSampler};
use crate::examples::griewank::griewank_evolver;
use crate::examples::knapsack::knapsack_evolver;
use crate::examples::rastrigin::rastrigin_evolver;
use crate::examples::target_string::target_string_evolver;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, ArgEnum)]
pub enum Example {
    Ackley,
    Griewank,
    Knapsack,
    Rastringin,
    TargetString,
    Lgp,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, ArgEnum)]
pub enum Op {
    Run,
}

#[derive(Debug, Parser)]
#[clap(name = "memega cli", about = "memega cli")]
pub struct Args {
    #[clap(arg_enum, help = "which operation to run")]
    pub op: Op,

    #[clap(arg_enum, help = "which example problem to solve")]
    pub example: Example,

    #[clap(
        long,
        default_value = "2",
        help = "dimension size for mathematical function example problems"
    )]
    pub func_dim: usize,

    #[clap(
        long,
        default_value = "x^2 + x + 1",
        help = "equation involving x for lgp to evolve (e.g. x^2 + x + 1)"
    )]
    pub lgp_target: String,

    #[clap(long, default_value = "2000", help = "population size")]
    pub pop_size: usize,

    #[clap(long, default_value = "2000", help = "number of generation")]
    pub num_gen: usize,

    #[clap(long, help = "how often to report to tensorboard")]
    pub report_gen: Option<usize>,
}

impl Args {
    fn cfg(&self) -> EvolveCfg {
        EvolveCfg::new(self.pop_size)
            .set_mutation(Mutation::Adaptive)
            .set_crossover(Crossover::Adaptive)
            .set_survival(Survival::TopProportion(0.1))
            .set_species(Species::None)
            .set_niching(Niching::None)
            .set_stagnation(Stagnation::ContinuousAfter(100))
            .set_stagnation_condition(StagnationCondition::Epsilon(0.5))
            .set_replacement(Replacement::ReplaceChildren(0.1))
            .set_par_fitness(true)
    }

    fn trainer_cfg(&self) -> TrainerCfg {
        let mut cfg = TrainerCfg::new()
            .set_termination(Termination::FixedGenerations(self.num_gen))
            .set_print_gen(10)
            .set_print_summary(10);
        cfg.report_gen = self.report_gen;
        cfg
    }

    pub fn run(&self) -> Result<()> {
        let func_dim = self.func_dim;
        let lgp_target = self.lgp_target.clone();
        let lgpcfg = LgpEvaluatorCfg::new();
        match self.example {
            Example::Ackley => {
                self.dispatch(move |cfg| ackley_evolver(func_dim, cfg), &EmptyDataSampler {})
            }
            Example::Griewank => {
                self.dispatch(move |cfg| griewank_evolver(func_dim, cfg), &EmptyDataSampler {})
            }
            Example::Knapsack => self.dispatch(knapsack_evolver, &EmptyDataSampler {}),
            Example::Rastringin => {
                self.dispatch(move |cfg| rastrigin_evolver(func_dim, cfg), &EmptyDataSampler {})
            }
            Example::TargetString => self.dispatch(target_string_evolver, &EmptyDataSampler {}),
            Example::Lgp => self.dispatch(
                move |cfg| expr_evolver(lgp_target.clone(), lgpcfg.clone(), cfg),
                &ExprDataSampler::new(),
            ),
        }
    }

    fn dispatch<D: Data, E: Evaluator<Data = D>>(
        &self,
        create_fn: impl CreateEvolverFn<E>,
        sampler: &impl DataSampler<E::Data>,
    ) -> Result<()> {
        match self.op {
            Op::Run => self.run_op(create_fn, sampler)?,
        }
        Ok(())
    }

    fn run_op<E: Evaluator>(
        &self,
        create_fn: impl CreateEvolverFn<E>,
        sampler: &impl DataSampler<E::Data>,
    ) -> Result<()> {
        let evolver = create_fn(self.cfg());
        let mut trainer = Trainer::new(self.trainer_cfg());
        let mut r = trainer.train(evolver, sampler)?;
        println!("Stats:");
        println!("{}", Stats::from_result(&mut r));
        Ok(())
    }
}
