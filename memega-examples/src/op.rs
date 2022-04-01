use clap::{ArgEnum, Parser};
use eyre::Result;
use memega::cfg::{
    Cfg, Crossover, Mutation, Niching, Replacement, Species, Stagnation, StagnationCondition,
    Survival,
};
use memega::eval::Evaluator;
use memega::harness::cfg::{HarnessCfg, Termination};
use memega::run::result::Stats;
use memega::run::runner::CreateRunnerFn;
use memestat::Grapher;

use crate::examples::all_cfg;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, ArgEnum)]
pub enum MemeGaOp {
    Ackley,
    Griewank,
    Hyper,
    Knapsack,
    Rastringin,
    TargetString,
    Lgp,
}

#[derive(Debug, Parser)]
#[clap(name = "memega cli", about = "memega cli")]
pub struct Args {
    #[clap(arg_enum)]
    pub op: MemeGaOp,

    #[clap(long, default_value = "2000", help = "population size")]
    pub pop_size: usize,

    #[clap(long, default_value = "2000", help = "number of generation")]
    pub num_gen: usize,
}

impl Args {
    pub fn run_op(&self) -> Result<()> {
        match self.op {}
        // run_grapher("knapsack", cfg.clone(), &knapsack_runner)?;
        // run_grapher("rastrigin", cfg.clone(), &|cfg| rastrigin_runner(2, cfg))?;
        // run_grapher("griewank", cfg.clone(), &|cfg| griewank_runner(2, cfg))?;
        // run_grapher("ackley", cfg.clone(), &|cfg| ackley_runner(2, cfg))?;
        // run_grapher("string", cfg, &target_string_runner)?;
        // run_once(rastrigin_runner(2, all_cfg()))?;
        // run_once(hyper_runner(100, Duration::from_millis(10)))?;
        // run_once(hyper_runner(&knapsack_runner))?;
        // run_once(hyper_runner(&target_string_runner))?;
        // run_once(hyper_runner)?;
        // run_evolve(lgp_runner_fn(LgpCfg::new(), cfg, lgp_fitness), 10000, 10, 100)?;

        Ok(())
    }

    fn cfg(&self) -> Cfg {
        Cfg::new(self.pop_size)
            .set_mutation(Mutation::Adaptive)
            .set_crossover(Crossover::Adaptive)
            .set_survival(Survival::TopProportion(0.1))
            .set_species(Species::None)
            .set_niching(Niching::None)
            .set_stagnation(Stagnation::ContinuousAfter(100))
            .set_stagnation_condition(StagnationCondition::Epsilon(2.0))
            .set_replacement(Replacement::ReplaceChildren(0.5))
            .set_par_fitness(true)
    }

    fn harness_cfg(&self) -> HarnessCfg {
        HarnessCfg::new()
            .set_termination(Termination::FixedGenerations(self.num_gen))
            .set_print_gen(Some(10))
            .set_print_summary(Some(10))
    }

    #[allow(unused)]
    fn eval_run<E: Evaluator>(
        &self,
        g: &mut Grapher,
        name: &str,
        run_id: &str,
        base_cfg: Cfg,
        runner_fn: &impl CreateRunnerFn<E>,
    ) -> Result<()> {
        const SAMPLES: usize = 100;
        let cfgs = [("100 pop", base_cfg)];
        for _ in 0..SAMPLES {
            for (cfg_name, cfg) in &cfgs {
                let mut runner = runner_fn(cfg.clone());
                for _ in 0..100 {
                    runner.run_iter()?;
                }
                let r = Stats::from_run(&mut runner.run_iter()?);
                g.add(&format!("{}:{}:best fitness", name, cfg_name), run_id, r.best_fitness);
                g.add(&format!("{}:{}:mean fitness", name, cfg_name), run_id, r.mean_fitness);
                g.add(&format!("{}:{}:dupes", name, cfg_name), run_id, r.num_dup as f64);
                g.add(&format!("{}:{}:mean dist", name, cfg_name), run_id, r.mean_distance);
                g.add(&format!("{}:{}:species", name, cfg_name), run_id, r.species.num as f64);
            }
        }
        Ok(())
    }

    #[allow(unused)]
    fn run_grapher<E: Evaluator>(
        &self,
        name: &str,
        base_cfg: Cfg,
        runner_fn: &impl CreateRunnerFn<E>,
    ) -> Result<()> {
        let mut g = Grapher::new();
        let mod_cfg = all_cfg();
        self.eval_run(&mut g, name, "def", base_cfg, runner_fn)?;
        self.eval_run(&mut g, name, "mod", mod_cfg, runner_fn)?;
        g.analyse();
        Ok(())
    }
}
