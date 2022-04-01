use clap::Parser;
use eyre::Result;
use memega::run::runner::CreateRunnerFn;
use memega_examples::MemeGaOp;

#[derive(Debug, Parser)]
#[clap(name = "memega cli", about = "memega cli")]
pub struct Args {
    #[clap(arg_enum)]
    pub op: MemeGaOp,
}

fn main() -> Result<()> {
    pretty_env_logger::init_timed();
    color_eyre::install()?;
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
    Ok(())
}
