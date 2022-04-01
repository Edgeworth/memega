use clap::Parser;
use eyre::Result;
use memega::run::runner::CreateRunnerFn;
use memega_examples::op::Args;

fn main() -> Result<()> {
    pretty_env_logger::init_timed();
    color_eyre::install()?;

    Args::parse().run_op()?;

    Ok(())
}
