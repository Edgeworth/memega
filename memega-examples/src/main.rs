use clap::Parser;
use eyre::Result;
use memega_examples::op::Args;

fn main() -> Result<()> {
    pretty_env_logger::init_timed();
    color_eyre::install()?;

    Args::parse().run()?;

    Ok(())
}
