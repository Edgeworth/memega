use clap::Parser;
use memega::Result;
use memega_examples::op::Args;

fn main() -> Result<()> {
    pretty_env_logger::init_timed();

    Args::parse().run()?;

    Ok(())
}
