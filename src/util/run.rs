use std::fmt;

use eyre::Result;

use crate::eval::Evaluator;
use crate::run::result::RunResult;
use crate::run::runner::Runner;

pub fn run_evolve<E: Evaluator>(
    mut runner: Runner<E>,
    max_gen: usize,
    print_gen: usize,
    print_summary: usize,
) -> Result<RunResult<E::Genome>>
where
    E::Genome: fmt::Display,
{
    let mut ret = None;
    for i in 0..max_gen {
        let mut r = runner.run_iter()?;
        if i % print_gen == 0 {
            println!("Generation {}: {}", i, r.nth(0).base_fitness);
        }
        if i % print_summary == 0 {
            println!("{}", runner.summary(&mut r));
            println!("{}", runner.summary_sample(&mut r, 5, |v| format!("{}", v)));
        }
        ret = Some(r);
    }
    Ok(ret.unwrap())
}

pub fn run_evolve_debug<E: Evaluator>(
    mut runner: Runner<E>,
    max_gen: usize,
    print_gen: usize,
    print_summary: usize,
) -> Result<RunResult<E::Genome>>
where
    E::Genome: fmt::Debug,
{
    let mut ret = None;
    for i in 0..max_gen {
        let mut r = runner.run_iter()?;
        if i % print_gen == 0 {
            println!("Generation {}: {}", i, r.nth(0).base_fitness);
        }
        if i % print_summary == 0 {
            println!("{}", runner.summary(&mut r));
            println!("{}", runner.summary_sample(&mut r, 5, |v| format!("{:?}", v)));
        }
        ret = Some(r);
    }
    Ok(ret.unwrap())
}
