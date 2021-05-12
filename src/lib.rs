#![warn(rust_2018_idioms, clippy::all)]
#![feature(
    array_chunks,
    array_windows,
    bool_to_option,
    destructuring_assignment,
    is_sorted,
    map_first_last,
    option_result_contains,
    stmt_expr_attributes,
    trait_alias
)]

use std::fmt::Display;

use eyre::Result;

use crate::eval::Evaluator;
use crate::runner::Runner;

pub mod cfg;
pub mod distributions;
pub mod eval;
pub mod examples;
pub mod gen;
pub mod hyper;
pub mod lgp;
pub mod multirun;
pub mod ops;
pub mod runner;

pub fn run_evolve<E: Evaluator>(
    mut runner: Runner<E>,
    max_gen: usize,
    print_gen: usize,
    print_summary: usize,
) -> Result<()>
where
    E::Genome: Display,
{
    for i in 0..max_gen {
        let mut r = runner.run_iter()?;
        if i % print_gen == 0 {
            println!("Generation {}: {}", i, r.nth(0).base_fitness);
        }
        if i % print_summary == 0 {
            println!("{}", runner.summary(&mut r));
            println!("{}", runner.summary_sample(&mut r, 5, |v| format!("{}", v)));
        }
    }
    Ok(())
}
