#![warn(rust_2018_idioms, clippy::all)]
#![feature(
    array_chunks,
    array_windows,
    bool_to_option,
    const_fn,
    destructuring_assignment,
    map_first_last,
    option_result_contains,
    option_unwrap_none,
    stmt_expr_attributes,
    trait_alias,
    type_alias_impl_trait
)]

use crate::gen::Params;
use std::fmt;

pub mod cfg;
pub mod distributions;
pub mod examples;
pub mod gen;
pub mod hyper;
pub mod ops;
pub mod runner;

pub trait Genome = Clone + Send + Sync + PartialOrd + PartialEq + fmt::Debug;
pub trait FitnessFn<G: Genome> = Fn(&G) -> f64 + Sync + Send + Clone;

pub type State<T> = (T, Params);

pub trait Evaluator: Send + Sync {
    type Genome: Genome;
    const NUM_CROSSOVER: usize = 2; // Specify the number of crossover operators.
    const NUM_MUTATION: usize = 1; // Specify the number of mutation operators.

    // |idx| specifies which crossover function to use. 0 is conventionally do nothing,
    // with actual crossover starting from index 1.
    fn crossover(&self, s1: &mut Self::Genome, s2: &mut Self::Genome, idx: usize);

    // Unlike crossover, mutation is called for every mutation operator. No need for a nop operator.
    fn mutate(&self, s: &mut Self::Genome, rate: f64, idx: usize);
    fn fitness(&self, s: &Self::Genome) -> f64;
    fn distance(&self, s1: &Self::Genome, s2: &Self::Genome) -> f64;
}
