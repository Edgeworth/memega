#![warn(rust_2018_idioms, clippy::all)]
#![feature(
    array_chunks,
    array_windows,
    bool_to_option,
    const_fn,
    destructuring_assignment,
    is_sorted,
    map_first_last,
    option_result_contains,
    stmt_expr_attributes,
    trait_alias
)]

pub mod cfg;
pub mod distributions;
pub mod eval;
pub mod examples;
pub mod gen;
pub mod hyper;
pub mod multirun;
pub mod ops;
pub mod runner;
