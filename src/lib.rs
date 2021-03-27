#![warn(rust_2018_idioms, clippy::all)]
#![feature(
    array_chunks,
    array_windows,
    bool_to_option,
    btree_retain,
    const_fn,
    destructuring_assignment,
    is_sorted,
    map_first_last,
    option_result_contains,
    option_unwrap_none,
    stmt_expr_attributes,
    trait_alias,
    type_alias_impl_trait
)]

use crate::cfg::Cfg;
use crate::gen::species::{SpeciesId, NO_SPECIES};
use crate::gen::Params;
use concurrent_lru::sharded::LruCache;
use derive_more::Display;
use float_pretty_print::PrettyPrintFloat;
use std::fmt;
use std::hash::Hash;

pub mod cfg;
pub mod distributions;
pub mod examples;
pub mod gen;
pub mod hyper;
pub mod multirun;
pub mod ops;
pub mod runner;
pub mod eval;
