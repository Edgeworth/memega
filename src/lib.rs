#![warn(
    clippy::all,
    clippy::pedantic,
    future_incompatible,
    macro_use_extern_crate,
    meta_variable_misuse,
    missing_abi,
    must_not_suspend,
    nonstandard_style,
    noop_method_call,
    rust_2018_compatibility,
    rust_2018_idioms,
    rust_2021_compatibility,
    trivial_casts,
    unreachable_pub,
    unsafe_code,
    unsafe_op_in_unsafe_fn,
    unused_import_braces,
    unused_lifetimes,
    unused_qualifications,
    unused,
    variant_size_differences
)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::items_after_statements,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]
#![feature(
    array_chunks,
    array_windows,
    bool_to_option,
    const_discriminant,
    const_for,
    const_mut_refs,
    const_trait_impl,
    is_sorted,
    map_first_last,
    must_not_suspend,
    once_cell,
    option_result_contains,
    stmt_expr_attributes,
    trait_alias
)]

pub mod cfg;
pub mod eval;
pub mod examples;
pub mod gen;
pub mod hyper;
pub mod lgp;
pub mod ops;
pub mod run;
pub mod util;
