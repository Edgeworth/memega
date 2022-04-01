#![warn(
    clippy::all,
    clippy::pedantic,
    future_incompatible,
    macro_use_extern_crate,
    meta_variable_misuse,
    missing_abi,
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
#![feature(array_chunks, is_sorted, trait_alias)]
pub mod examples;




use memega::util::run::run_evolve;
use memega::evaluators::lgp::builder::lgp_runner_fn;
use memega::evaluators::lgp::cfg::LgpCfg;
use std::f64::consts::PI;

use clap::ArgEnum;
use eyre::Result;
use memega::cfg::{
    Cfg, Crossover, Mutation, Niching, Replacement, Species, Stagnation, StagnationCondition,
    Survival,
};
use memega::eval::Evaluator;


use memega::evaluators::lgp::eval::State;
use memega::evaluators::lgp::vm::asm::lgp_asm;
use memega::evaluators::lgp::vm::exec::LgpExec;
use memega::run::result::Stats;
use memega::run::runner::CreateRunnerFn;

use memestat::Grapher;
use rand::Rng;

use crate::examples::all_cfg;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, ArgEnum)]
pub enum MemeGaOp {
    Lgp,
}

fn lgp_cfg() -> Cfg {
    Cfg::new(2000)
        .set_mutation(Mutation::Adaptive)
        .set_crossover(Crossover::Adaptive)
        .set_survival(Survival::TopProportion(0.1))
        .set_species(Species::None)
        .set_niching(Niching::None)
        .set_stagnation(Stagnation::ContinuousAfter(100))
        .set_stagnation_condition(StagnationCondition::Epsilon(2.0))
        .set_replacement(Replacement::ReplaceChildren(0.5))
        .set_par_fitness(true)
}

fn lgp_fitness(s: &State, _gen: usize) -> f64 {
    let mut fitness = 0.0;
    for _ in 0..100 {
        let mut r = rand::thread_rng();
        let mut reg = vec![0.0; s.cfg.num_reg()]; // Space for work and answer.
        let x = r.gen_range(0.0..100.0);
        reg[1] = -1.0;
        reg[2] = 1.0;
        reg[3] = x;
        let mut exec = LgpExec::new(&reg, &s.ops, 200);
        exec.run();

        let mut ans = 0.0;
        for i in 1..(x as usize) {
            ans += 1.0 / (i as f64);
        }
        fitness += 1.0 / (1.0 + (ans - exec.reg(0)).abs());
    }
    fitness + 1.0 / (1.0 + s.ops.len() as f64)
}

fn run_lgp() -> Result<()> {
    use plotters::prelude::*;

    let code = lgp_asm(
        "add r0, r3
div r1, r0
abs r3
mul r0, r0
add r0, r3
add r0, r1
",
    )?;

    let xleft = -PI;
    let xright = PI;

    let root = BitMapBackend::new("test.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("stuff", ("sans-serif", 50).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(xleft..xright, -50.0..50.0)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            (-50..=50).map(|x| x as f64 / 50.0 * (xright - xleft)).map(|x| {
                let mut lgp = LgpExec::new(&[0.0, -1.0, 1.0, x], &code, 200);
                lgp.run();
                (x, lgp.reg(0))
            }),
            &RED,
        ))?
        .label("y = stuff");

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

#[test]
fn test_lgp() -> Result<()> {
    let cfg = lgp_cfg();
    run_evolve(lgp_runner_fn(LgpCfg::new(), cfg, lgp_fitness), 10000, 10, 100)?;
    run_lgp()?;
    Ok(())
}

#[allow(unused)]
fn eval_run<E: Evaluator>(
    g: &mut Grapher,
    name: &str,
    run_id: &str,
    base_cfg: Cfg,
    runner_fn: &impl CreateRunnerFn<E>,
) -> Result<()> {
    const SAMPLES: usize = 100;
    let cfgs = [("100 pop", base_cfg)];
    for _ in 0..SAMPLES {
        for (cfg_name, cfg) in &cfgs {
            let mut runner = runner_fn(cfg.clone());
            for _ in 0..100 {
                runner.run_iter()?;
            }
            let r = Stats::from_run(&mut runner.run_iter()?);
            g.add(&format!("{}:{}:best fitness", name, cfg_name), run_id, r.best_fitness);
            g.add(&format!("{}:{}:mean fitness", name, cfg_name), run_id, r.mean_fitness);
            g.add(&format!("{}:{}:dupes", name, cfg_name), run_id, r.num_dup as f64);
            g.add(&format!("{}:{}:mean dist", name, cfg_name), run_id, r.mean_distance);
            g.add(&format!("{}:{}:species", name, cfg_name), run_id, r.species.num as f64);
        }
    }
    Ok(())
}

#[allow(unused)]
fn run_grapher<E: Evaluator>(
    name: &str,
    base_cfg: Cfg,
    runner_fn: &impl CreateRunnerFn<E>,
) -> Result<()> {
    let mut g = Grapher::new();
    let mod_cfg = all_cfg();
    eval_run(&mut g, name, "def", base_cfg, runner_fn)?;
    eval_run(&mut g, name, "mod", mod_cfg, runner_fn)?;
    g.analyse();
    Ok(())
}
