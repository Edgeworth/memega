use std::f64::consts::PI;

use eyre::Result;
use memega::cfg::{
    Cfg, Crossover, Mutation, Niching, Replacement, Species, Stagnation, StagnationCondition,
    Survival,
};
use memega::lgp::asm::lgp_asm;
use memega::lgp::exec::LgpExec;
use memega::lgp::state::{lgp_runner, State};
use memega::run_evolve;
use rand::Rng;
fn lgp_cfg() -> Cfg {
    Cfg::new(2000)
        .with_mutation(Mutation::Adaptive)
        .with_crossover(Crossover::Adaptive)
        .with_survival(Survival::TopProportion(0.1))
        .with_species(Species::None)
        .with_niching(Niching::None)
        .with_stagnation(Stagnation::ContinuousAfter(100))
        .with_stagnation_condition(StagnationCondition::Epsilon(2.0))
        .with_replacement(Replacement::ReplaceChildren(0.5))
        .with_par_fitness(true)
}

fn lgp_fitness(s: &State) -> f64 {
    let mut fitness = 0.0;
    for _ in 0..100 {
        let mut r = rand::thread_rng();
        let mut reg = vec![0.0; s.num_reg]; // Space for work and answer.
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
        fitness += 1.0 / (1.0 + (ans - exec.reg(0)).abs())
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
    run_evolve(lgp_runner(4, 6, cfg, lgp_fitness), 10000, 10, 100)?;
    run_lgp()?;
    Ok(())
}
