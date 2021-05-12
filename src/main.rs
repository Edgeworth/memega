use std::f64::consts::PI;
use std::fmt::Display;

use eyre::Result;
use memega::cfg::{
    Cfg, Crossover, Mutation, Niching, Replacement, Species, Stagnation, StagnationCondition,
    Survival,
};
use memega::eval::Evaluator;
use memega::examples::all_cfg;
use memega::lgp::asm::lgp_asm;
use memega::lgp::exec::LgpExec;
use memega::lgp::state::{lgp_runner, State};
use memega::runner::{Runner, RunnerFn, Stats};
use memestat::Grapher;
use rand::Rng;

fn eval_run<E: Evaluator>(
    g: &mut Grapher,
    name: &str,
    run_id: &str,
    base_cfg: Cfg,
    runner_fn: &impl RunnerFn<E>,
) -> Result<()> {
    const SAMPLES: usize = 100;
    let cfgs = [("100 pop", base_cfg)];
    for _ in 0..SAMPLES {
        for (cfg_name, cfg) in cfgs.iter() {
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

fn run_grapher<E: Evaluator>(
    name: &str,
    base_cfg: Cfg,
    runner_fn: &impl RunnerFn<E>,
) -> Result<()> {
    let mut g = Grapher::new();
    let mod_cfg = all_cfg();
    eval_run(&mut g, name, "def", base_cfg, runner_fn)?;
    eval_run(&mut g, name, "mod", mod_cfg, runner_fn)?;
    g.analyse();
    Ok(())
}

fn run_once<E: Evaluator>(mut runner: Runner<E>, print_gen: i32, print_summary: i32) -> Result<()>
where
    E::Genome: Display,
{
    for i in 0..100000 {
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
    // Cfg::new(100)
    //     .with_mutation(Mutation::Adaptive)
    //     .with_crossover(Crossover::Adaptive)
    //     .with_survival(Survival::SpeciesTopProportion(0.1))
    //     .with_species(Species::TargetNumber(10))
    //     .with_niching(Niching::None)
    //     .with_par_fitness(true)
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

fn main() -> Result<()> {
    pretty_env_logger::init_timed();
    color_eyre::install()?;
    // run_grapher("knapsack", cfg.clone(), &knapsack_runner)?;
    // run_grapher("rastrigin", cfg.clone(), &|cfg| rastrigin_runner(2, cfg))?;
    // run_grapher("griewank", cfg.clone(), &|cfg| griewank_runner(2, cfg))?;
    // run_grapher("ackley", cfg.clone(), &|cfg| ackley_runner(2, cfg))?;
    // run_grapher("string", cfg, &target_string_runner)?;
    // run_once(rastrigin_runner(2, all_cfg()))?;
    // run_once(hyper_runner(100, Duration::from_millis(10)))?;
    // run_once(hyper_runner(&knapsack_runner))?;
    // run_once(hyper_runner(&target_string_runner))?;
    // run_once(hyper_runner))?;

    let cfg = lgp_cfg();
    run_once(lgp_runner(4, 6, cfg, lgp_fitness), 10, 100)?;
    run_lgp()?;
    Ok(())
}
