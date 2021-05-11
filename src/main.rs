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
use memega::lgp::state::lgp_runner;
use memega::runner::{Runner, RunnerFn, Stats};
use memestat::Grapher;

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
    for i in 0..10000 {
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
    let code = lgp_asm(
        "mov r0, r3
abs r3
abs r2
jle r3, r2, -2
pow r2, r3
add r1, r1
div r1, r3
div r0, r3
add r3, r1
jle r3, r3, -8",
    )?;

    let x = PI / 7.0;
    let mut lgp = LgpExec::new(&[0.0, -1.0, 1.0, x], &code, 200);
    lgp.run();
    println!("result: {}", lgp.reg(0));
    Ok(())
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
    run_once(lgp_runner(10, cfg), 10, 100)?;
    // run_lgp()?;
    Ok(())
}
