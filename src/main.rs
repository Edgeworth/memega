use std::time::Duration;

use eyre::Result;
use memega::cfg::Cfg;
use memega::eval::Evaluator;
use memega::examples::all_cfg;
use memega::hyper::hyper_runner;
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
            g.add(
                &format!("{}:{}:best fitness", name, cfg_name),
                run_id,
                r.best_fitness,
            );
            g.add(
                &format!("{}:{}:mean fitness", name, cfg_name),
                run_id,
                r.mean_fitness,
            );
            g.add(
                &format!("{}:{}:dupes", name, cfg_name),
                run_id,
                r.num_dup as f64,
            );
            g.add(
                &format!("{}:{}:mean dist", name, cfg_name),
                run_id,
                r.mean_distance,
            );
            g.add(
                &format!("{}:{}:species", name, cfg_name),
                run_id,
                r.species.num as f64,
            );
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

fn run_once<E: Evaluator>(mut runner: Runner<E>) -> Result<()> {
    for i in 0..1000 {
        let mut r = runner.run_iter()?;
        println!("Generation {}: {}", i + 1, r.nth(0).base_fitness);
        if i % 10 == 0 {
            println!("{}", runner.summary(&mut r));
            println!(
                "{}",
                runner.summary_sample(&mut r, 5, |v| format!("{:?}", v))
            );
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    pretty_env_logger::init_timed();
    color_eyre::install()?;
    // let cfg = none_cfg();
    // let cfg = all_cfg();
    // run_grapher("knapsack", cfg.clone(), &knapsack_runner)?;
    // run_grapher("rastrigin", cfg.clone(), &|cfg| rastrigin_runner(2, cfg))?;
    // run_grapher("griewank", cfg.clone(), &|cfg| griewank_runner(2, cfg))?;
    // run_grapher("ackley", cfg.clone(), &|cfg| ackley_runner(2, cfg))?;
    // run_grapher("string", cfg, &target_string_runner)?;
    // run_once(rastrigin_runner(2, all_cfg()))?;
    run_once(hyper_runner(100, Duration::from_millis(10)))?;
    // run_once(hyper_runner(&knapsack_runner))?;
    // run_once(hyper_runner(&target_string_runner))?;
    // run_once(hyper_runner))?;
    Ok(())
}
