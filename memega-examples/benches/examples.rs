use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use memega::evolve::cfg::{Crossover, EvolveCfg, Mutation, Niching, Selection, Species, Survival};
use memega_examples::examples::ackley::ackley_evolver;
use memega_examples::examples::griewank::griewank_evolver;
use memega_examples::examples::hyper::hyper_evolver;
use memega_examples::examples::knapsack::knapsack_evolver;
use memega_examples::examples::rastrigin::rastrigin_evolver;
use memega_examples::examples::target_string::target_string_evolver;

fn get_cfg() -> EvolveCfg {
    EvolveCfg::new(100)
        .set_mutation(Mutation::Adaptive)
        .set_crossover(Crossover::Adaptive)
        .set_survival(Survival::TopProportion(0.25))
        .set_selection(Selection::Sus)
        .set_species(Species::None)
        .set_niching(Niching::None)
        .set_par_dist(false)
        .set_par_fitness(false)
}

fn rastrigin(c: &mut Criterion) {
    c.bench_function("rastrigin", |b| {
        let mut r = rastrigin_evolver(2, get_cfg());
        b.iter(|| r.run())
    });
}

fn griewank(c: &mut Criterion) {
    c.bench_function("griewank", |b| {
        let mut r = griewank_evolver(2, get_cfg());
        b.iter(|| r.run())
    });
}

fn ackley(c: &mut Criterion) {
    c.bench_function("ackley", |b| {
        let mut r = ackley_evolver(2, get_cfg());
        b.iter(|| r.run())
    });
}

fn knapsack(c: &mut Criterion) {
    c.bench_function("knapsack", |b| {
        let mut r = knapsack_evolver(get_cfg());
        b.iter(|| r.run())
    });
}

fn target_string(c: &mut Criterion) {
    c.bench_function("target_string", |b| {
        let mut r = target_string_evolver(get_cfg());
        b.iter(|| r.run())
    });
}

fn hyper(c: &mut Criterion) {
    c.bench_function("hyper", |b| {
        let cfg = get_cfg().set_pop_size(10);
        let mut r = hyper_evolver(2, Duration::from_millis(1), cfg);
        b.iter(|| r.run())
    });
}

criterion_group!(benches, rastrigin, griewank, ackley, knapsack, target_string, hyper);
criterion_main!(benches);
