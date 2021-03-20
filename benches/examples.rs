use criterion::{criterion_group, criterion_main, Criterion};
use memega::cfg::{Cfg, Crossover, Mutation, Niching, Selection, Species, Survival};
use memega::examples::ackley::ackley_runner;
use memega::examples::griewank::griewank_runner;
use memega::examples::knapsack::knapsack_runner;
use memega::examples::rastrigin::rastrigin_runner;
use memega::examples::target_string::target_string_runner;
use memega::hyper::hyper_runner;
use std::time::Duration;

fn get_cfg() -> Cfg {
    Cfg::new(100)
        .with_mutation(Mutation::Adaptive)
        .with_crossover(Crossover::Adaptive)
        .with_survival(Survival::TopProportion(0.25))
        .with_selection(Selection::Sus)
        .with_species(Species::None)
        .with_niching(Niching::None)
        .with_par_dist(false)
        .with_par_fitness(false)
}

fn rastrigin(c: &mut Criterion) {
    c.bench_function("rastrigin", |b| {
        let mut r = rastrigin_runner(2, get_cfg());
        b.iter(|| r.run_iter())
    });
}

fn griewank(c: &mut Criterion) {
    c.bench_function("griewank", |b| {
        let mut r = griewank_runner(2, get_cfg());
        b.iter(|| r.run_iter())
    });
}

fn ackley(c: &mut Criterion) {
    c.bench_function("ackley", |b| {
        let mut r = ackley_runner(2, get_cfg());
        b.iter(|| r.run_iter())
    });
}

fn knapsack(c: &mut Criterion) {
    c.bench_function("knapsack", |b| {
        let mut r = knapsack_runner(get_cfg());
        b.iter(|| r.run_iter())
    });
}

fn target_string(c: &mut Criterion) {
    c.bench_function("target_string", |b| {
        let mut r = target_string_runner(get_cfg());
        b.iter(|| r.run_iter())
    });
}

fn hyper(c: &mut Criterion) {
    c.bench_function("hyper", |b| {
        let mut r = hyper_runner(Duration::from_millis(1));
        b.iter(|| r.run_iter())
    });
}

criterion_group!(
    benches,
    rastrigin,
    griewank,
    ackley,
    knapsack,
    target_string,
    hyper
);
criterion_main!(benches);
