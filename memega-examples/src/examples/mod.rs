use memega::evolve::cfg::{Crossover, EvolveCfg, Mutation, Niching, Species, Survival};

pub mod ackley;
pub mod func;
pub mod griewank;
pub mod hyper;
pub mod knapsack;
pub mod lgp;
pub mod rastrigin;
pub mod target_string;

#[must_use]
pub fn all_cfg() -> EvolveCfg {
    EvolveCfg::new(100)
        .set_mutation(Mutation::Adaptive)
        .set_crossover(Crossover::Adaptive)
        .set_survival(Survival::SpeciesTopProportion(0.1))
        .set_species(Species::TargetNumber(10))
        .set_niching(Niching::SpeciesSharedFitness)
}

#[must_use]
pub fn none_cfg() -> EvolveCfg {
    EvolveCfg::new(100)
        .set_mutation(Mutation::Fixed(vec![0.9, 0.1]))
        .set_crossover(Crossover::Fixed(vec![0.3, 0.7]))
}
