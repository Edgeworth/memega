use crate::cfg::{Cfg, Crossover, Mutation, Niching, Species, Survival};

pub mod ackley;
pub mod func;
pub mod griewank;
pub mod knapsack;
pub mod rastrigin;
pub mod target_string;

#[must_use]
pub fn all_cfg() -> Cfg {
    Cfg::new(100)
        .with_mutation(Mutation::Adaptive)
        .with_crossover(Crossover::Adaptive)
        .with_survival(Survival::SpeciesTopProportion(0.1))
        .with_species(Species::TargetNumber(10))
        .with_niching(Niching::SpeciesSharedFitness)
}

#[must_use]
pub fn none_cfg() -> Cfg {
    Cfg::new(100)
        .with_mutation(Mutation::Fixed(vec![0.9, 0.1]))
        .with_crossover(Crossover::Fixed(vec![0.3, 0.7]))
}
