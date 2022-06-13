use rand::Rng;
use rand_distr::{Distribution, Standard};

use crate::gen::species::SpeciesId;

#[must_use]
#[derive(Debug, Clone, PartialEq, PartialOrd)]
// Only one crossover function will be applied at a time.
pub enum Crossover {
    // Fixed with given rate. Specify the weights for each crossover function.
    Fixed(Vec<f64>),
    // Adaptive - uses 1/sqrt(pop size) as learning rate.
    Adaptive,
}

#[must_use]
#[derive(Debug, Clone, PartialEq, PartialOrd)]
// Each mutation function will be applied with the given rate. This is different to crossover,
// which is only applied once.
pub enum Mutation {
    // Fixed with given rate. Specify the weights for each mutation function.
    Fixed(Vec<f64>),
    // Adaptive - uses 1/sqrt(pop size) as learning rate.
    Adaptive,
}

#[must_use]
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Survival {
    TopProportion(f64),
    SpeciesTopProportion(f64), // Top proportion for each species.
    Youngest,                  // Only the youngest members survive. Age based replacement.
    Tournament(usize),         // Tournament selection. Tournament size is given.
}

impl Distribution<Survival> for Standard {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Survival {
        match r.gen_range(0..2) {
            0 => Survival::TopProportion(r.gen_range(0.0..0.9)),
            _ => Survival::SpeciesTopProportion(r.gen_range(0.0..0.9)),
        }
    }
}

#[must_use]
#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum Selection {
    Sus,
    Roulette,
}

impl Distribution<Selection> for Standard {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Selection {
        match r.gen_range(0..2) {
            0 => Selection::Sus,
            _ => Selection::Roulette,
        }
    }
}

#[must_use]
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Niching {
    None,
    SharedFitness(f64),   // Takes a distance for fitness sharing
    SpeciesSharedFitness, // Derives sharing distance from species information.
}

impl Distribution<Niching> for Standard {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Niching {
        match r.gen_range(0..3) {
            0 => Niching::None,
            1 => Niching::SharedFitness(r.gen_range(0.0..100.0)), // TODO: Hardcoded.
            _ => Niching::SpeciesSharedFitness,
        }
    }
}

#[must_use]
#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum Species {
    None,
    TargetNumber(SpeciesId), // Target number of species.
}

impl Distribution<Species> for Standard {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Species {
        match r.gen_range(0..2) {
            0 => Species::None,
            _ => Species::TargetNumber(r.gen_range(1..10)), // TODO: Hardcoded.
        }
    }
}

#[must_use]
#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum Stagnation {
    None,
    // After N generations of the same best fitness, trigger stagnation once.
    OneShotAfter(usize),
    // Stagnation continuously after N generations of the same best fitness.
    ContinuousAfter(usize),
}

impl Distribution<Stagnation> for Standard {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Stagnation {
        match r.gen_range(0..2) {
            0 => Stagnation::None,
            1 => Stagnation::OneShotAfter(r.gen_range(1..1000)),
            _ => Stagnation::ContinuousAfter(r.gen_range(1..1000)),
        }
    }
}

#[must_use]
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum StagnationCondition {
    // Use default epsilon and relative comparison to determine stagnation.
    Default,
    // Compare fitnesses to determine stagnation with this epsilon. Useful if
    // the fitness is somewhat random.
    Epsilon(f64),
}

impl Distribution<StagnationCondition> for Standard {
    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> StagnationCondition {
        // Just return default for now - evolving a stagnation condition epsilon
        // probably not that useful.
        StagnationCondition::Default
    }
}

#[must_use]
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Replacement {
    // During stagnation make a proportion of the children with random individuals.
    ReplaceChildren(f64),
}

impl Distribution<Replacement> for Standard {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Replacement {
        Replacement::ReplaceChildren(r.gen())
    }
}

#[must_use]
#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum Duplicates {
    DisallowDuplicates, // Don't allow duplicate states in the population.
    AllowDuplicates,
}

impl Distribution<Duplicates> for Standard {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Duplicates {
        match r.gen_range(0..1) {
            0 => Duplicates::DisallowDuplicates,
            _ => Duplicates::AllowDuplicates,
        }
    }
}

/// How to combine fitnesses for a single member, if multiple inputs are
/// given (`Evaluator::Data`)
#[must_use]
#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum FitnessReduction {
    ArithmeticMean,
    GeometricMean,
}

impl Distribution<FitnessReduction> for Standard {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> FitnessReduction {
        match r.gen_range(0..1) {
            0 => FitnessReduction::ArithmeticMean,
            _ => FitnessReduction::GeometricMean,
        }
    }
}

#[must_use]
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct EvolveCfg {
    pub pop_size: usize,
    pub crossover: Crossover,
    pub mutation: Mutation, // Mutation rate per bit / basic block.
    pub survival: Survival,
    pub selection: Selection,
    pub niching: Niching,
    pub species: Species,
    pub stagnation: Stagnation,
    pub stagnation_condition: StagnationCondition,
    pub replacement: Replacement,
    pub duplicates: Duplicates,
    pub fitness_reduction: FitnessReduction,

    /// Run fitness computations in parallel
    pub par_fitness: bool,

    /// Run distance computations in parallel
    pub par_dist: bool,
}

impl EvolveCfg {
    pub fn new(pop_size: usize) -> Self {
        Self {
            pop_size,
            crossover: Crossover::Adaptive,
            mutation: Mutation::Adaptive,
            survival: Survival::TopProportion(0.2),
            selection: Selection::Sus,
            niching: Niching::None,
            species: Species::None,
            stagnation: Stagnation::None,
            stagnation_condition: StagnationCondition::Default,
            replacement: Replacement::ReplaceChildren(0.2),
            duplicates: Duplicates::DisallowDuplicates,
            fitness_reduction: FitnessReduction::ArithmeticMean,
            par_fitness: false,
            par_dist: false,
        }
    }

    pub fn set_pop_size(self, pop_size: usize) -> Self {
        Self { pop_size, ..self }
    }

    pub fn set_crossover(self, crossover: Crossover) -> Self {
        Self { crossover, ..self }
    }

    pub fn set_mutation(self, mutation: Mutation) -> Self {
        Self { mutation, ..self }
    }

    pub fn set_survival(self, survival: Survival) -> Self {
        Self { survival, ..self }
    }

    pub fn set_selection(self, selection: Selection) -> Self {
        Self { selection, ..self }
    }

    pub fn set_niching(self, niching: Niching) -> Self {
        Self { niching, ..self }
    }

    pub fn set_species(self, species: Species) -> Self {
        Self { species, ..self }
    }

    pub fn set_stagnation(self, stagnation: Stagnation) -> Self {
        Self { stagnation, ..self }
    }

    pub fn set_stagnation_condition(self, stagnation_condition: StagnationCondition) -> Self {
        Self { stagnation_condition, ..self }
    }

    pub fn set_replacement(self, replacement: Replacement) -> Self {
        Self { replacement, ..self }
    }

    pub fn set_duplicates(self, duplicates: Duplicates) -> Self {
        Self { duplicates, ..self }
    }

    pub fn set_fitness_reduction(self, fitness_reduction: FitnessReduction) -> Self {
        Self { fitness_reduction, ..self }
    }

    pub fn set_par_fitness(self, par_fitness: bool) -> Self {
        Self { par_fitness, ..self }
    }

    pub fn set_par_dist(self, par_dist: bool) -> Self {
        Self { par_dist, ..self }
    }
}
