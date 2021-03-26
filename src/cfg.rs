use rand::Rng;
use rand_distr::{Distribution, Standard};

pub const EP: f64 = 1.0e-6;

#[derive(Debug, Clone, PartialEq, PartialOrd)]
// Only one crossover function will be applied at a time.
pub enum Crossover {
    // Fixed with given rate. Specify the weights for each crossover function.
    Fixed(Vec<f64>),
    // Adaptive - uses 1/sqrt(pop size) as learning rate.
    Adaptive,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
// Each mutation function will be applied with the given rate. This is different to crossover,
// which is only applied once.
pub enum Mutation {
    // Fixed with given rate. Specify the weights for each mutation function.
    Fixed(Vec<f64>),
    // Adaptive - uses 1/sqrt(pop size) as learning rate.
    Adaptive,
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd)]
pub enum Survival {
    TopProportion(f64),
    SpeciesTopProportion(f64), // Top proportion for each species.
}

impl Distribution<Survival> for Standard {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Survival {
        match r.gen_range(0..2) {
            0 => Survival::TopProportion(r.gen_range(0.0..0.9)),
            _ => Survival::SpeciesTopProportion(r.gen_range(0.0..0.9)),
        }
    }
}

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

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum Niching {
    None,
    SharedFitness,
}

impl Distribution<Niching> for Standard {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Niching {
        match r.gen_range(0..2) {
            0 => Niching::None,
            _ => Niching::SharedFitness,
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum Species {
    None,
    TargetNumber(u64), // Target number of species.
}

impl Distribution<Species> for Standard {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Species {
        match r.gen_range(0..2) {
            0 => Species::None,
            _ => Species::TargetNumber(r.gen_range(1..10)),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum Stagnation {
    None,
    // After N generations of stagnation, don't do crossover/mutation - replace with random individuals.
    NumGenerations(usize),
}

impl Distribution<Stagnation> for Standard {
    fn sample<R: Rng + ?Sized>(&self, r: &mut R) -> Stagnation {
        match r.gen_range(0..1) {
            0 => Stagnation::None,
            _ => Stagnation::NumGenerations(r.gen_range(1..1000)),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum Duplicates {
    DisallowDuplicates, // Don't allow duplicate genomes in the population.
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

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Cfg {
    pub pop_size: usize,
    pub crossover: Crossover,
    pub mutation: Mutation, // Mutation rate per bit / basic block.
    pub survival: Survival,
    pub selection: Selection,
    pub niching: Niching,
    pub species: Species,
    pub stagnation: Stagnation,
    pub duplicates: Duplicates,
    pub par_fitness: bool, // Run fitness computations in parallel
    pub par_dist: bool,    // Run distance computations in parallel
}

impl Cfg {
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
            duplicates: Duplicates::DisallowDuplicates,
            par_fitness: false,
            par_dist: false,
        }
    }

    pub fn with_pop_size(self, pop_size: usize) -> Self {
        Self { pop_size, ..self }
    }

    pub fn with_crossover(self, crossover: Crossover) -> Self {
        Self { crossover, ..self }
    }

    pub fn with_mutation(self, mutation: Mutation) -> Self {
        Self { mutation, ..self }
    }

    pub fn with_survival(self, survival: Survival) -> Self {
        Self { survival, ..self }
    }

    pub fn with_selection(self, selection: Selection) -> Self {
        Self { selection, ..self }
    }

    pub fn with_niching(self, niching: Niching) -> Self {
        Self { niching, ..self }
    }

    pub fn with_species(self, species: Species) -> Self {
        Self { species, ..self }
    }

    pub fn with_stagnation(self, stagnation: Stagnation) -> Self {
        Self { stagnation, ..self }
    }

    pub fn with_duplicates(self, duplicates: Duplicates) -> Self {
        Self { duplicates, ..self }
    }

    pub fn with_par_fitness(self, par_fitness: bool) -> Self {
        Self {
            par_fitness,
            ..self
        }
    }

    pub fn with_par_dist(self, par_dist: bool) -> Self {
        Self { par_dist, ..self }
    }
}
