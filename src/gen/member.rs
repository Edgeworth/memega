use derive_more::Display;

use crate::eval::{Evaluator, State};
use crate::evolve::cfg::EvolveCfg;
use crate::gen::params::Params;
use crate::gen::species::{SpeciesId, NO_SPECIES};

#[must_use]
#[derive(Clone, PartialOrd, PartialEq, Debug, Display)]
#[display(fmt = "fitness {fitness:5.5} species {species:>3}")]
pub struct Member<S: State> {
    pub state: S,               // Actual state.
    pub params: Params,         // Adaptively evolved parameters
    pub species: SpeciesId,     // Species index
    pub fitness: f64,           // Original fitness, generated by Evaluator fitness function.
    pub selection_fitness: f64, // Potentially adjusted fitness, for selection.
    pub age: usize,             // Age of the member in generations.
}

impl<S: State> Member<S> {
    pub fn new<E: Evaluator>(state: S, cfg: &EvolveCfg) -> Self {
        Self {
            state,
            params: Params::new::<E>(cfg),
            species: NO_SPECIES,
            fitness: 0.0,
            selection_fitness: 0.0,
            age: 0,
        }
    }
}
