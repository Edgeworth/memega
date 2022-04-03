use derive_more::Display;
use float_pretty_print::PrettyPrintFloat;

use crate::cfg::Cfg;
use crate::eval::{Evaluator, State};
use crate::gen::params::Params;
use crate::gen::species::{SpeciesId, NO_SPECIES};

#[derive(Clone, PartialOrd, PartialEq, Debug, Display)]
#[display(fmt = "fitness {} species {}", "PrettyPrintFloat(*base_fitness)", species)]
pub struct Member<S: State> {
    pub state: S,               // Actual state.
    pub params: Params,         // Adaptively evolved parameters
    pub species: SpeciesId,     // Species index
    pub base_fitness: f64,      // Original fitness, generated by Evaluator fitness function.
    pub selection_fitness: f64, // Potentially adjusted fitness, for selection.
}

impl<S: State> Member<S> {
    pub fn new<E: Evaluator>(state: S, cfg: &Cfg) -> Self {
        Self {
            state,
            params: Params::new::<E>(cfg),
            species: NO_SPECIES,
            base_fitness: 0.0,
            selection_fitness: 0.0,
        }
    }
}
