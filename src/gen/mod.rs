use rand::Rng;

use crate::cfg::{Cfg, Crossover, Mutation};
use crate::ops::util::rand_vec;
use crate::Evaluator;

pub mod evaluated;
mod species;
pub mod unevaluated;

// Potentially self-adaptive parameters per state.
#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct Params {
    // Conventionally, the first element will be the weight of doing no mutation or crossover.
    pub mutation: Vec<f64>,
    pub crossover: Vec<f64>,
}

impl Params {
    pub fn new<E: Evaluator>(cfg: &Cfg) -> Self {
        let mut r = rand::thread_rng();
        let mutation = if let Mutation::Fixed(v) = &cfg.mutation {
            v.clone()
        } else {
            rand_vec(E::NUM_MUTATION, || r.gen())
        };

        let crossover = if let Crossover::Fixed(v) = &cfg.crossover {
            v.clone()
        } else {
            rand_vec(E::NUM_CROSSOVER, || r.gen())
        };

        Self {
            mutation,
            crossover,
        }
    }
}
