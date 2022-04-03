use derive_more::Display;
use float_pretty_print::PrettyPrintFloat;

use crate::eval::State;
use crate::gen::evaluated::EvaluatedGen;
use crate::gen::member::Member;
use crate::gen::species::SpeciesInfo;
use crate::gen::unevaluated::UnevaluatedGen;

#[derive(Debug, Copy, Clone, PartialEq, Display)]
#[display(
    fmt = "best: {}, mean: {}, pop: {}, dupes: {}, dist: {}, stagnant: {}, species: {}",
    "PrettyPrintFloat(*best_fitness)",
    "PrettyPrintFloat(*mean_fitness)",
    pop_size,
    num_dup,
    "PrettyPrintFloat(*mean_distance)",
    stagnant,
    species
)]
pub struct Stats {
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub pop_size: usize,
    pub num_dup: usize,
    pub mean_distance: f64,
    pub stagnant: bool,
    pub species: SpeciesInfo,
}

impl Stats {
    pub fn from_result<S: State>(r: &mut EvolveResult<S>) -> Self {
        Self {
            best_fitness: r.nth(0).base_fitness,
            mean_fitness: r.mean_fitness(),
            pop_size: r.size(),
            num_dup: r.num_dup(),
            mean_distance: r.mean_distance(),
            stagnant: r.stagnant,
            species: r.unevaluated.species,
        }
    }
}

#[derive(Display, Clone, PartialEq)]
#[display(fmt = "Run({})", gen)]
pub struct EvolveResult<S: State> {
    pub unevaluated: UnevaluatedGen<S>,
    pub gen: EvaluatedGen<S>,
    pub stagnant: bool,
}

impl<S: State> EvolveResult<S> {
    #[must_use]
    pub fn size(&self) -> usize {
        self.gen.mems.len()
    }

    #[must_use]
    pub fn nth(&self, n: usize) -> &Member<S> {
        &self.gen.mems[n]
    }

    #[must_use]
    pub fn mean_fitness(&self) -> f64 {
        self.gen.mems.iter().map(|v| v.base_fitness).sum::<f64>() / self.gen.mems.len() as f64
    }

    #[must_use]
    pub fn mean_distance(&self) -> f64 {
        self.unevaluated.dists.mean()
    }

    #[must_use]
    pub fn num_dup(&self) -> usize {
        let mut states = self.gen.mems.iter().map(|v| &v.state).cloned().collect::<Vec<_>>();
        states.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        states.dedup();
        self.gen.mems.len() - states.len()
    }
}
