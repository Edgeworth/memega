use derive_more::Display;
use float_pretty_print::PrettyPrintFloat;

use crate::eval::{Genome, Mem};
use crate::gen::evaluated::EvaluatedGen;
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
    pub fn from_run<G: Genome>(r: &mut RunResult<G>) -> Self {
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
pub struct RunResult<G: Genome> {
    pub unevaluated: UnevaluatedGen<G>,
    pub gen: EvaluatedGen<G>,
    pub stagnant: bool,
}

impl<G: Genome> RunResult<G> {
    #[must_use]
    pub fn size(&self) -> usize {
        self.gen.mems.len()
    }

    #[must_use]
    pub fn nth(&self, n: usize) -> &Mem<G> {
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
        let mut mems_copy = self.gen.mems.iter().map(|v| &v.genome).cloned().collect::<Vec<_>>();
        mems_copy.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        mems_copy.dedup();
        self.gen.mems.len() - mems_copy.len()
    }
}
