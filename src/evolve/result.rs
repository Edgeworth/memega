use derive_more::Display;

use crate::eval::State;
use crate::genr::evaluated::EvaluatedGen;
use crate::genr::member::Member;
use crate::genr::species::SpeciesInfo;
use crate::genr::unevaluated::UnevaluatedGenr;

#[must_use]
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Stats {
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub pop_size: usize,
    pub num_dup: usize,
    pub mean_distance: f64,
    pub stagnant: bool,
    pub species: SpeciesInfo,
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "best: {:5.5}, mean: {:5.5}\npop: {:>5}, dupes: {:>5}, stagnant: {}",
            self.best_fitness, self.mean_fitness, self.pop_size, self.num_dup, self.stagnant
        )?;
        if self.mean_distance.is_finite() {
            write!(f, "dist: {:5.5}, {}", self.mean_distance, self.species)?;
        }
        Ok(())
    }
}

impl Stats {
    pub fn from_result<S: State>(r: &mut EvolveResult<S>) -> Self {
        Self {
            best_fitness: r.nth(0).fitness,
            mean_fitness: r.mean_fitness(),
            pop_size: r.size(),
            num_dup: r.num_dup(),
            mean_distance: r.mean_distance(),
            stagnant: r.stagnant,
            species: r.unevaluated.species,
        }
    }
}

#[must_use]
#[derive(Display, Clone, PartialEq)]
#[display("Run({genr})")]
pub struct EvolveResult<S: State> {
    pub unevaluated: UnevaluatedGenr<S>,
    pub genr: EvaluatedGen<S>,
    pub stagnant: bool,
}

impl<S: State> EvolveResult<S> {
    #[must_use]
    pub fn size(&self) -> usize {
        self.genr.mems.len()
    }

    pub fn nth(&self, n: usize) -> &Member<S> {
        &self.genr.mems[n]
    }

    #[must_use]
    pub fn mean_fitness(&self) -> f64 {
        self.genr.mems.iter().map(|v| v.fitness).sum::<f64>() / self.genr.mems.len() as f64
    }

    #[must_use]
    pub fn mean_distance(&self) -> f64 {
        self.unevaluated.dists.mean()
    }

    #[must_use]
    pub fn num_dup(&self) -> usize {
        let mut states = self.genr.mems.iter().map(|v| &v.state).cloned().collect::<Vec<_>>();
        states.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        states.dedup();
        self.genr.mems.len() - states.len()
    }
}
