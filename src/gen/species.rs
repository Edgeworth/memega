use std::collections::VecDeque;
use std::ops::Index;

use derive_more::Display;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::eval::{Evaluator, State};
use crate::gen::member::Member;

pub type SpeciesId = u64;
pub const NO_SPECIES: SpeciesId = 0;

#[derive(Copy, Clone, PartialOrd, PartialEq, Debug, Display)]
#[display(fmt = "species: {:>3}, radius: {:5.5}", num, radius)]
pub struct SpeciesInfo {
    pub num: u64,
    pub radius: f64,
}

impl SpeciesInfo {
    #[must_use]
    pub fn new() -> Self {
        Self { num: 1, radius: 1.0 }
    }
}

impl Default for SpeciesInfo {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct DistCache {
    n: usize,
    cache: Vec<f64>,
    max: f64,
    sum: f64,
}

impl DistCache {
    #[must_use]
    pub fn new() -> Self {
        Self { n: 0, cache: Vec::new(), max: 0.0, sum: 0.0 }
    }

    pub fn ensure<E: Evaluator>(&mut self, s: &[Member<E::State>], par: bool, eval: &E) {
        if self.is_empty() {
            self.n = s.len();
            self.cache = if par {
                let cache: Vec<f64> = (0..self.n * self.n)
                    .into_par_iter()
                    .map(|v| {
                        let i = v / self.n;
                        let j = v % self.n;
                        eval.distance(&s[i].state, &s[j].state)
                    })
                    .collect();
                (self.max, self.sum) = cache
                    .par_iter()
                    .fold(|| (0.0, 0.0), |(m, s): (f64, f64), &v| (m.max(v), s + v))
                    .reduce(|| (0.0, 0.0), |(m0, s0), (m1, s1)| (m0.max(m1), s0 + s1));
                cache
            } else {
                let mut cache = vec![0.0; self.n * self.n];
                for i in 0..self.n {
                    for j in 0..self.n {
                        let dist = eval.distance(&s[i].state, &s[j].state);
                        cache[i * self.n + j] = dist;
                        self.max = self.max.max(dist);
                        self.sum += dist;
                    }
                }
                cache
            };
        }
    }

    pub fn speciate<S: State>(
        &self,
        s: &[Member<S>],
        radius: f64,
    ) -> (Vec<SpeciesId>, SpeciesInfo) {
        // Copy any existing species over.
        assert!(s.is_sorted_by_key(|v| -v.fitness), "Must be sorted by fitness (bug)");
        let mut ids: Vec<SpeciesId> = vec![NO_SPECIES; s.len()];
        let mut unassigned: VecDeque<usize> = (0..s.len()).collect();
        let mut num = 1;
        while !unassigned.is_empty() {
            // Take next highest fitness to define the next species.
            let next = unassigned.pop_front().unwrap();
            ids[next] = num;

            unassigned.retain(|&v| {
                if self[(next, v)] <= radius {
                    ids[v] = num;
                    false
                } else {
                    true
                }
            });
            num += 1;
        }

        // Assign species to ones not assigned yet.
        (ids, SpeciesInfo { num, radius })
    }

    pub fn shared_fitness<S: State>(&self, s: &mut [Member<S>], radius: f64, alpha: f64) {
        // Compute fitness as F'(i) = F(i) / sum of 1 - (d(i, j) / species_radius) ^ alpha.
        for i in 0..s.len() {
            let mut sum = 0.0;
            for j in 0..s.len() {
                let d = self[(i, j)];
                if d < radius {
                    sum += 1.0 - (d / radius).powf(alpha);
                }
            }
            s[i].selection_fitness = s[i].fitness / sum;
        }
    }

    pub fn species_shared_fitness<S: State>(&self, s: &mut [Member<S>], species: &SpeciesInfo) {
        // Compute alpha as: radius / num_species ^ (1 / dimensionality)
        let alpha = species.radius / species.num as f64;
        self.shared_fitness(s, species.radius, alpha);
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    #[must_use]
    pub fn mean(&self) -> f64 {
        self.sum / ((self.n * self.n) as f64)
    }

    #[must_use]
    pub fn max(&self) -> f64 {
        self.max
    }
}

impl Default for DistCache {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<(usize, usize)> for DistCache {
    type Output = f64;

    fn index(&self, i: (usize, usize)) -> &f64 {
        &self.cache[i.0 * self.n + i.1]
    }
}
