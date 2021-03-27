use crate::{Evaluator, Genome, Mem};
use derive_more::Display;
use float_pretty_print::PrettyPrintFloat;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::collections::VecDeque;
use std::ops::Index;

pub type SpeciesId = u64;
pub const NO_SPECIES: SpeciesId = 0;

#[derive(Copy, Clone, PartialOrd, PartialEq, Debug, Display)]
#[display(fmt = "{}, radius: {}", num, "PrettyPrintFloat(*radius)")]
pub struct SpeciesInfo {
    pub num: u64,
    pub radius: f64,
}

impl SpeciesInfo {
    pub fn new() -> Self {
        Self {
            num: 1,
            radius: 1.0,
        }
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
    pub fn new() -> Self {
        Self {
            n: 0,
            cache: Vec::new(),
            max: 0.0,
            sum: 0.0,
        }
    }

    pub fn ensure<E: Evaluator>(&mut self, s: &[Mem<E::Genome>], par: bool, eval: &E) {
        if self.is_empty() {
            self.n = s.len();
            self.cache = if par {
                let cache: Vec<f64> = (0..self.n * self.n)
                    .into_par_iter()
                    .map(|v| {
                        let i = v / self.n;
                        let j = v % self.n;
                        eval.distance(&s[i].genome, &s[j].genome)
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
                        let dist = eval.distance(&s[i].genome, &s[j].genome);
                        cache[i * self.n + j] = dist;
                        self.max = self.max.max(dist);
                        self.sum += dist;
                    }
                }
                cache
            };
        }
    }

    pub fn speciate<G: Genome>(&self, s: &[Mem<G>], radius: f64) -> (Vec<SpeciesId>, SpeciesInfo) {
        // Copy any existing species over.
        assert!(
            s.is_sorted_by_key(|v| -v.base_fitness),
            "Must be sorted by fitness (bug)"
        );
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

    pub fn shared_fitness<G: Genome>(&self, s: &mut [Mem<G>], radius: f64, alpha: f64) {
        // Compute fitness as F'(i) = F(i) / sum of 1 - (d(i, j) / species_radius) ^ alpha.
        for i in 0..s.len() {
            let mut sum = 0.0;
            for j in 0..s.len() {
                let d = self[(i, j)];
                if d < radius {
                    sum += 1.0 - (d / radius).powf(alpha)
                }
            }
            s[i].selection_fitness = s[i].base_fitness / sum;
        }
    }

    pub fn species_shared_fitness<G: Genome>(&self, s: &mut [Mem<G>], species: &SpeciesInfo) {
        // Compute alpha as: radius / num_species ^ (1 / dimensionality)
        let alpha = species.radius / species.num as f64;
        self.shared_fitness(s, species.radius, alpha)
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn mean(&self) -> f64 {
        self.sum / ((self.n * self.n) as f64)
    }

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
