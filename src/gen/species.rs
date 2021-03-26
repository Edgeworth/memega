use crate::{Evaluator, Genome, State};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::ops::Index;

pub const NO_SPECIES: u64 = 0;

#[derive(Clone, PartialOrd, PartialEq, Debug)]
pub struct SpeciesInfo {
    pub ids: Vec<u64>,
    pub num: u64,
    pub radius: f64,
}

impl SpeciesInfo {
    pub fn new(n: usize) -> Self {
        Self {
            ids: vec![NO_SPECIES; n],
            num: 1,
            radius: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct DistCache {
    n: usize,
    cache: Vec<f64>,
}

impl DistCache {
    pub fn new() -> Self {
        Self {
            n: 0,
            cache: Vec::new(),
        }
    }

    pub fn ensure<E: Evaluator>(&mut self, s: &[State<E::Genome>], par: bool, eval: &E) {
        if self.is_empty() {
            let n = s.len();
            self.cache = if par {
                (0..n * n)
                    .into_par_iter()
                    .map(|v| {
                        let i = v / n;
                        let j = v % n;
                        eval.distance(&s[i].genome, &s[j].genome)
                    })
                    .collect()
            } else {
                let mut cache = vec![0.0; n * n];
                for i in 0..n {
                    for j in 0..n {
                        cache[i * n + j] = eval.distance(&s[i].genome, &s[j].genome);
                    }
                }
                cache
            };
        }
    }

    pub fn speciate<G: Genome>(&self, s: &[State<G>], radius: f64) -> SpeciesInfo {
        // Copy any existing species over.
        let mut ids: Vec<u64> = s.iter().map(|v| v.species).collect();
        let mut sorted_ids = ids.clone();
        sorted_ids.sort_unstable();
        sorted_ids.dedup();
        let mut num = sorted_ids.len() as u64;
        let mut next_id = sorted_ids.last().copied().unwrap_or(NO_SPECIES) + 1;
        for i in 0..s.len() {
            if ids[i] != NO_SPECIES {
                continue;
            }
            ids[i] = next_id;
            for j in (i + 1)..s.len() {
                if self[(i, j)] <= radius {
                    ids[j] = next_id;
                }
            }
            num += 1;
            next_id += 1;
        }
        SpeciesInfo { ids, num, radius }
    }

    pub fn shared_fitness(&self, base_fitness: &[f64], species: &SpeciesInfo) -> Vec<f64> {
        // Compute alpha as: radius / num_species ^ (1 / dimensionality)
        let alpha = species.radius / species.num as f64;
        let mut fitness = base_fitness.to_vec();

        // Compute fitness as F'(i) = F(i) / sum of 1 - (d(i, j) / species_radius) ^ alpha.
        for i in 0..species.ids.len() {
            let mut sum = 0.0;
            for j in 0..species.ids.len() {
                let d = self[(i, j)];
                if d < species.radius {
                    sum += 1.0 - (d / species.radius).powf(alpha)
                }
            }
            fitness[i] /= sum;
        }
        fitness
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn mean(&self) -> f64 {
        self.cache.iter().sum::<f64>() / ((self.n * self.n) as f64)
    }

    pub fn max(&self) -> f64 {
        self.cache.iter().fold(0.0, |a: f64, &b| a.max(b))
    }
}

impl Index<(usize, usize)> for DistCache {
    type Output = f64;

    fn index(&self, i: (usize, usize)) -> &f64 {
        &self.cache[i.0 * self.n + i.1]
    }
}
