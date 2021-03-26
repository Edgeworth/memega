use crate::{Evaluator, Genome, State};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::collections::HashSet;
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
            self.n = s.len();
            self.cache = if par {
                (0..self.n * self.n)
                    .into_par_iter()
                    .map(|v| {
                        let i = v / self.n;
                        let j = v % self.n;
                        eval.distance(&s[i].genome, &s[j].genome)
                    })
                    .collect()
            } else {
                let mut cache = vec![0.0; self.n * self.n];
                for i in 0..self.n {
                    for j in 0..self.n {
                        cache[i * self.n + j] = eval.distance(&s[i].genome, &s[j].genome);
                    }
                }
                cache
            };
        }
    }

    // TODO: Use species representatives - ones with the highest fitness, and
    // maybe re-speciate other ones.
    pub fn speciate<G: Genome>(&self, s: &[State<G>], radius: f64) -> SpeciesInfo {
        // Copy any existing species over.
        let mut ids: Vec<u64> = s.iter().map(|v| v.species).collect();


        // Split assigned and unassigned individuals.
        let mut assigned: HashSet<usize> = HashSet::new();
        let mut unassigned: HashSet<usize> = HashSet::new();
        for (i, &id) in ids.iter().enumerate() {
            if id == NO_SPECIES {
                unassigned.insert(i);
            } else {
                assigned.insert(i);
            }
        }

        let mut sorted_ids = ids.clone();
        sorted_ids.sort_unstable();
        sorted_ids.dedup();
        let mut num = sorted_ids.len() as u64;
        let mut next_id = sorted_ids.last().copied().unwrap_or(NO_SPECIES) + 1;

        while !unassigned.is_empty() {
            // Try to assign to existing species.
            while !unassigned.is_empty() {
                // Assign to existing species:
                let mut remaining: HashSet<usize> = HashSet::new();
                for &cand_idx in unassigned.iter() {
                    let mut cand_species = NO_SPECIES;
                    for &rep_idx in assigned.iter() {
                        if self[(rep_idx, cand_idx)] <= radius {
                            cand_species = ids[rep_idx];
                            break;
                        }
                    }
                    if cand_species == NO_SPECIES {
                        remaining.insert(cand_idx);
                    } else {
                        ids[cand_idx] = cand_species;
                    }
                }
                // Could not assign any more existing species.
                if unassigned == remaining {
                    break;
                }
                unassigned = remaining;
            }
            // We tried to match everything in |assigned|, now throw them away.
            assigned.clear();

            // Create new species for the next remaining unassigned individual.
            if let Some(&new_idx) = unassigned.iter().next() {
                ids[new_idx] = next_id;
                unassigned.remove(&new_idx);
                assigned.insert(new_idx);
                num += 1;
                next_id += 1;
            }
        }


        // Assign species to ones not assigned yet.
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
