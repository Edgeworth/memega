use std::collections::VecDeque;
use std::ops::Index;

use derive_more::Display;
use eyre::Result;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::eval::{Evaluator, State};
use crate::genr::member::Member;

pub type SpeciesId = u64;
pub const NO_SPECIES: SpeciesId = 0;

#[must_use]
#[derive(Copy, Clone, PartialOrd, PartialEq, Debug, Display)]
#[display("species: {num:>3}, radius: {radius:5.5}")]
pub struct SpeciesInfo {
    pub num: u64,
    pub radius: f64,
}

impl SpeciesInfo {
    pub fn new() -> Self {
        Self { num: 1, radius: 1.0 }
    }
}

impl Default for SpeciesInfo {
    fn default() -> Self {
        Self::new()
    }
}

#[must_use]
#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct DistCache {
    n: usize,
    cache: Vec<f64>,
    max: f64,
    sum: f64,
}

impl DistCache {
    pub fn new() -> Self {
        Self { n: 0, cache: Vec::new(), max: 0.0, sum: 0.0 }
    }

    pub fn ensure<E: Evaluator>(
        &mut self,
        s: &[Member<E::State>],
        par: bool,
        eval: &E,
    ) -> Result<()> {
        if self.is_empty() {
            self.n = s.len();
            self.cache = if par {
                let cache = (0..self.n * self.n)
                    .into_par_iter()
                    .map(|v| {
                        let i = v / self.n;
                        let j = v % self.n;
                        eval.distance(&s[i].state, &s[j].state)
                    })
                    .collect::<Result<Vec<f64>>>()?;
                (self.max, self.sum) = cache
                    .par_iter()
                    .fold(|| (0.0, 0.0), |(m, s): (f64, f64), &v| (m.max(v), s + v))
                    .reduce(|| (0.0, 0.0), |(m0, s0), (m1, s1)| (m0.max(m1), s0 + s1));
                cache
            } else {
                let mut cache = vec![0.0; self.n * self.n];
                for i in 0..self.n {
                    for j in 0..self.n {
                        let dist = eval.distance(&s[i].state, &s[j].state)?;
                        cache[i * self.n + j] = dist;
                        self.max = self.max.max(dist);
                        self.sum += dist;
                    }
                }
                cache
            };
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use eyre::Result;
    use pretty_assertions::assert_eq;

    #[derive(Debug, Clone, PartialEq, PartialOrd)]
    struct TestState(f64);

    impl std::fmt::Display for TestState {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    struct TestEvaluator;

    impl Evaluator for TestEvaluator {
        type State = TestState;

        fn crossover(&self, _s1: &mut Self::State, _s2: &mut Self::State, _idx: usize) {}
        fn mutate(&self, _s: &mut Self::State, _rate: f64, _idx: usize) {}
        fn fitness(&self, s: &Self::State, _data: &Self::Data) -> Result<f64> {
            Ok(s.0)
        }
        fn distance(&self, s1: &Self::State, s2: &Self::State) -> Result<f64> {
            Ok((s1.0 - s2.0).abs())
        }
    }

    fn make_members(values: &[f64]) -> Vec<Member<TestState>> {
        values.iter().map(|&v| Member {
            state: TestState(v),
            params: crate::genr::params::Params {
                mutation: vec![0.1],
                crossover: vec![0.5, 0.5],
            },
            species: NO_SPECIES,
            fitness: v,
            selection_fitness: v,
            age: 0,
        }).collect()
    }

    #[test]
    fn test_species_info_default() {
        let info = SpeciesInfo::new();
        assert_eq!(info.num, 1);
        assert_eq!(info.radius, 1.0);
    }

    #[test]
    fn test_dist_cache_new() {
        let cache = DistCache::new();
        assert!(cache.is_empty());
        // mean() is NaN when empty (division by zero)
        assert!(cache.mean().is_nan() || cache.mean() == 0.0);
        assert_eq!(cache.max(), 0.0);
    }

    #[test]
    fn test_dist_cache_ensure() -> Result<()> {
        let mut cache = DistCache::new();
        let members = make_members(&[1.0, 2.0, 3.0]);
        let eval = TestEvaluator;

        cache.ensure(&members, false, &eval)?;
        assert!(!cache.is_empty());
        assert_eq!(cache.n, 3);

        // Check some distances
        assert_eq!(cache[(0, 0)], 0.0); // Distance to self
        assert_eq!(cache[(0, 1)], 1.0); // |1-2| = 1
        assert_eq!(cache[(0, 2)], 2.0); // |1-3| = 2
        assert_eq!(cache[(1, 2)], 1.0); // |2-3| = 1

        Ok(())
    }

    #[test]
    fn test_dist_cache_mean() -> Result<()> {
        let mut cache = DistCache::new();
        let members = make_members(&[0.0, 10.0]);
        let eval = TestEvaluator;

        cache.ensure(&members, false, &eval)?;
        // Distances: (0,0)=0, (0,1)=10, (1,0)=10, (1,1)=0
        // Mean = (0 + 10 + 10 + 0) / 4 = 5.0
        assert_eq!(cache.mean(), 5.0);

        Ok(())
    }

    #[test]
    fn test_dist_cache_max() -> Result<()> {
        let mut cache = DistCache::new();
        let members = make_members(&[0.0, 5.0, 10.0]);
        let eval = TestEvaluator;

        cache.ensure(&members, false, &eval)?;
        assert_eq!(cache.max(), 10.0); // Max distance is |0-10| = 10

        Ok(())
    }

    #[test]
    fn test_speciate_single_species() {
        let mut cache = DistCache::new();
        let members = make_members(&[10.0, 9.5, 9.0]); // All close together
        let eval = TestEvaluator;

        cache.ensure(&members, false, &eval).unwrap();
        let (ids, info) = cache.speciate(&members, 5.0); // Large radius

        // All should be in same species
        assert!(ids.iter().all(|&id| id == 1));
        assert_eq!(info.num, 2); // num is incremented at the end
    }

    #[test]
    fn test_speciate_multiple_species() {
        let mut cache = DistCache::new();
        // Members must be sorted by fitness (descending)
        // Two clusters: (100, 98) and (15, 13)
        let members = make_members(&[100.0, 98.0, 15.0, 13.0]); // Two clusters, sorted
        let eval = TestEvaluator;

        cache.ensure(&members, false, &eval).unwrap();
        let (ids, info) = cache.speciate(&members, 3.0); // Radius of 3.0

        // Should have at least 2 species
        assert!(info.num >= 2);
        // First two should be same species (distance = 2.0 < 3.0)
        assert_eq!(ids[0], ids[1]);
        // Last two should be same species (distance = 2.0 < 3.0)
        assert_eq!(ids[2], ids[3]);
        // First and third should be different species (distance = 85.0 > 3.0)
        assert_ne!(ids[0], ids[2]);
    }

    #[test]
    fn test_shared_fitness() -> Result<()> {
        let mut cache = DistCache::new();
        let mut members = make_members(&[10.0, 10.0, 10.0]); // All same fitness
        let eval = TestEvaluator;

        cache.ensure(&members, false, &eval)?;
        cache.shared_fitness(&mut members, 2.0, 1.0);

        // All should have same selection fitness (shared with neighbors)
        assert!(members.iter().all(|m| m.selection_fitness > 0.0));
        assert!(members.iter().all(|m| m.selection_fitness < 10.0)); // Reduced by sharing

        Ok(())
    }

    #[test]
    fn test_species_shared_fitness() -> Result<()> {
        let mut cache = DistCache::new();
        let mut members = make_members(&[10.0, 10.0]);
        let eval = TestEvaluator;
        let species = SpeciesInfo { num: 2, radius: 5.0 };

        cache.ensure(&members, false, &eval)?;
        cache.species_shared_fitness(&mut members, &species);

        // Should have computed shared fitness
        assert!(members.iter().all(|m| m.selection_fitness > 0.0));

        Ok(())
    }

    #[test]
    fn test_dist_cache_parallel() -> Result<()> {
        let mut cache = DistCache::new();
        let members = make_members(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let eval = TestEvaluator;

        // Test parallel computation
        cache.ensure(&members, true, &eval)?;
        assert!(!cache.is_empty());
        assert_eq!(cache.n, 5);

        // Verify some distances
        assert_eq!(cache[(0, 4)], 4.0); // |1-5| = 4
        assert_eq!(cache[(2, 2)], 0.0); // Distance to self

        Ok(())
    }

    #[test]
    fn test_speciate_sorted_by_fitness() {
        let mut cache = DistCache::new();
        // Members should be sorted by fitness (highest first)
        let members = make_members(&[100.0, 50.0, 25.0]);
        let eval = TestEvaluator;

        cache.ensure(&members, false, &eval).unwrap();
        let (ids, _) = cache.speciate(&members, 30.0);

        // Highest fitness member should define first species
        assert_eq!(ids[0], 1);
    }
}
