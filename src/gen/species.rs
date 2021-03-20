use crate::{Evaluator, State};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::ops::Index;

#[derive(Debug, Clone, PartialOrd, PartialEq)]
pub struct DistCache {
    n: usize,
    cache: Vec<f64>,
}

impl DistCache {
    pub fn new<E: Evaluator>(eval: &E, s: &[State<E::Genome>], par: bool) -> Self {
        let n = s.len();
        let cache = if par {
            (0..n * n)
                .into_par_iter()
                .map(|v| {
                    let i = v / n;
                    let j = v % n;
                    eval.distance(&s[i].0, &s[j].0)
                })
                .collect()
        } else {
            let mut cache = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    cache[i * n + j] = eval.distance(&s[i].0, &s[j].0);
                }
            }
            cache
        };
        Self { n, cache }
    }

    pub fn empty() -> Self {
        Self {
            n: 0,
            cache: vec![],
        }
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
