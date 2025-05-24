use std::marker::PhantomData;

use rand::SeedableRng;
use rand::prelude::StdRng;
use rand::seq::IndexedRandom;

use crate::eval::Data;

pub trait DataSampler<D: Data> {
    fn train(&self, genr: usize) -> Vec<D>;
    fn valid(&self, genr: usize) -> Vec<D>;
    fn test(&self, genr: usize) -> Vec<D>;
}

#[must_use]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd)]
pub struct EmptyDataSampler {}

impl DataSampler<()> for EmptyDataSampler {
    fn train(&self, _: usize) -> Vec<()> {
        vec![()]
    }

    fn valid(&self, _: usize) -> Vec<()> {
        vec![()]
    }

    fn test(&self, _: usize) -> Vec<()> {
        vec![()]
    }
}

/// Wraps a given `DataSampler` and returns random subsets of the data for
/// each generation.
#[must_use]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd)]
pub struct BatchDataSampler<D: Data, S: DataSampler<D>> {
    sampler: S,
    batch_size: usize,
    _u: PhantomData<D>,
}

impl<D: Data, S: DataSampler<D>> BatchDataSampler<D, S> {
    pub fn new(sampler: S, batch_size: usize) -> Self {
        BatchDataSampler { sampler, batch_size, _u: PhantomData }
    }
}

impl<D: Data, S: DataSampler<D>> DataSampler<D> for BatchDataSampler<D, S> {
    fn train(&self, genr: usize) -> Vec<D> {
        // Randomly select batch_size samples from the training set,
        // randomly seeded based on the generation to keep it consistent.
        let mut r = StdRng::seed_from_u64(genr as u64);
        let v = self.sampler.train(genr);
        v.choose_multiple(&mut r, self.batch_size).cloned().collect()
    }

    fn valid(&self, genr: usize) -> Vec<D> {
        self.sampler.valid(genr)
    }

    fn test(&self, genr: usize) -> Vec<D> {
        self.sampler.test(genr)
    }
}
