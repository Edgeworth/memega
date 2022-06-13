use std::collections::HashMap;

use eyre::Result;

use crate::eval::Evaluator;
use crate::evolve::evolver::Evolver;
use crate::evolve::result::EvolveResult;
use crate::train::cfg::{Termination, TrainerCfg};
use crate::train::sampler::DataSampler;

/// Runs evolution with the given parameters and prints some info.
#[must_use]
pub struct Trainer {
    cfg: TrainerCfg,
    #[cfg(feature = "tensorboard")]
    writer: Option<tensorboard_rs::summary_writer::SummaryWriter>,
}

impl Trainer {
    #[cfg(feature = "tensorboard")]
    fn new_tensorboard(cfg: TrainerCfg) -> Self {
        use std::mem;

        use chrono::Local;
        use log::warn;
        use tempfile::Builder;

        let report_path = if let Some(report_path) = &cfg.report_path {
            Some(report_path.clone())
        } else if cfg.report_gen.is_some() {
            let path = std::env::temp_dir().join("tensorboard");
            std::fs::create_dir_all(path.clone()).unwrap();
            let prefix = format!("{}-{}", cfg.name, Local::now().format("%Y-%m-%d-%H-%M-%S"));
            let tmp = Builder::new().prefix(&prefix).tempdir_in(path).unwrap();
            warn!("No report path specified for tensorboard, writing to {}", tmp.path().display());
            let path = Some(tmp.path().to_path_buf());
            mem::forget(tmp); // Don't delete the tempdir.
            path
        } else {
            None
        };
        let writer = report_path.as_ref().map(tensorboard_rs::summary_writer::SummaryWriter::new);
        Self { cfg, writer }
    }

    pub fn new(cfg: TrainerCfg) -> Self {
        #[cfg(feature = "tensorboard")]
        let s = Self::new_tensorboard(cfg);
        #[cfg(not(feature = "tensorboard"))]
        let s = Self { cfg };
        s
    }

    pub fn train<E: Evaluator>(
        &mut self,
        mut evolver: Evolver<E>,
        sampler: &impl DataSampler<E::Data>,
    ) -> Result<EvolveResult<E::State>> {
        let mut ret = None;
        let mut fitness_sum = 0.0;
        let mut fitness_count = 0.0;
        for i in 0.. {
            match self.cfg.termination {
                Termination::FixedGenerations(gen) => {
                    if i >= gen {
                        break;
                    }
                }
            }
            let mut r = evolver.run_data(&sampler.train(i))?;

            fitness_sum += r.nth(0).fitness;
            fitness_count += 1.0;

            if let Some(print_gen) = self.cfg.print_gen && i % print_gen == 0 {
                 println!("Gen {i:>6}\ntrain best {:5.5}", r.nth(0).fitness);
            }

            if let Some(print_valid) = self.cfg.print_valid && i % print_valid == 0 {
                let valid_fitness = evolver.eval().multi_fitness(
                    &r.nth(0).state,
                    &sampler.valid(i),
                    evolver.cfg().fitness_reduction,
                )?;
                println!("valid best: {valid_fitness:5.5}");
            }

            if let Some(print_summary) = self.cfg.print_summary && i % print_summary == 0 {
                println!("{}", evolver.summary(&mut r));
            }

            if let Some(print_samples) = self.cfg.print_samples && i % print_samples == 0 {
                println!("{}", evolver.summary_sample(&mut r, 5));
            }

            #[cfg(feature = "tensorboard")]
            if let Some(report_gen) = self.cfg.report_gen &&
                    let Some(writer) = &mut self.writer && i % report_gen == 0 {
                let valid_fitness = evolver.eval().multi_fitness(
                    &r.nth(0).state,
                    &sampler.valid(i),
                    evolver.cfg().fitness_reduction,
                )?;
                let scalars = HashMap::from([
                    ("train".to_string(), (fitness_sum / fitness_count) as f32),
                    ("valid".to_string(), valid_fitness as f32),
                ]);
                writer.add_scalars("fitness", &scalars, i);

                writer.flush();
                fitness_sum = 0.0;
                fitness_count = 0.0;
            }
            ret = Some(r);
        }
        Ok(ret.unwrap())
    }
}
