use eyre::Result;
use log::warn;
use tempfile::{tempdir, TempDir};
use tensorboard_rs::summary_writer::SummaryWriter;

use crate::eval::Evaluator;
use crate::evolve::evolver::Evolver;
use crate::evolve::result::EvolveResult;
use crate::train::cfg::{Termination, TrainerCfg};

/// Runs evolution with the given parameters and prints some info.
pub struct Trainer {
    cfg: TrainerCfg,
    writer: Option<SummaryWriter>,
    _tmp: Option<TempDir>,
}

impl Trainer {
    #[must_use]
    pub fn new(cfg: TrainerCfg) -> Self {
        let (report_path, tmp) = if let Some(report_path) = &cfg.report_path {
            (Some(report_path.clone()), None)
        } else if cfg.report_gen.is_some() {
            let tmp = tempdir().unwrap();
            warn!("No report path specified for tensorboard, writing to {}", tmp.path().display());
            (Some(tmp.path().to_path_buf()), Some(tmp))
        } else {
            (None, None)
        };
        let writer = report_path.as_ref().map(SummaryWriter::new);
        Self { cfg, writer, _tmp: tmp }
    }

    pub fn evolve<E: Evaluator>(
        &mut self,
        mut evolver: Evolver<E>,
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
            let mut r = evolver.run_iter()?;

            fitness_sum += r.nth(0).fitness;
            fitness_count += 1.0;

            if let Some(print_gen) = self.cfg.print_gen && i % print_gen == 0 {
                println!("Generation {}: {}", i, r.nth(0).fitness);
            }
            if let Some(print_summary) = self.cfg.print_summary && i % print_summary == 0 {
                println!("{}", evolver.summary(&mut r));
                println!("{}", evolver.summary_sample(&mut r, 5));
            }
            if let Some(report_gen) = self.cfg.report_gen && let Some(writer) = &mut self.writer && i % report_gen == 0  {
                writer.add_scalar("train fitness", (fitness_sum / fitness_count) as f32, i);

            }
            ret = Some(r);
        }
        Ok(ret.unwrap())
    }
}
