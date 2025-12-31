use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

#[must_use]
#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum Termination {
    FixedGenerations(NonZeroUsize), // After fixed number of generations.
}

#[must_use]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd)]
pub struct TrainerCfg {
    pub name: String,
    pub termination: Termination,
    pub print_gen: Option<NonZeroUsize>, // How often to print basic generation info.
    pub print_summary: Option<NonZeroUsize>, // How often to print summary info.
    pub print_samples: Option<NonZeroUsize>, // How often to print samples.
    pub print_valid: Option<NonZeroUsize>, // How often to print validation info.
    pub report_gen: Option<NonZeroUsize>, // How often to report generation info via tensorboard.
    pub report_path: Option<PathBuf>,    // Where to write tensorboard reports.
}

impl TrainerCfg {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            termination: Termination::FixedGenerations(NonZeroUsize::new(2000).unwrap()),
            print_gen: None,
            print_summary: None,
            print_samples: None,
            print_valid: None,
            report_gen: None,
            report_path: None,
        }
    }

    pub fn set_termination(mut self, termination: Termination) -> Self {
        self.termination = termination;
        self
    }

    pub fn set_print_gen(mut self, print_gen: NonZeroUsize) -> Self {
        self.print_gen = Some(print_gen);
        self
    }

    pub fn set_print_summary(mut self, print_summary: NonZeroUsize) -> Self {
        self.print_summary = Some(print_summary);
        self
    }

    pub fn set_print_samples(mut self, print_samples: NonZeroUsize) -> Self {
        self.print_samples = Some(print_samples);
        self
    }

    pub fn set_print_valid(mut self, print_valid: NonZeroUsize) -> Self {
        self.print_valid = Some(print_valid);
        self
    }

    pub fn set_report_gen(mut self, report_gen: NonZeroUsize) -> Self {
        self.report_gen = Some(report_gen);
        self
    }

    pub fn set_report_path(mut self, report_path: impl AsRef<Path>) -> Self {
        self.report_path = Some(report_path.as_ref().into());
        self
    }
}
