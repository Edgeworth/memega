use std::path::{Path, PathBuf};

#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum Termination {
    FixedGenerations(usize), // After fixed number of generations.
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd)]
pub struct TrainerCfg {
    pub termination: Termination,
    pub print_gen: Option<usize>, // How often to print basic generation info.
    pub print_summary: Option<usize>, // How often to print summary info.
    pub report_gen: Option<usize>, // How often to report generation info via tensorboard.
    pub report_path: Option<PathBuf>, // Where to write tensorboard reports.
}

impl TrainerCfg {
    #[must_use]
    pub fn new() -> Self {
        Self {
            termination: Termination::FixedGenerations(2000),
            print_gen: None,
            print_summary: None,
            report_gen: None,
            report_path: None,
        }
    }

    #[must_use]
    pub fn set_termination(mut self, termination: Termination) -> Self {
        self.termination = termination;
        self
    }

    #[must_use]
    pub fn set_print_gen(mut self, print_gen: usize) -> Self {
        self.print_gen = Some(print_gen);
        self
    }

    #[must_use]
    pub fn set_print_summary(mut self, print_summary: usize) -> Self {
        self.print_summary = Some(print_summary);
        self
    }

    #[must_use]
    pub fn set_report_gen(mut self, report_gen: usize) -> Self {
        self.report_gen = Some(report_gen);
        self
    }

    #[must_use]
    pub fn set_report_path(mut self, report_path: impl AsRef<Path>) -> Self {
        self.report_path = Some(report_path.as_ref().into());
        self
    }
}

impl Default for TrainerCfg {
    fn default() -> Self {
        Self::new()
    }
}
