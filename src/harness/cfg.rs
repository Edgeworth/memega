#[derive(Debug, Copy, Clone, Eq, PartialEq, PartialOrd)]
pub enum Termination {
    FixedGenerations(usize), // After fixed number of generations.
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd)]
pub struct HarnessCfg {
    termination: Termination,
    print_gen: Option<usize>, // How often to print basic generation info.
    print_summary: Option<usize>, // How often to print summary info.
}

impl HarnessCfg {
    #[must_use]
    pub fn new() -> Self {
        Self {
            termination: Termination::FixedGenerations(2000),
            print_gen: None,
            print_summary: None,
        }
    }

    #[must_use]
    pub fn termination(&self) -> Termination {
        self.termination
    }

    #[must_use]
    pub fn set_termination(mut self, termination: Termination) -> Self {
        self.termination = termination;
        self
    }

    #[must_use]
    pub fn print_gen(&self) -> Option<usize> {
        self.print_gen
    }

    #[must_use]
    pub fn set_print_gen(mut self, print_gen: usize) -> Self {
        self.print_gen = Some(print_gen);
        self
    }

    #[must_use]
    pub fn print_summary(&self) -> Option<usize> {
        self.print_summary
    }

    #[must_use]
    pub fn set_print_summary(mut self, print_summary: usize) -> Self {
        self.print_summary = Some(print_summary);
        self
    }
}

impl Default for HarnessCfg {
    fn default() -> Self {
        Self::new()
    }
}
