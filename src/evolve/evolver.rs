use std::fmt::Write;

use approx::{abs_diff_eq, relative_eq};
use eyre::Result;
use textwrap::indent;

use crate::eval::{Evaluator, State};
use crate::evolve::cfg::{Crossover, EvolveCfg, Mutation, Stagnation, StagnationCondition};
use crate::evolve::result::{EvolveResult, Stats};
use crate::genr::member::Member;
use crate::genr::unevaluated::UnevaluatedGenr;
use crate::ops::util::rand_vec;

pub trait CreateEvolverFn<E: Evaluator> =
    Fn(EvolveCfg) -> Evolver<E> + Sync + Send + Clone + 'static;
pub trait RandState<S: State> = FnMut() -> S + Send;

/// Runs iterations of GA w.r.t. the given evaluator.
#[must_use]
pub struct Evolver<E: Evaluator> {
    cfg: EvolveCfg,
    eval: E,
    genr: UnevaluatedGenr<E::State>,
    rand_state: Box<dyn RandState<E::State>>,
    gen_count: usize,
    stagnation_count: usize,
    last_fitness: f64,
}

/// Default runner for no data.
impl<E: Evaluator<Data = ()>> Evolver<E> {
    pub fn run(&mut self) -> Result<EvolveResult<E::State>> {
        self.run_data(&[()])
    }
}

impl<E: Evaluator> Evolver<E> {
    pub fn from_initial(
        eval: E,
        cfg: EvolveCfg,
        mut genr: Vec<E::State>,
        mut rand_state: impl RandState<E::State> + 'static,
    ) -> Self {
        // Fill out the rest of |genr| if it's smaller than pop_size.
        // If speciation is on, this lets more random species be generated at
        // the beginning.
        while genr.len() < cfg.pop_size {
            genr.push(rand_state());
        }
        let genr = UnevaluatedGenr::initial::<E>(genr, &cfg);
        Self {
            cfg,
            eval,
            genr,
            rand_state: Box::new(rand_state),
            gen_count: 0,
            stagnation_count: 0,
            last_fitness: 0.0,
        }
    }

    pub fn new(
        eval: E,
        cfg: EvolveCfg,
        mut rand_state: impl RandState<E::State> + 'static,
    ) -> Self {
        #[allow(clippy::redundant_closure)] // This closure is actually necessary.
        let genr = UnevaluatedGenr::initial::<E>(rand_vec(cfg.pop_size, || rand_state()), &cfg);
        Self {
            eval,
            cfg,
            genr,
            rand_state: Box::new(rand_state),
            gen_count: 0,
            stagnation_count: 0,
            last_fitness: 0.0,
        }
    }

    pub fn run_data(&mut self, inputs: &[E::Data]) -> Result<EvolveResult<E::State>> {
        let genr = self.genr.evaluate(inputs, &self.cfg, &self.eval)?;
        let stagnant = match self.cfg.stagnation_condition {
            StagnationCondition::Default => {
                relative_eq!(genr.mems[0].fitness, self.last_fitness)
            }
            StagnationCondition::Epsilon(ep) => {
                abs_diff_eq!(genr.mems[0].fitness, self.last_fitness, epsilon = ep)
            }
        };
        self.gen_count += 1;
        if stagnant {
            self.stagnation_count += 1;
        } else {
            self.stagnation_count = 0;
        }
        self.last_fitness = genr.mems[0].fitness;

        let stagnant = match self.cfg.stagnation {
            Stagnation::None => false,
            Stagnation::OneShotAfter(count) => {
                if self.stagnation_count >= count {
                    self.stagnation_count = 0;
                    true
                } else {
                    false
                }
            }
            Stagnation::ContinuousAfter(count) => self.stagnation_count >= count,
        };

        let mut next = genr.next_gen(self.rand_state.as_mut(), stagnant, &self.cfg, &self.eval)?;
        std::mem::swap(&mut next, &mut self.genr);
        Ok(EvolveResult { unevaluated: next, genr, stagnant })
    }

    pub fn cfg(&self) -> &EvolveCfg {
        &self.cfg
    }

    pub fn eval(&self) -> &E {
        &self.eval
    }

    pub fn summary(&self, r: &mut EvolveResult<E::State>) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "{}", Stats::from_result(r));
        if self.cfg.mutation == Mutation::Adaptive {
            s += "mutation:  ";
            for &v in &r.nth(0).params.mutation {
                let _ = write!(s, "{v:5.5}, ");
            }
            s += "\n";
        }
        if self.cfg.crossover == Crossover::Adaptive {
            s += "crossover: ";
            for &v in &r.nth(0).params.crossover {
                let _ = write!(s, "{v:5.5}, ");
            }
            s += "\n";
        }
        s
    }

    // Prints the top #n individuals. If there are multiple species, prints the
    // top n / # species for each species. If n isn't divisble by number of
    // species, the remainder will go to print the top n % # out of the #
    // species.
    #[allow(clippy::unused_self)]
    pub fn summary_sample(&self, r: &mut EvolveResult<E::State>, n: usize) -> String {
        let mut s = String::new();
        let species = r.genr.species();
        let mut by_species: Vec<(usize, Vec<Member<E::State>>)> = Vec::new();
        for &id in &species {
            by_species.push((0, r.genr.species_mems(id)));
        }

        let mut processed = 0;
        while processed < n {
            // What we added this round.
            let mut added: Vec<(f64, usize)> = Vec::new();
            for (idx, (pointer, v)) in by_species.iter_mut().enumerate() {
                // Try adding this one.
                if *pointer < v.len() {
                    added.push((v[*pointer].fitness, idx));
                    *pointer += 1;
                    processed += 1;
                }
            }
            if added.is_empty() {
                break;
            }
            if processed > n {
                // Remove |overflow| weakest individuals.
                let overflow = processed - n;
                added.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                for &(_, species) in added.iter().take(overflow) {
                    by_species[species].0 -= 1;
                }
            }
        }

        // Order species by highest fitness individual.
        by_species.sort_unstable_by(|a, b| {
            b.1.first().unwrap().fitness.partial_cmp(&a.1.first().unwrap().fitness).unwrap()
        });

        for (count, mems) in &by_species {
            if *count > 0 {
                let _ = writeln!(s, "Species {} top {count}:", mems[0].species);
                for mem in mems.iter().take(*count) {
                    let state_str = indent(&format!("{}", mem.state), "  ");
                    let _ = writeln!(s, "fitness: {:5.5}\n{state_str}", mem.fitness);
                }
                s += "\n";
            }
        }
        s.truncate(s.trim_end().len());
        s
    }
}
