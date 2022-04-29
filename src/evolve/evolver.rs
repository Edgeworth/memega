use approx::{abs_diff_eq, relative_eq};
use eyre::Result;
use float_pretty_print::PrettyPrintFloat;

use crate::eval::{Evaluator, State};
use crate::evolve::cfg::{Crossover, EvolveCfg, Mutation, Stagnation, StagnationCondition};
use crate::evolve::result::{EvolveResult, Stats};
use crate::gen::member::Member;
use crate::gen::unevaluated::UnevaluatedGen;
use crate::ops::util::rand_vec;

pub trait CreateEvolverFn<E: Evaluator> =
    Fn(EvolveCfg) -> Evolver<E> + Sync + Send + Clone + 'static;
pub trait RandState<S: State> = FnMut() -> S + Send;

/// Runs iterations of GA w.r.t. the given evaluator.
pub struct Evolver<E: Evaluator> {
    cfg: EvolveCfg,
    eval: E,
    gen: UnevaluatedGen<E::State>,
    rand_state: Box<dyn RandState<E::State>>,
    gen_count: usize,
    stagnation_count: usize,
    last_fitness: f64,
}

impl<E: Evaluator> Evolver<E> {
    pub fn from_initial(
        eval: E,
        cfg: EvolveCfg,
        mut gen: Vec<E::State>,
        mut rand_state: impl RandState<E::State> + 'static,
    ) -> Self {
        // Fill out the rest of |gen| if it's smaller than pop_size.
        // If speciation is on, this lets more random species be generated at
        // the beginning.
        while gen.len() < cfg.pop_size {
            gen.push(rand_state());
        }
        let gen = UnevaluatedGen::initial::<E>(gen, &cfg);
        Self {
            cfg,
            eval,
            gen,
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
        let gen = UnevaluatedGen::initial::<E>(rand_vec(cfg.pop_size, || rand_state()), &cfg);
        Self {
            eval,
            cfg,
            gen,
            rand_state: Box::new(rand_state),
            gen_count: 0,
            stagnation_count: 0,
            last_fitness: 0.0,
        }
    }

    pub fn run_iter(&mut self) -> Result<EvolveResult<E::State>> {
        let gen = self.gen.evaluate(self.gen_count, &self.cfg, &self.eval)?;
        let stagnant = match self.cfg.stagnation_condition {
            StagnationCondition::Default => {
                relative_eq!(gen.mems[0].fitness, self.last_fitness)
            }
            StagnationCondition::Epsilon(ep) => {
                abs_diff_eq!(gen.mems[0].fitness, self.last_fitness, epsilon = ep)
            }
        };
        self.gen_count += 1;
        if stagnant {
            self.stagnation_count += 1;
        } else {
            self.stagnation_count = 0;
        }
        self.last_fitness = gen.mems[0].fitness;

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

        let mut next = gen.next_gen(self.rand_state.as_mut(), stagnant, &self.cfg, &self.eval)?;
        std::mem::swap(&mut next, &mut self.gen);
        Ok(EvolveResult { unevaluated: next, gen, stagnant })
    }

    pub fn cfg(&self) -> &EvolveCfg {
        &self.cfg
    }

    pub fn eval(&self) -> &E {
        &self.eval
    }

    pub fn summary(&self, r: &mut EvolveResult<E::State>) -> String {
        let mut s = String::new();
        s += &format!("{}\n", Stats::from_result(r));
        if self.cfg.mutation == Mutation::Adaptive {
            s += "  mutation weights: ";
            for &v in &r.nth(0).params.mutation {
                s += &format!("{}, ", PrettyPrintFloat(v));
            }
            s += "\n";
        }
        if self.cfg.crossover == Crossover::Adaptive {
            s += "  crossover weights: ";
            for &v in &r.nth(0).params.crossover {
                s += &format!("{}, ", PrettyPrintFloat(v));
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
        let species = r.gen.species();
        let mut by_species: Vec<(usize, Vec<Member<E::State>>)> = Vec::new();
        for &id in &species {
            by_species.push((0, r.gen.species_mems(id)));
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
            b.1.first()
                .unwrap()
                .fitness
                .partial_cmp(&a.1.first().unwrap().fitness)
                .unwrap()
        });

        for (count, mems) in &by_species {
            if *count > 0 {
                s += &format!("Species {} top {}:\n", mems[0].species, count);
                for mem in mems.iter().take(*count) {
                    s += &format!("{}\n{}\n", PrettyPrintFloat(mem.fitness), mem.state);
                }
                s += "\n";
            }
        }
        s.truncate(s.trim_end().len());
        s
    }
}
