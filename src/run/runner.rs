use approx::{abs_diff_eq, relative_eq};
use eyre::Result;
use float_pretty_print::PrettyPrintFloat;

use crate::cfg::{Cfg, Crossover, Mutation, Stagnation, StagnationCondition};
use crate::eval::{Evaluator, Genome, Mem};
use crate::gen::unevaluated::UnevaluatedGen;
use crate::ops::util::rand_vec;
use crate::run::result::{RunResult, Stats};

pub trait CreateRunnerFn<E: Evaluator> = Fn(Cfg) -> Runner<E> + Sync + Send + Clone + 'static;
pub trait RandGenome<G: Genome> = FnMut() -> G + Send;

/// Runs iterations of GA w.r.t. the given evaluator.
pub struct Runner<E: Evaluator> {
    pub cfg: Cfg,
    eval: E,
    gen: UnevaluatedGen<E::Genome>,
    rand_genome: Box<dyn RandGenome<E::Genome>>,
    gen_count: usize,
    stagnation_count: usize,
    last_fitness: f64,
}

impl<E: Evaluator> Runner<E> {
    pub fn from_initial(
        eval: E,
        cfg: Cfg,
        mut gen: Vec<E::Genome>,
        mut rand_genome: impl RandGenome<E::Genome> + 'static,
    ) -> Self {
        // Fill out the rest of |gen| if it's smaller than pop_size.
        // If speciation is on, this lets more random species be generated at
        // the beginning.
        while gen.len() < cfg.pop_size {
            gen.push(rand_genome());
        }
        let gen = UnevaluatedGen::initial::<E>(gen, &cfg);
        Self {
            cfg,
            eval,
            gen,
            rand_genome: Box::new(rand_genome),
            gen_count: 0,
            stagnation_count: 0,
            last_fitness: 0.0,
        }
    }

    pub fn new(eval: E, cfg: Cfg, mut rand_genome: impl RandGenome<E::Genome> + 'static) -> Self {
        #[allow(clippy::redundant_closure)] // This closure is actually necessary.
        let gen = UnevaluatedGen::initial::<E>(rand_vec(cfg.pop_size, || rand_genome()), &cfg);
        Self {
            eval,
            cfg,
            gen,
            rand_genome: Box::new(rand_genome),
            gen_count: 0,
            stagnation_count: 0,
            last_fitness: 0.0,
        }
    }

    pub fn run_iter(&mut self) -> Result<RunResult<E::Genome>> {
        let gen = self.gen.evaluate(self.gen_count, &self.cfg, &self.eval)?;
        let stagnant = match self.cfg.stagnation_condition {
            StagnationCondition::Default => {
                relative_eq!(gen.mems[0].base_fitness, self.last_fitness)
            }
            StagnationCondition::Epsilon(ep) => {
                abs_diff_eq!(gen.mems[0].base_fitness, self.last_fitness, epsilon = ep)
            }
        };
        self.gen_count += 1;
        if stagnant {
            self.stagnation_count += 1;
        } else {
            self.stagnation_count = 0;
        }
        self.last_fitness = gen.mems[0].base_fitness;

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

        let mut next = gen.next_gen(self.rand_genome.as_mut(), stagnant, &self.cfg, &self.eval)?;
        std::mem::swap(&mut next, &mut self.gen);
        Ok(RunResult { unevaluated: next, gen, stagnant })
    }

    pub fn cfg(&self) -> &Cfg {
        &self.cfg
    }

    pub fn eval(&self) -> &E {
        &self.eval
    }

    pub fn summary(&self, r: &mut RunResult<E::Genome>) -> String {
        let mut s = String::new();
        s += &format!("{}\n", Stats::from_run(r));
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
    pub fn summary_sample(
        &self,
        r: &mut RunResult<E::Genome>,
        n: usize,
        mut f: impl FnMut(&E::Genome) -> String,
    ) -> String {
        let mut s = String::new();
        let species = r.gen.species();
        let mut by_species: Vec<(usize, Vec<Mem<E::Genome>>)> = Vec::new();
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
                    added.push((v[*pointer].base_fitness, idx));
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
                .base_fitness
                .partial_cmp(&a.1.first().unwrap().base_fitness)
                .unwrap()
        });

        for (count, mems) in &by_species {
            if *count > 0 {
                s += &format!("Species {} top {}:\n", mems[0].species, count);
                for mem in mems.iter().take(*count) {
                    s += &format!("{}\n{}\n", PrettyPrintFloat(mem.base_fitness), f(&mem.genome));
                }
                s += "\n";
            }
        }
        s.truncate(s.trim_end().len());
        s
    }
}