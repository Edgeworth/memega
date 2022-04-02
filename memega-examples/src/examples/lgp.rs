use memega::cfg::Cfg;
use memega::eval::Evaluator;
use memega::evaluators::lgp::builder::lgp_fitness_evolver;
use memega::evaluators::lgp::cfg::LgpCfg;
use memega::evaluators::lgp::eval::State;
use memega::evaluators::lgp::vm::exec::LgpExec;
use memega::evolve::evolver::Evolver;
use rand::Rng;

#[must_use]
pub fn lgp_fitness(s: &State, _gen: usize) -> f64 {
    let mut fitness = 0.0;
    for _ in 0..100 {
        let mut r = rand::thread_rng();
        let mut reg = vec![0.0; s.cfg.num_reg()]; // Space for work and answer.
        let x = r.gen_range(0.0..100.0);
        reg[1] = -1.0;
        reg[2] = 1.0;
        reg[3] = x;
        let mut exec = LgpExec::new(&reg, &s.ops, 200);
        exec.run();

        let mut ans = 0.0;
        for i in 1..(x as usize) {
            ans += 1.0 / (i as f64);
        }
        fitness += 1.0 / (1.0 + (ans - exec.reg(0)).abs());
    }
    fitness + 1.0 / (1.0 + s.ops.len() as f64)
}

pub fn lgp_evolver(cfg: Cfg) -> Evolver<impl Evaluator> {
    lgp_fitness_evolver(LgpCfg::new(), cfg, lgp_fitness)
}
