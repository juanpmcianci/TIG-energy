/*!
 * Comprehensive Parameter Range Sweep — All 5 Tracks
 *
 * For each track, sweeps each parameter independently and reports:
 * - Greedy profit (mean, std)
 * - Decomp profit (mean, std)
 * - Gap (%)
 * - Feasibility (%)
 * - CV
 *
 * Run with: cargo run --example range_sweep --release
 */

use tig_challenges::energy_arbitrage_v2::{
    constants, Level2Challenge, Level2Difficulty, Level2Solution,
    Track, PortfolioAction, SignedAction,
};
use tig_algorithms::energy_arbitrage_v2::{level2_greedy, level2_decomposition};

fn make_seed(idx: usize) -> [u8; 32] {
    let mut seed = [0u8; 32];
    let bytes = (idx as u64).to_le_bytes();
    seed[..8].copy_from_slice(&bytes);
    seed[8] = 0xBE; seed[9] = 0xEF;
    seed
}

fn stats(values: &[f64]) -> (f64, f64) {
    if values.is_empty() { return (0.0, 0.0); }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// Measure congestion under decomp solution
fn measure_congestion(challenge: &Level2Challenge, solution: &Level2Solution) -> f64 {
    let params = challenge.difficulty.effective_params();
    let h = params.num_steps;
    let mut cong_steps = 0;
    for t in 0..h {
        let action = &solution.schedule[t];
        let injections = challenge.compute_total_injections(action, t);
        let flows = challenge.network.compute_flows(&injections);
        let has_cong = flows.iter().zip(challenge.network.flow_limits.iter())
            .any(|(&f, &lim)| f.abs() >= constants::TAU_CONG * lim);
        if has_cong { cong_steps += 1; }
    }
    cong_steps as f64 / h as f64 * 100.0
}

struct SweepResult {
    param_value: f64,
    greedy_mean: f64,
    greedy_std: f64,
    greedy_feas: f64,
    decomp_mean: f64,
    decomp_std: f64,
    decomp_feas: f64,
    gap: f64,
    cv: f64,
    cong: f64,
}

fn run_sweep_point(track: Track, difficulty: &Level2Difficulty, num_seeds: usize) -> SweepResult {
    let mut gp = Vec::new();
    let mut dp = Vec::new();
    let mut congs = Vec::new();
    let mut total = 0;

    for s in 0..num_seeds {
        let seed = make_seed(s);
        let ch = match Level2Challenge::generate_instance(seed, difficulty) {
            Ok(c) => c,
            Err(_) => continue,
        };
        total += 1;

        if let Ok(Some(ref sol)) = level2_greedy::solve_challenge(&ch) {
            if let Ok(p) = ch.verify_solution(sol) { gp.push(p); }
        }
        if let Ok(Some(ref sol)) = level2_decomposition::solve_challenge(&ch) {
            if let Ok(p) = ch.verify_solution(sol) {
                dp.push(p);
                congs.push(measure_congestion(&ch, sol));
            }
        }
    }

    let (gm, gs) = stats(&gp);
    let (dm, ds) = stats(&dp);
    let gap = if dm.abs() > 1e-6 && !gp.is_empty() { (1.0 - gm / dm) * 100.0 } else { f64::NAN };
    let cv = if dm.abs() > 1e-6 { ds / dm.abs() } else { f64::NAN };
    let (cm, _) = stats(&congs);

    SweepResult {
        param_value: 0.0, // filled by caller
        greedy_mean: gm,
        greedy_std: gs,
        greedy_feas: if total > 0 { gp.len() as f64 / total as f64 * 100.0 } else { 0.0 },
        decomp_mean: dm,
        decomp_std: ds,
        decomp_feas: if total > 0 { dp.len() as f64 / total as f64 * 100.0 } else { 0.0 },
        gap,
        cv,
        cong: cm,
    }
}

fn print_sweep_header(param_name: &str) {
    println!("  {:>8} | {:>10} {:>8} {:>6} | {:>10} {:>8} {:>6} | {:>6} {:>6} {:>6}",
        param_name, "Gr mean", "Gr std", "Gr F%",
        "De mean", "De std", "De F%",
        "Gap%", "CV", "Cong%");
    println!("  {}", "-".repeat(105));
}

fn print_sweep_row(r: &SweepResult) {
    println!("  {:>8.3} | {:>10.0} {:>8.0} {:>5.0}% | {:>10.0} {:>8.0} {:>5.0}% | {:>5.1}% {:>5.3} {:>5.1}%",
        r.param_value, r.greedy_mean, r.greedy_std, r.greedy_feas,
        r.decomp_mean, r.decomp_std, r.decomp_feas,
        r.gap, r.cv, r.cong);
}

fn main() {
    let tracks = [Track::Track1, Track::Track2, Track::Track3, Track::Track4, Track::Track5];
    let num_seeds = 5;

    println!("================================================================");
    println!("  Comprehensive Parameter Range Sweep — All 5 Tracks");
    println!("  {} seeds per point, F_base={}, τ_cong={}, flow_margin=0.85",
        num_seeds, constants::NOMINAL_FLOW_LIMIT, constants::TAU_CONG);
    println!("================================================================\n");

    for &track in &tracks {
        let base = track.parameters();
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║  {:?} — n={}, L={}, m={}, H={}",
            track, base.num_nodes, base.num_lines, base.num_batteries, base.num_steps);
        println!("║  Base: γ_cong={:.2}, σ={:.2}, ρ_jump={:.2}, α={:.1}, h={:.1}",
            base.gamma_cong, base.sigma, base.rho_jump, base.alpha, base.heterogeneity);
        println!("╚══════════════════════════════════════════════════════════════╝\n");

        // --- γ_cong sweep ---
        println!("  === γ_cong sweep (base={:.2}) ===", base.gamma_cong);
        print_sweep_header("γ_cong");
        for &val in &[0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.20, 1.50] {
            let mut d = Level2Difficulty::from_track(track);
            d.congestion_factor = Some(val);
            d.profit_threshold = -1e12;
            let mut r = run_sweep_point(track, &d, num_seeds);
            r.param_value = val;
            print_sweep_row(&r);
        }
        println!();

        // --- σ sweep ---
        println!("  === σ sweep (base={:.2}) ===", base.sigma);
        print_sweep_header("σ");
        for &val in &[0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50] {
            let mut d = Level2Difficulty::from_track(track);
            d.volatility = Some(val);
            d.profit_threshold = -1e12;
            let mut r = run_sweep_point(track, &d, num_seeds);
            r.param_value = val;
            print_sweep_row(&r);
        }
        println!();

        // --- ρ_jump sweep ---
        println!("  === ρ_jump sweep (base={:.2}) ===", base.rho_jump);
        print_sweep_header("ρ_jump");
        for &val in &[0.00, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15] {
            let mut d = Level2Difficulty::from_track(track);
            d.jump_probability = Some(val);
            d.profit_threshold = -1e12;
            let mut r = run_sweep_point(track, &d, num_seeds);
            r.param_value = val;
            print_sweep_row(&r);
        }
        println!();

        // --- α sweep ---
        println!("  === α sweep (base={:.1}) ===", base.alpha);
        print_sweep_header("α");
        for &val in &[2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0] {
            let mut d = Level2Difficulty::from_track(track);
            d.tail_index = Some(val);
            d.profit_threshold = -1e12;
            let mut r = run_sweep_point(track, &d, num_seeds);
            r.param_value = val;
            print_sweep_row(&r);
        }
        println!();

        // --- h sweep ---
        println!("  === h sweep (base={:.1}) ===", base.heterogeneity);
        print_sweep_header("h");
        for &val in &[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
            let mut d = Level2Difficulty::from_track(track);
            d.heterogeneity = Some(val);
            d.profit_threshold = -1e12;
            let mut r = run_sweep_point(track, &d, num_seeds);
            r.param_value = val;
            print_sweep_row(&r);
        }
        println!("\n");
    }
}
