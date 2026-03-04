/*!
 * Focused Parameter Sweep for Congestion Tuning
 *
 * Systematically tests combinations of:
 * - NOMINAL_FLOW_LIMIT (F_base)
 * - flow_margin
 * - TAU_CONG
 *
 * Since these are constants in the challenge code, we instead measure
 * congestion at different thresholds and compute derived metrics.
 *
 * Run with: cargo run --example param_sweep --release
 */

use tig_challenges::energy_arbitrage_v2::{
    constants, Level2Challenge, Level2Difficulty, Level2Solution,
    Track, TrackParameters, PortfolioAction, SignedAction,
};
use tig_algorithms::energy_arbitrage_v2::{level2_greedy, level2_decomposition};
use std::time::Instant;

fn make_seed(idx: usize) -> [u8; 32] {
    let mut seed = [0u8; 32];
    let bytes = (idx as u64).to_le_bytes();
    seed[..8].copy_from_slice(&bytes);
    seed[8] = 0xDE; seed[9] = 0xAD;
    seed
}

/// Analyze flows under a given solution at multiple congestion thresholds
fn analyze_congestion_detailed(
    challenge: &Level2Challenge,
    solution: &Level2Solution,
) -> CongestionProfile {
    let params = challenge.difficulty.effective_params();
    let h = params.num_steps;
    let m = challenge.batteries.len();
    let num_lines = challenge.network.flow_limits.len();

    let mut max_util_per_step = Vec::with_capacity(h);
    let mut avg_util_per_step = Vec::with_capacity(h);
    let mut active_count = 0;

    // Also measure exo-only (idle) flows
    let idle_portfolio = PortfolioAction {
        actions: vec![SignedAction::idle(); m],
    };
    let mut exo_max_util_per_step = Vec::with_capacity(h);

    for t in 0..h {
        // Exo-only flows
        let exo_inj = challenge.compute_total_injections(&idle_portfolio, t);
        let exo_flows = challenge.network.compute_flows(&exo_inj);
        let exo_max_u = exo_flows.iter().zip(challenge.network.flow_limits.iter())
            .map(|(&f, &lim)| f.abs() / lim)
            .fold(0.0f64, f64::max);
        exo_max_util_per_step.push(exo_max_u);

        // Solution flows
        let action = &solution.schedule[t];
        let injections = challenge.compute_total_injections(action, t);
        let flows = challenge.network.compute_flows(&injections);

        let mut max_u = 0.0f64;
        let mut sum_u = 0.0;
        for (l, &flow) in flows.iter().enumerate() {
            let u = flow.abs() / challenge.network.flow_limits[l];
            max_u = max_u.max(u);
            sum_u += u;
        }
        max_util_per_step.push(max_u);
        avg_util_per_step.push(sum_u / num_lines as f64);

        // Count active batteries
        for b in 0..m {
            if action.actions[b].power_mw.abs() > 0.1 {
                active_count += 1;
            }
        }
    }

    // Compute congestion at various thresholds
    let thresholds = [0.70, 0.80, 0.85, 0.90, 0.95, 0.97];
    let mut cong_at_threshold = Vec::new();
    for &tau in &thresholds {
        let count = max_util_per_step.iter().filter(|&&u| u >= tau).count();
        cong_at_threshold.push((tau, count as f64 / h as f64 * 100.0));
    }

    let mut exo_cong_at_threshold = Vec::new();
    for &tau in &thresholds {
        let count = exo_max_util_per_step.iter().filter(|&&u| u >= tau).count();
        exo_cong_at_threshold.push((tau, count as f64 / h as f64 * 100.0));
    }

    // Utilization distribution
    let avg_max_util = max_util_per_step.iter().sum::<f64>() / h as f64;
    let avg_avg_util = avg_util_per_step.iter().sum::<f64>() / h as f64;
    let exo_avg_max_util = exo_max_util_per_step.iter().sum::<f64>() / h as f64;

    CongestionProfile {
        avg_max_util,
        avg_avg_util,
        exo_avg_max_util,
        cong_at_threshold,
        exo_cong_at_threshold,
        active_battery_frac: active_count as f64 / (h * m) as f64,
    }
}

struct CongestionProfile {
    avg_max_util: f64,       // Average of max(|flow_l|/limit_l) across timesteps
    avg_avg_util: f64,       // Average of mean(|flow_l|/limit_l) across timesteps
    exo_avg_max_util: f64,   // Same as avg_max_util but under idle (exo-only) dispatch
    cong_at_threshold: Vec<(f64, f64)>,     // (τ, % of steps with max_util ≥ τ) under solution
    exo_cong_at_threshold: Vec<(f64, f64)>, // Same under idle
    active_battery_frac: f64,
}

fn main() {
    println!("==========================================================");
    println!("  Parameter Sweep: Congestion Profile Analysis");
    println!("  Current settings: F_base={}, flow_margin in code, τ_cong={}",
        constants::NOMINAL_FLOW_LIMIT, constants::TAU_CONG);
    println!("==========================================================\n");

    let tracks = [Track::Track1, Track::Track2, Track::Track3, Track::Track4, Track::Track5];
    let num_seeds = 5;

    // Header
    println!("{:>7} | {:>6} {:>6} {:>6} | {:>8} {:>8} {:>8} | {:>7} {:>7} {:>7} {:>7} {:>7} {:>7} | {:>7}",
        "Track", "n", "γ_cong", "EffLim",
        "ExoMax%", "SolMax%", "SolAvg%",
        "τ=.70", "τ=.80", "τ=.85", "τ=.90", "τ=.95", "τ=.97",
        "Act%");
    println!("{}", "-".repeat(130));

    for &track in &tracks {
        let params = track.parameters();
        let eff_lim = constants::NOMINAL_FLOW_LIMIT * params.gamma_cong;

        let mut profiles: Vec<CongestionProfile> = Vec::new();
        let mut greedy_profits: Vec<f64> = Vec::new();
        let mut decomp_profits: Vec<f64> = Vec::new();

        for s in 0..num_seeds {
            let seed = make_seed(s);
            let difficulty = Level2Difficulty {
                track,
                profit_threshold: -1e12,
                ..Default::default()
            };

            let challenge = match Level2Challenge::generate_instance(seed, &difficulty) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("  {:?} seed {}: {}", track, s, e);
                    continue;
                }
            };

            // Solve with decomp
            let decomp_result = level2_decomposition::solve_challenge(&challenge);
            let (decomp_sol, decomp_profit) = match decomp_result {
                Ok(Some(ref sol)) => {
                    match challenge.verify_solution(sol) {
                        Ok(p) => (Some(sol.clone()), Some(p)),
                        Err(_) => (None, None),
                    }
                }
                _ => (None, None),
            };

            // Solve with greedy
            if let Ok(Some(ref sol)) = level2_greedy::solve_challenge(&challenge) {
                if let Ok(p) = challenge.verify_solution(sol) {
                    greedy_profits.push(p);
                }
            }

            if let Some(p) = decomp_profit {
                decomp_profits.push(p);
            }

            // Analyze congestion under decomp solution
            if let Some(ref sol) = decomp_sol {
                let profile = analyze_congestion_detailed(&challenge, sol);
                profiles.push(profile);
            }
        }

        if profiles.is_empty() {
            println!("{:>7} | NO VALID SOLUTIONS", format!("{:?}", track));
            continue;
        }

        // Average profiles
        let n_prof = profiles.len() as f64;
        let avg_exo_max = profiles.iter().map(|p| p.exo_avg_max_util).sum::<f64>() / n_prof;
        let avg_sol_max = profiles.iter().map(|p| p.avg_max_util).sum::<f64>() / n_prof;
        let avg_sol_avg = profiles.iter().map(|p| p.avg_avg_util).sum::<f64>() / n_prof;
        let avg_active = profiles.iter().map(|p| p.active_battery_frac).sum::<f64>() / n_prof;

        // Average congestion at thresholds
        let num_thresholds = profiles[0].cong_at_threshold.len();
        let mut avg_cong = vec![0.0; num_thresholds];
        for p in &profiles {
            for (i, &(_, pct)) in p.cong_at_threshold.iter().enumerate() {
                avg_cong[i] += pct;
            }
        }
        for v in &mut avg_cong {
            *v /= n_prof;
        }

        println!("{:>7} | {:>6} {:>6.2} {:>6.1} | {:>7.1}% {:>7.1}% {:>7.1}% | {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% | {:>6.1}%",
            format!("{:?}", track),
            params.num_nodes, params.gamma_cong, eff_lim,
            avg_exo_max * 100.0, avg_sol_max * 100.0, avg_sol_avg * 100.0,
            avg_cong[0], avg_cong[1], avg_cong[2], avg_cong[3], avg_cong[4], avg_cong[5],
            avg_active * 100.0);

        // Print profit info
        if !decomp_profits.is_empty() {
            let gm = if greedy_profits.is_empty() { 0.0 }
                else { greedy_profits.iter().sum::<f64>() / greedy_profits.len() as f64 };
            let dm = decomp_profits.iter().sum::<f64>() / decomp_profits.len() as f64;
            let gap = if dm.abs() > 1e-6 { (1.0 - gm / dm) * 100.0 } else { f64::NAN };
            println!("         Profit: Greedy=${:.0}, Decomp=${:.0}, Gap={:.0}%", gm, dm, gap);
        }
    }

    println!("\n");

    // Part 2: Show exo-only congestion profile
    println!("==========================================================");
    println!("  Exo-Only Congestion (idle batteries)");
    println!("==========================================================\n");

    println!("{:>7} | {:>7} {:>7} {:>7} {:>7} {:>7} {:>7}",
        "Track", "τ=.70", "τ=.80", "τ=.85", "τ=.90", "τ=.95", "τ=.97");
    println!("{}", "-".repeat(60));

    for &track in &tracks {
        let params = track.parameters();
        let mut all_exo_cong = vec![vec![]; 6];

        for s in 0..num_seeds {
            let seed = make_seed(s);
            let difficulty = Level2Difficulty {
                track,
                profit_threshold: -1e12,
                ..Default::default()
            };

            let challenge = match Level2Challenge::generate_instance(seed, &difficulty) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Compute exo-only max utilization per timestep
            let h = params.num_steps;
            let m = challenge.batteries.len();
            let idle = PortfolioAction {
                actions: vec![SignedAction::idle(); m],
            };

            let thresholds = [0.70, 0.80, 0.85, 0.90, 0.95, 0.97];
            let mut counts = vec![0usize; 6];

            for t in 0..h {
                let inj = challenge.compute_total_injections(&idle, t);
                let flows = challenge.network.compute_flows(&inj);
                let max_u = flows.iter().zip(challenge.network.flow_limits.iter())
                    .map(|(&f, &lim)| f.abs() / lim)
                    .fold(0.0f64, f64::max);

                for (i, &tau) in thresholds.iter().enumerate() {
                    if max_u >= tau {
                        counts[i] += 1;
                    }
                }
            }

            for (i, &count) in counts.iter().enumerate() {
                all_exo_cong[i].push(count as f64 / h as f64 * 100.0);
            }
        }

        let avgs: Vec<f64> = all_exo_cong.iter()
            .map(|v| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 })
            .collect();

        println!("{:>7} | {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}%",
            format!("{:?}", track),
            avgs[0], avgs[1], avgs[2], avgs[3], avgs[4], avgs[5]);
    }

    println!("\n");

    // Part 3: Per-line utilization distribution
    println!("==========================================================");
    println!("  Flow Utilization Distribution (fraction of lines above threshold)");
    println!("==========================================================\n");

    println!("{:>7} | {:>6} {:>6} | {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Track", "Lines", "EffLim",
        ">50%", ">70%", ">80%", ">90%", ">95%");
    println!("{}", "-".repeat(75));

    for &track in &tracks {
        let params = track.parameters();
        let eff_lim = constants::NOMINAL_FLOW_LIMIT * params.gamma_cong;

        let mut line_frac_above = vec![vec![]; 5]; // for thresholds 50, 70, 80, 90, 95

        for s in 0..num_seeds {
            let seed = make_seed(s);
            let difficulty = Level2Difficulty {
                track,
                profit_threshold: -1e12,
                ..Default::default()
            };

            let challenge = match Level2Challenge::generate_instance(seed, &difficulty) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let h = params.num_steps;
            let m = challenge.batteries.len();
            let num_lines = challenge.network.flow_limits.len();
            let idle = PortfolioAction {
                actions: vec![SignedAction::idle(); m],
            };

            let thresholds = [0.50, 0.70, 0.80, 0.90, 0.95];
            let mut total_above = vec![0usize; 5];
            let total_pairs = h * num_lines;

            for t in 0..h {
                let inj = challenge.compute_total_injections(&idle, t);
                let flows = challenge.network.compute_flows(&inj);
                for (l, &flow) in flows.iter().enumerate() {
                    let u = flow.abs() / challenge.network.flow_limits[l];
                    for (i, &tau) in thresholds.iter().enumerate() {
                        if u >= tau {
                            total_above[i] += 1;
                        }
                    }
                }
            }

            for (i, &count) in total_above.iter().enumerate() {
                line_frac_above[i].push(count as f64 / total_pairs as f64 * 100.0);
            }
        }

        let avgs: Vec<f64> = line_frac_above.iter()
            .map(|v| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 })
            .collect();

        println!("{:>7} | {:>6} {:>6.1} | {:>7.2}% {:>7.2}% {:>7.2}% {:>7.2}% {:>7.2}%",
            format!("{:?}", track),
            params.num_lines, eff_lim,
            avgs[0], avgs[1], avgs[2], avgs[3], avgs[4]);
    }
}
