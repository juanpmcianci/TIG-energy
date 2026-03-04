/*!
 * Track-level Parameter & Difficulty Progression Experiments
 *
 * Uses the spec-compliant implementation (outer workspace) with:
 * - Proper Track enum with (n, L, m, H, γ_cong, σ, ρ_jump, α, h)
 * - Spanning tree + random edge network topology
 * - Exogenous injections with spatiotemporal correlation
 * - Hash-in-counter-mode RT price generation
 * - Signed action model with quantized hashing
 * - Δt = 0.25 h (15-minute steps)
 *
 * Metrics evaluated (from parameter_justification.tex):
 * - M1: Greedy profit mean and std
 * - M2: Decomp profit mean and std
 * - M3: Greedy-to-decomp gap
 * - M4: Feasibility rate (both solvers)
 * - M5: Solve time
 * - M6: Verification time
 * - M7: Profit CV
 * - M8: Congestion frequency
 * - M9: Active battery fraction
 * - M10: Per-battery profit distribution
 *
 * Run with: cargo run --example track_experiments --release
 */

use tig_challenges::energy_arbitrage_v2::{
    constants, Level2Challenge, Level2Difficulty, Level2Solution,
    Track, TrackParameters, PortfolioAction, SignedAction,
};
use tig_algorithms::energy_arbitrage_v2::{level2_greedy, level2_decomposition};
use std::time::Instant;

// ============================================================================
// Result types
// ============================================================================

#[derive(Debug, Clone)]
struct TrackResult {
    track: String,
    params: TrackParameters,
    // Instance-level metrics
    instance_results: Vec<InstanceResult>,
}

#[derive(Debug, Clone)]
struct InstanceResult {
    seed_idx: usize,
    // Instance properties
    da_price_mean: f64,
    da_price_std: f64,
    da_price_spread: f64, // cross-node spread
    exo_flow_utilization: f64, // avg |flow|/limit under zero storage
    // Greedy solver
    greedy_profit: Option<f64>,
    greedy_time_us: u128,
    greedy_error: Option<String>,
    // Decomposition solver
    decomp_profit: Option<f64>,
    decomp_time_us: u128,
    decomp_error: Option<String>,
    // Verification time
    verify_time_us: u128,
    // Detailed metrics from solution
    congestion_events: usize,   // how many steps had flow > τ_cong * limit
    active_battery_frac: f64,   // fraction of (battery, step) with |u| > 0.1
    total_energy_traded: f64,   // Σ |u| Δt across all batteries and steps
}

fn make_seed(idx: usize) -> [u8; 32] {
    let mut seed = [0u8; 32];
    let bytes = (idx as u64).to_le_bytes();
    seed[..8].copy_from_slice(&bytes);
    seed[8] = 0xCA; seed[9] = 0xFE; seed[10] = 0xBA; seed[11] = 0xBE;
    seed
}

// ============================================================================
// Instance analysis (before solving)
// ============================================================================

fn analyze_instance(challenge: &Level2Challenge) -> (f64, f64, f64, f64) {
    let params = challenge.difficulty.effective_params();
    let n = params.num_nodes;
    let h = params.num_steps;

    // DA price stats
    let mut all_prices = Vec::new();
    for node in 0..n {
        for t in 0..h {
            all_prices.push(challenge.day_ahead_prices[node][t]);
        }
    }
    let mean = all_prices.iter().sum::<f64>() / all_prices.len() as f64;
    let var = all_prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / all_prices.len() as f64;
    let std = var.sqrt();

    // Cross-node price spread
    let mut spread_sum = 0.0;
    for t in 0..h {
        let prices_t: Vec<f64> = (0..n).map(|i| challenge.day_ahead_prices[i][t]).collect();
        let min = prices_t.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = prices_t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        spread_sum += max - min;
    }
    let avg_spread = spread_sum / h as f64;

    // Exogenous flow utilization
    let m = challenge.batteries.len();
    let idle_portfolio = PortfolioAction {
        actions: vec![SignedAction::idle(); m],
    };
    let mut util_sum = 0.0;
    let mut util_count = 0;
    for t in 0..h {
        let inj = challenge.compute_total_injections(&idle_portfolio, t);
        let flows = challenge.network.compute_flows(&inj);
        for (l, &flow) in flows.iter().enumerate() {
            util_sum += flow.abs() / challenge.network.flow_limits[l];
            util_count += 1;
        }
    }
    let avg_util = if util_count > 0 { util_sum / util_count as f64 } else { 0.0 };

    (mean, std, avg_spread, avg_util)
}

// ============================================================================
// Solution analysis
// ============================================================================

fn analyze_solution(
    challenge: &Level2Challenge,
    solution: &Level2Solution,
) -> (usize, f64, f64) {
    let params = challenge.difficulty.effective_params();
    let h = params.num_steps;
    let m = challenge.batteries.len();
    let dt = constants::DELTA_T;

    // Count congestion events and active batteries
    let mut congestion_events = 0;
    let mut active_count = 0;
    let mut total_energy = 0.0;

    // Need to simulate to get flows
    let mut socs: Vec<f64> = challenge.batteries.iter()
        .map(|b| b.spec.soc_initial_mwh).collect();

    for t in 0..h {
        let action = &solution.schedule[t];
        let injections = challenge.compute_total_injections(action, t);
        let flows = challenge.network.compute_flows(&injections);

        // Check congestion
        let has_congestion = flows.iter().zip(challenge.network.flow_limits.iter())
            .any(|(&f, &lim)| f.abs() >= constants::TAU_CONG * lim);
        if has_congestion {
            congestion_events += 1;
        }

        // Active batteries and energy
        for b in 0..m {
            let u = action.actions[b].power_mw;
            if u.abs() > 0.1 {
                active_count += 1;
            }
            total_energy += u.abs() * dt;
        }

        // Update SOCs
        for b in 0..m {
            let new_soc = challenge.apply_action_to_soc(b, socs[b], &action.actions[b]);
            socs[b] = new_soc.clamp(
                challenge.batteries[b].spec.soc_min_mwh,
                challenge.batteries[b].spec.soc_max_mwh,
            );
        }
    }

    let active_frac = active_count as f64 / (h * m) as f64;
    (congestion_events, active_frac, total_energy)
}

// ============================================================================
// Run single track experiment
// ============================================================================

fn run_track_experiment(track: Track, num_seeds: usize) -> TrackResult {
    let params = track.parameters();
    let track_name = format!("{:?}", track);

    let mut instance_results = Vec::new();

    for s in 0..num_seeds {
        let seed = make_seed(s);
        let difficulty = Level2Difficulty {
            track,
            profit_threshold: -1e12, // No threshold for experiments
            ..Default::default()
        };

        // Generate instance
        let challenge = match Level2Challenge::generate_instance(seed, &difficulty) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  Seed {}: Generation failed: {}", s, e);
                continue;
            }
        };

        // Analyze instance
        let (da_mean, da_std, da_spread, exo_util) = analyze_instance(&challenge);

        // --- Greedy solver ---
        let t0 = Instant::now();
        let greedy_result = level2_greedy::solve_challenge(&challenge);
        let greedy_time = t0.elapsed().as_micros();

        let (greedy_profit, greedy_error) = match greedy_result {
            Ok(Some(ref sol)) => {
                match challenge.verify_solution(sol) {
                    Ok(p) => (Some(p), None),
                    Err(e) => (None, Some(e.to_string())),
                }
            }
            Ok(None) => (None, Some("No solution".to_string())),
            Err(ref e) => (None, Some(e.to_string())),
        };

        // --- Decomposition solver ---
        let t0 = Instant::now();
        let decomp_result = level2_decomposition::solve_challenge(&challenge);
        let decomp_time = t0.elapsed().as_micros();

        let (decomp_profit, decomp_error) = match decomp_result {
            Ok(Some(ref sol)) => {
                let t0v = Instant::now();
                let verify_result = challenge.verify_solution(sol);
                let _verify_time = t0v.elapsed().as_micros();
                match verify_result {
                    Ok(p) => (Some(p), None),
                    Err(e) => (None, Some(e.to_string())),
                }
            }
            Ok(None) => (None, Some("No solution".to_string())),
            Err(ref e) => (None, Some(e.to_string())),
        };

        // Measure verification time separately (use decomp solution if available)
        let verify_time = if let Ok(Some(ref sol)) = decomp_result {
            let t0v = Instant::now();
            let _ = challenge.verify_solution(sol);
            t0v.elapsed().as_micros()
        } else if let Ok(Some(ref sol)) = greedy_result {
            let t0v = Instant::now();
            let _ = challenge.verify_solution(sol);
            t0v.elapsed().as_micros()
        } else {
            0
        };

        // Analyze solution (use decomp if valid, else greedy)
        let (cong_events, active_frac, energy) = if let Ok(Some(ref sol)) = decomp_result {
            analyze_solution(&challenge, sol)
        } else if let Ok(Some(ref sol)) = greedy_result {
            analyze_solution(&challenge, sol)
        } else {
            (0, 0.0, 0.0)
        };

        instance_results.push(InstanceResult {
            seed_idx: s,
            da_price_mean: da_mean,
            da_price_std: da_std,
            da_price_spread: da_spread,
            exo_flow_utilization: exo_util,
            greedy_profit,
            greedy_time_us: greedy_time,
            greedy_error,
            decomp_profit,
            decomp_time_us: decomp_time,
            decomp_error,
            verify_time_us: verify_time,
            congestion_events: cong_events,
            active_battery_frac: active_frac,
            total_energy_traded: energy,
        });
    }

    TrackResult {
        track: track_name,
        params,
        instance_results,
    }
}

// ============================================================================
// Statistics helpers
// ============================================================================

fn stats(values: &[f64]) -> (f64, f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (mean, var.sqrt(), min, max)
}

fn feasible_profits(results: &[InstanceResult], solver: &str) -> Vec<f64> {
    results.iter().filter_map(|r| {
        match solver {
            "greedy" => r.greedy_profit,
            "decomp" => r.decomp_profit,
            _ => None,
        }
    }).collect()
}

// ============================================================================
// Report generation
// ============================================================================

fn print_track_report(result: &TrackResult) {
    let p = &result.params;
    let n = result.instance_results.len();

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  {} — n={}, L={}, m={}, H={}", result.track, p.num_nodes, p.num_lines, p.num_batteries, p.num_steps);
    println!("║  γ_cong={:.2}, σ={:.2}, ρ_jump={:.2}, α={:.1}, h={:.1}",
        p.gamma_cong, p.sigma, p.rho_jump, p.alpha, p.heterogeneity);
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Instance characteristics
    let da_means: Vec<f64> = result.instance_results.iter().map(|r| r.da_price_mean).collect();
    let da_spreads: Vec<f64> = result.instance_results.iter().map(|r| r.da_price_spread).collect();
    let exo_utils: Vec<f64> = result.instance_results.iter().map(|r| r.exo_flow_utilization).collect();

    let (dam, _, _, _) = stats(&da_means);
    let (dsp, _, _, _) = stats(&da_spreads);
    let (exu, _, _, _) = stats(&exo_utils);

    println!("  Instance characteristics ({} seeds):", n);
    println!("    DA price mean:         ${:.1}/MWh", dam);
    println!("    DA cross-node spread:  ${:.1}/MWh", dsp);
    println!("    Exo flow utilization:  {:.1}% of line limits", exu * 100.0);
    println!();

    // Greedy results
    let gp = feasible_profits(&result.instance_results, "greedy");
    let gfeas = gp.len() as f64 / n as f64;
    let (gm, gs, gmin, gmax) = stats(&gp);
    let gcv = if gm.abs() > 1e-6 { gs / gm.abs() } else { f64::NAN };

    let gt: Vec<f64> = result.instance_results.iter().map(|r| r.greedy_time_us as f64 / 1000.0).collect();
    let (gtm, _, _, _) = stats(&gt);

    println!("  GREEDY solver:");
    println!("    Feasibility rate:  {:.0}% ({}/{})", gfeas * 100.0, gp.len(), n);
    if !gp.is_empty() {
        println!("    Profit:  mean=${:.1}, std=${:.1}, min=${:.1}, max=${:.1}", gm, gs, gmin, gmax);
        println!("    Profit CV:  {:.3}", gcv);
    }
    println!("    Solve time:  {:.2} ms avg", gtm);

    // Print greedy errors
    let errors: Vec<&str> = result.instance_results.iter()
        .filter_map(|r| r.greedy_error.as_deref())
        .collect();
    if !errors.is_empty() && errors.len() <= 3 {
        for e in &errors {
            let truncated: String = e.chars().take(80).collect();
            println!("    Error: {}", truncated);
        }
    } else if !errors.is_empty() {
        // Categorize errors
        let flow_errors = errors.iter().filter(|e| e.contains("flow")).count();
        let soc_errors = errors.iter().filter(|e| e.contains("SOC")).count();
        let profit_errors = errors.iter().filter(|e| e.contains("profit")).count();
        let other = errors.len() - flow_errors - soc_errors - profit_errors;
        println!("    Errors: {} flow, {} SOC, {} profit, {} other", flow_errors, soc_errors, profit_errors, other);
    }
    println!();

    // Decomposition results
    let dp = feasible_profits(&result.instance_results, "decomp");
    let dfeas = dp.len() as f64 / n as f64;
    let (dm, ds, dmin, dmax) = stats(&dp);
    let dcv = if dm.abs() > 1e-6 { ds / dm.abs() } else { f64::NAN };

    let dt_solve: Vec<f64> = result.instance_results.iter().map(|r| r.decomp_time_us as f64 / 1000.0).collect();
    let (dtm, _, _, _) = stats(&dt_solve);

    println!("  DECOMPOSITION solver:");
    println!("    Feasibility rate:  {:.0}% ({}/{})", dfeas * 100.0, dp.len(), n);
    if !dp.is_empty() {
        println!("    Profit:  mean=${:.1}, std=${:.1}, min=${:.1}, max=${:.1}", dm, ds, dmin, dmax);
        println!("    Profit CV:  {:.3}", dcv);
    }
    println!("    Solve time:  {:.2} ms avg", dtm);

    // Print decomp errors
    let errors: Vec<&str> = result.instance_results.iter()
        .filter_map(|r| r.decomp_error.as_deref())
        .collect();
    if !errors.is_empty() && errors.len() <= 3 {
        for e in &errors {
            let truncated: String = e.chars().take(80).collect();
            println!("    Error: {}", truncated);
        }
    } else if !errors.is_empty() {
        let flow_errors = errors.iter().filter(|e| e.contains("flow")).count();
        let soc_errors = errors.iter().filter(|e| e.contains("SOC")).count();
        let profit_errors = errors.iter().filter(|e| e.contains("profit")).count();
        let other = errors.len() - flow_errors - soc_errors - profit_errors;
        println!("    Errors: {} flow, {} SOC, {} profit, {} other", flow_errors, soc_errors, profit_errors, other);
    }
    println!();

    // Gap analysis
    if !gp.is_empty() && !dp.is_empty() && dm.abs() > 1e-6 {
        let gap = 1.0 - gm / dm;
        println!("  GREEDY-DECOMP GAP: {:.1}%", gap * 100.0);
    }
    println!();

    // Verification time
    let vt: Vec<f64> = result.instance_results.iter()
        .filter(|r| r.verify_time_us > 0)
        .map(|r| r.verify_time_us as f64 / 1000.0)
        .collect();
    if !vt.is_empty() {
        let (vtm, _, _, vtmax) = stats(&vt);
        println!("  Verification time:  {:.2} ms avg, {:.2} ms max", vtm, vtmax);
    }

    // Congestion and activity
    let cong: Vec<f64> = result.instance_results.iter()
        .map(|r| r.congestion_events as f64 / result.params.num_steps as f64 * 100.0)
        .collect();
    let active: Vec<f64> = result.instance_results.iter().map(|r| r.active_battery_frac).collect();
    let energy: Vec<f64> = result.instance_results.iter().map(|r| r.total_energy_traded).collect();

    let (cm, _, _, _) = stats(&cong);
    let (am, _, _, _) = stats(&active);
    let (em, _, _, _) = stats(&energy);

    println!("  Congestion frequency:   {:.1}% of steps", cm);
    println!("  Active battery-steps:   {:.1}%", am * 100.0);
    println!("  Total energy traded:    {:.0} MWh avg", em);
    println!();
}

// ============================================================================
// Summary table
// ============================================================================

fn print_summary_table(results: &[TrackResult]) {
    println!("╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         SUMMARY TABLE — ALL TRACKS                                              ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝\n");

    println!("  {:>7} | {:>4} {:>4} {:>4} {:>4} | {:>10} {:>8} | {:>10} {:>8} | {:>6} | {:>7} {:>6}",
        "Track", "n", "L", "m", "H",
        "Gr profit", "Gr feas",
        "De profit", "De feas",
        "Gap",
        "De ms", "V ms");
    println!("  {}", "-".repeat(100));

    for r in results {
        let p = &r.params;
        let n = r.instance_results.len();

        let gp = feasible_profits(&r.instance_results, "greedy");
        let dp = feasible_profits(&r.instance_results, "decomp");

        let (gm, _, _, _) = stats(&gp);
        let (dm, _, _, _) = stats(&dp);

        let gfeas = gp.len() as f64 / n as f64 * 100.0;
        let dfeas = dp.len() as f64 / n as f64 * 100.0;

        let gap = if dm.abs() > 1e-6 && !gp.is_empty() {
            format!("{:.0}%", (1.0 - gm / dm) * 100.0)
        } else {
            "N/A".to_string()
        };

        let dt_solve: Vec<f64> = r.instance_results.iter().map(|r| r.decomp_time_us as f64 / 1000.0).collect();
        let vt: Vec<f64> = r.instance_results.iter()
            .filter(|r| r.verify_time_us > 0)
            .map(|r| r.verify_time_us as f64 / 1000.0).collect();
        let (dtm, _, _, _) = stats(&dt_solve);
        let (vtm, _, _, _) = stats(&vt);

        println!("  {:>7} | {:>4} {:>4} {:>4} {:>4} | {:>10.0} {:>7.0}% | {:>10.0} {:>7.0}% | {:>6} | {:>7.1} {:>5.1}",
            r.track, p.num_nodes, p.num_lines, p.num_batteries, p.num_steps,
            gm, gfeas, dm, dfeas, gap, dtm, vtm);
    }

    println!();

    // Difficulty progression metrics
    println!("  {:>7} | {:>8} {:>8} | {:>8} {:>8} | {:>8} {:>8} {:>8}",
        "Track", "σ", "α", "γ_cong", "h",
        "Cong%", "Active%", "CV");
    println!("  {}", "-".repeat(85));

    for r in results {
        let p = &r.params;
        let dp = feasible_profits(&r.instance_results, "decomp");
        let (dm, ds, _, _) = stats(&dp);
        let cv = if dm.abs() > 1e-6 { ds / dm.abs() } else { f64::NAN };

        let cong: Vec<f64> = r.instance_results.iter()
            .map(|ir| ir.congestion_events as f64 / p.num_steps as f64 * 100.0)
            .collect();
        let active: Vec<f64> = r.instance_results.iter().map(|ir| ir.active_battery_frac * 100.0).collect();
        let (cm, _, _, _) = stats(&cong);
        let (am, _, _, _) = stats(&active);

        println!("  {:>7} | {:>7.2} {:>7.1} | {:>7.2} {:>7.1} | {:>7.1} {:>7.1} {:>7.3}",
            r.track, p.sigma, p.alpha, p.gamma_cong, p.heterogeneity,
            cm, am, cv);
    }
    println!();
}

// ============================================================================
// Individual parameter sensitivity within tracks
// ============================================================================

fn experiment_sensitivity_within_track(track: Track) {
    let base = track.parameters();
    let track_name = format!("{:?}", track);
    let num_seeds = 5;

    println!("┌──────────────────────────────────────────────────────────────────┐");
    println!("│  Sensitivity analysis within {} (n={}, L={}, m={}, H={})",
        track_name, base.num_nodes, base.num_lines, base.num_batteries, base.num_steps);
    println!("└──────────────────────────────────────────────────────────────────┘\n");

    // Sweep γ_cong
    println!("  --- γ_cong sweep (line limit tightness) ---");
    println!("  {:>8} | {:>10} {:>10} | {:>10} {:>10}",
        "γ_cong", "Gr mean", "Gr feas%", "De mean", "De feas%");
    println!("  {}", "-".repeat(60));

    for &gamma in &[0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.2] {
        let mut difficulty = Level2Difficulty::from_track(track);
        difficulty.congestion_factor = Some(gamma);
        difficulty.profit_threshold = -1e12;

        let mut gp = Vec::new();
        let mut dp = Vec::new();
        let mut total_seeds = 0;

        for s in 0..num_seeds {
            let seed = make_seed(s + 100);
            if let Ok(ch) = Level2Challenge::generate_instance(seed, &difficulty) {
                total_seeds += 1;
                if let Ok(Some(sol)) = level2_greedy::solve_challenge(&ch) {
                    if let Ok(p) = ch.verify_solution(&sol) { gp.push(p); }
                }
                if let Ok(Some(sol)) = level2_decomposition::solve_challenge(&ch) {
                    if let Ok(p) = ch.verify_solution(&sol) { dp.push(p); }
                }
            }
        }

        let (gm, _, _, _) = stats(&gp);
        let (dm, _, _, _) = stats(&dp);
        println!("  {:>8.2} | {:>10.0} {:>9.0}% | {:>10.0} {:>9.0}%",
            gamma, gm, gp.len() as f64 / total_seeds.max(1) as f64 * 100.0,
            dm, dp.len() as f64 / total_seeds.max(1) as f64 * 100.0);
    }
    println!();

    // Sweep σ (volatility)
    println!("  --- σ sweep (volatility) ---");
    println!("  {:>8} | {:>10} {:>10} | {:>10} {:>10}",
        "σ", "Gr mean", "Gr std", "De mean", "De std");
    println!("  {}", "-".repeat(60));

    for &sigma in &[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40] {
        let mut difficulty = Level2Difficulty::from_track(track);
        difficulty.volatility = Some(sigma);
        difficulty.profit_threshold = -1e12;

        let mut gp = Vec::new();
        let mut dp = Vec::new();

        for s in 0..num_seeds {
            let seed = make_seed(s + 200);
            if let Ok(ch) = Level2Challenge::generate_instance(seed, &difficulty) {
                if let Ok(Some(sol)) = level2_greedy::solve_challenge(&ch) {
                    if let Ok(p) = ch.verify_solution(&sol) { gp.push(p); }
                }
                if let Ok(Some(sol)) = level2_decomposition::solve_challenge(&ch) {
                    if let Ok(p) = ch.verify_solution(&sol) { dp.push(p); }
                }
            }
        }

        let (gm, gs, _, _) = stats(&gp);
        let (dm, ds, _, _) = stats(&dp);
        println!("  {:>8.2} | {:>10.0} {:>10.0} | {:>10.0} {:>10.0}",
            sigma, gm, gs, dm, ds);
    }
    println!();

    // Sweep h (heterogeneity)
    println!("  --- h sweep (battery heterogeneity) ---");
    println!("  {:>8} | {:>10} | {:>10} | {:>8}",
        "h", "Gr mean", "De mean", "Gap");
    println!("  {}", "-".repeat(48));

    for &h in &[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
        let mut difficulty = Level2Difficulty::from_track(track);
        difficulty.heterogeneity = Some(h);
        difficulty.profit_threshold = -1e12;

        let mut gp = Vec::new();
        let mut dp = Vec::new();

        for s in 0..num_seeds {
            let seed = make_seed(s + 300);
            if let Ok(ch) = Level2Challenge::generate_instance(seed, &difficulty) {
                if let Ok(Some(sol)) = level2_greedy::solve_challenge(&ch) {
                    if let Ok(p) = ch.verify_solution(&sol) { gp.push(p); }
                }
                if let Ok(Some(sol)) = level2_decomposition::solve_challenge(&ch) {
                    if let Ok(p) = ch.verify_solution(&sol) { dp.push(p); }
                }
            }
        }

        let (gm, _, _, _) = stats(&gp);
        let (dm, _, _, _) = stats(&dp);
        let gap = if dm.abs() > 1e-6 && !gp.is_empty() {
            format!("{:.0}%", (1.0 - gm / dm) * 100.0)
        } else {
            "N/A".to_string()
        };
        println!("  {:>8.1} | {:>10.0} | {:>10.0} | {:>8}",
            h, gm, dm, gap);
    }
    println!();
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("\n{}", "=".repeat(70));
    println!("  TIG Energy Arbitrage — Spec-Compliant Track Experiments");
    println!("  (Δt=0.25h, spanning tree + edges, exo injections, quantized hash)");
    println!("{}\n", "=".repeat(70));

    let total_start = Instant::now();

    let num_seeds = 10; // Per track

    // ========================================================================
    // Part 1: Full track progression experiments
    // ========================================================================
    println!("████████████████████████████████████████████████████████████████████");
    println!("  PART 1: Full Track Progression (all 5 tracks, {} seeds each)", num_seeds);
    println!("████████████████████████████████████████████████████████████████████\n");

    let mut all_results = Vec::new();

    for track in Track::all() {
        let track_start = Instant::now();
        let result = run_track_experiment(track, num_seeds);
        let elapsed = track_start.elapsed();
        println!("  {:?} completed in {:.1?}", track, elapsed);
        print_track_report(&result);
        all_results.push(result);
    }

    print_summary_table(&all_results);

    // ========================================================================
    // Part 2: Per-track parameter sensitivity
    // ========================================================================
    println!("████████████████████████████████████████████████████████████████████");
    println!("  PART 2: Parameter Sensitivity Within Tracks");
    println!("████████████████████████████████████████████████████████████████████\n");

    // Focus sensitivity on Tracks 1 and 3 (small and medium - tractable)
    experiment_sensitivity_within_track(Track::Track1);
    experiment_sensitivity_within_track(Track::Track2);

    let total_elapsed = total_start.elapsed();
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Total experiment time: {:.1?}", total_elapsed);
    println!("══════════════════════════════════════════════════════════════════");
}
