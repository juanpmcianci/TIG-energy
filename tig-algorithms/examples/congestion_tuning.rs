/*!
 * Congestion Tuning Experiments
 *
 * The core problem: exogenous flows use only ~5-9% of line capacity,
 * so γ_cong has no effect and congestion never occurs.
 *
 * This experiment tests different parameter combinations to find ranges
 * that produce meaningful congestion while keeping instances solvable.
 *
 * Approach: post-process generated instances by scaling:
 * - Exogenous injections (↑ increases base flow utilization)
 * - Flow limits (↓ tightens constraints)
 *
 * Run with: cargo run --example congestion_tuning --release
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
    seed[8] = 0xCA; seed[9] = 0xFE;
    seed
}

// ============================================================================
// Instance modification helpers
// ============================================================================

/// Scale exogenous injections by a factor (rebalances slack bus)
fn scale_exogenous(challenge: &mut Level2Challenge, factor: f64) {
    let n = challenge.network.num_nodes;
    let h = challenge.difficulty.effective_params().num_steps;
    let slack = challenge.network.slack_bus;

    for i in 0..n {
        if i != slack {
            for t in 0..h {
                challenge.exogenous_injections[i][t] *= factor;
            }
        }
    }
    // Rebalance slack
    for t in 0..h {
        let mut sum = 0.0;
        for i in 0..n {
            if i != slack {
                sum += challenge.exogenous_injections[i][t];
            }
        }
        challenge.exogenous_injections[slack][t] = -sum;
    }
}

/// Scale flow limits by a factor (on top of γ_cong already applied)
fn scale_flow_limits(challenge: &mut Level2Challenge, factor: f64) {
    for l in 0..challenge.network.num_lines {
        challenge.network.flow_limits[l] *= factor;
    }
}

// ============================================================================
// Analysis helpers
// ============================================================================

struct InstanceMetrics {
    exo_flow_util_mean: f64,     // avg |flow|/limit under zero storage
    exo_flow_util_max: f64,      // max |flow|/limit
    exo_congestion_frac: f64,    // fraction of (line,step) with |flow| > τ_cong * limit
    exo_violation_frac: f64,     // fraction with |flow| > limit (infeasible baseline)
    greedy_profit: Option<f64>,
    greedy_cong_steps: f64,      // fraction of steps with any congestion (from solution)
    decomp_profit: Option<f64>,
    decomp_cong_steps: f64,
    greedy_time_ms: f64,
    decomp_time_ms: f64,
    greedy_active_frac: f64,
    decomp_active_frac: f64,
}

fn analyze_instance(challenge: &Level2Challenge) -> (f64, f64, f64, f64) {
    let params = challenge.difficulty.effective_params();
    let h = params.num_steps;
    let m = challenge.batteries.len();

    let idle = PortfolioAction { actions: vec![SignedAction::idle(); m] };

    let mut util_sum = 0.0;
    let mut util_max: f64 = 0.0;
    let mut cong_count = 0;
    let mut violation_count = 0;
    let mut total_count = 0;

    for t in 0..h {
        let inj = challenge.compute_total_injections(&idle, t);
        let flows = challenge.network.compute_flows(&inj);
        for (l, &flow) in flows.iter().enumerate() {
            let limit = challenge.network.flow_limits[l];
            let ratio = flow.abs() / limit;
            util_sum += ratio;
            util_max = util_max.max(ratio);
            if ratio >= constants::TAU_CONG {
                cong_count += 1;
            }
            if ratio > 1.0 + constants::EPS_FLOW {
                violation_count += 1;
            }
            total_count += 1;
        }
    }

    let util_mean = util_sum / total_count as f64;
    let cong_frac = cong_count as f64 / total_count as f64;
    let viol_frac = violation_count as f64 / total_count as f64;

    (util_mean, util_max, cong_frac, viol_frac)
}

fn solve_and_analyze(challenge: &Level2Challenge) -> InstanceMetrics {
    let params = challenge.difficulty.effective_params();
    let h = params.num_steps;
    let m = challenge.batteries.len();

    let (exo_util_mean, exo_util_max, exo_cong_frac, exo_viol_frac) = analyze_instance(challenge);

    // Greedy
    let t0 = Instant::now();
    let greedy_result = level2_greedy::solve_challenge(challenge);
    let greedy_time = t0.elapsed().as_micros() as f64 / 1000.0;

    let (greedy_profit, greedy_cong, greedy_active) = match &greedy_result {
        Ok(Some(sol)) => {
            match challenge.verify_solution(sol) {
                Ok(p) => {
                    let (cong, active) = solution_metrics(challenge, sol);
                    (Some(p), cong, active)
                }
                Err(_) => (None, 0.0, 0.0),
            }
        }
        _ => (None, 0.0, 0.0),
    };

    // Decomposition
    let t0 = Instant::now();
    let decomp_result = level2_decomposition::solve_challenge(challenge);
    let decomp_time = t0.elapsed().as_micros() as f64 / 1000.0;

    let (decomp_profit, decomp_cong, decomp_active) = match &decomp_result {
        Ok(Some(sol)) => {
            match challenge.verify_solution(sol) {
                Ok(p) => {
                    let (cong, active) = solution_metrics(challenge, sol);
                    (Some(p), cong, active)
                }
                Err(_) => (None, 0.0, 0.0),
            }
        }
        _ => (None, 0.0, 0.0),
    };

    InstanceMetrics {
        exo_flow_util_mean: exo_util_mean,
        exo_flow_util_max: exo_util_max,
        exo_congestion_frac: exo_cong_frac,
        exo_violation_frac: exo_viol_frac,
        greedy_profit,
        greedy_cong_steps: greedy_cong,
        decomp_profit,
        decomp_cong_steps: decomp_cong,
        greedy_time_ms: greedy_time,
        decomp_time_ms: decomp_time,
        greedy_active_frac: greedy_active,
        decomp_active_frac: decomp_active,
    }
}

fn solution_metrics(challenge: &Level2Challenge, solution: &Level2Solution) -> (f64, f64) {
    let params = challenge.difficulty.effective_params();
    let h = params.num_steps;
    let m = challenge.batteries.len();

    let mut cong_steps = 0;
    let mut active_count = 0;

    for t in 0..h {
        let action = &solution.schedule[t];
        let inj = challenge.compute_total_injections(action, t);
        let flows = challenge.network.compute_flows(&inj);

        let has_cong = flows.iter().zip(challenge.network.flow_limits.iter())
            .any(|(&f, &lim)| f.abs() >= constants::TAU_CONG * lim);
        if has_cong { cong_steps += 1; }

        for b in 0..m {
            if action.actions[b].power_mw.abs() > 0.1 { active_count += 1; }
        }
    }

    (cong_steps as f64 / h as f64, active_count as f64 / (h * m) as f64)
}

fn stats(vals: &[f64]) -> (f64, f64) {
    if vals.is_empty() { return (0.0, 0.0); }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    (mean, var.sqrt())
}

// ============================================================================
// Experiment 1: Exogenous injection scaling sweep
// ============================================================================

fn experiment_exo_scaling() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 1: Exogenous Injection Scaling (Track 1)            ║");
    println!("║  Increases background flows to induce congestion                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let exo_factors = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0];
    let num_seeds = 5;

    println!("  {:>6} | {:>8} {:>8} {:>8} {:>8} | {:>10} {:>10} | {:>6}",
        "exo_x", "util%", "max_ut%", "cong%", "viol%", "Gr profit", "De profit", "gap%");
    println!("  {}", "-".repeat(90));

    for &factor in &exo_factors {
        let mut utils = Vec::new();
        let mut maxutils = Vec::new();
        let mut congs = Vec::new();
        let mut viols = Vec::new();
        let mut gps = Vec::new();
        let mut dps = Vec::new();

        for s in 0..num_seeds {
            let seed = make_seed(s);
            let difficulty = Level2Difficulty {
                track: Track::Track1,
                profit_threshold: -1e12,
                ..Default::default()
            };
            let mut challenge = Level2Challenge::generate_instance(seed, &difficulty).unwrap();
            scale_exogenous(&mut challenge, factor);

            let metrics = solve_and_analyze(&challenge);
            utils.push(metrics.exo_flow_util_mean * 100.0);
            maxutils.push(metrics.exo_flow_util_max * 100.0);
            congs.push(metrics.exo_congestion_frac * 100.0);
            viols.push(metrics.exo_violation_frac * 100.0);
            if let Some(p) = metrics.greedy_profit { gps.push(p); }
            if let Some(p) = metrics.decomp_profit { dps.push(p); }
        }

        let (um, _) = stats(&utils);
        let (mum, _) = stats(&maxutils);
        let (cm, _) = stats(&congs);
        let (vm, _) = stats(&viols);
        let (gm, _) = stats(&gps);
        let (dm, _) = stats(&dps);
        let gap = if dm.abs() > 1e-6 && !gps.is_empty() {
            format!("{:.0}", (1.0 - gm / dm) * 100.0)
        } else { "N/A".to_string() };

        println!("  {:>5.0}x | {:>7.1} {:>7.1} {:>7.2} {:>7.2} | {:>10.0} {:>10.0} | {:>6}",
            factor, um, mum, cm, vm, gm, dm, gap);
    }
    println!();
}

// ============================================================================
// Experiment 2: Flow limit scaling sweep
// ============================================================================

fn experiment_flow_limit_scaling() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 2: Flow Limit Scaling (Track 1)                     ║");
    println!("║  Tightens line limits to induce congestion                      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let limit_factors = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.10, 0.08];
    let num_seeds = 5;

    println!("  {:>8} | {:>8} {:>8} {:>8} {:>8} | {:>10} {:>10} | {:>6}",
        "lim_x", "util%", "max_ut%", "cong%", "viol%", "Gr profit", "De profit", "gap%");
    println!("  {}", "-".repeat(90));

    for &factor in &limit_factors {
        let mut utils = Vec::new();
        let mut maxutils = Vec::new();
        let mut congs = Vec::new();
        let mut viols = Vec::new();
        let mut gps = Vec::new();
        let mut dps = Vec::new();

        for s in 0..num_seeds {
            let seed = make_seed(s);
            let difficulty = Level2Difficulty {
                track: Track::Track1,
                profit_threshold: -1e12,
                ..Default::default()
            };
            let mut challenge = Level2Challenge::generate_instance(seed, &difficulty).unwrap();
            scale_flow_limits(&mut challenge, factor);

            let metrics = solve_and_analyze(&challenge);
            utils.push(metrics.exo_flow_util_mean * 100.0);
            maxutils.push(metrics.exo_flow_util_max * 100.0);
            congs.push(metrics.exo_congestion_frac * 100.0);
            viols.push(metrics.exo_violation_frac * 100.0);
            if let Some(p) = metrics.greedy_profit { gps.push(p); }
            if let Some(p) = metrics.decomp_profit { dps.push(p); }
        }

        let (um, _) = stats(&utils);
        let (mum, _) = stats(&maxutils);
        let (cm, _) = stats(&congs);
        let (vm, _) = stats(&viols);
        let (gm, _) = stats(&gps);
        let (dm, _) = stats(&dps);
        let gap = if dm.abs() > 1e-6 && !gps.is_empty() {
            format!("{:.0}", (1.0 - gm / dm) * 100.0)
        } else { "N/A".to_string() };

        println!("  {:>7.2}x | {:>7.1} {:>7.1} {:>7.2} {:>7.2} | {:>10.0} {:>10.0} | {:>6}",
            factor, um, mum, cm, vm, gm, dm, gap);
    }
    println!();
}

// ============================================================================
// Experiment 3: Combined scaling — exo UP + limits DOWN
// ============================================================================

fn experiment_combined_scaling() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 3: Combined Scaling (Track 1)                       ║");
    println!("║  exo ↑ AND limits ↓ simultaneously                             ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // (exo_factor, limit_factor)
    let combos: Vec<(f64, f64)> = vec![
        (1.0, 1.0),    // baseline
        (3.0, 0.5),    // moderate
        (4.0, 0.4),
        (5.0, 0.3),
        (6.0, 0.25),
        (8.0, 0.20),
        (10.0, 0.15),
        (5.0, 0.5),
        (5.0, 0.4),
        (5.0, 0.3),
        (5.0, 0.2),
    ];
    let num_seeds = 5;

    println!("  {:>5} {:>5} | {:>7} {:>7} {:>7} {:>7} | {:>9} {:>9} | {:>5} {:>6} {:>6}",
        "exo", "lim", "util%", "max%", "cong%", "viol%", "Gr $", "De $", "gap%", "GrAct", "DeAct");
    println!("  {}", "-".repeat(100));

    for (exo_f, lim_f) in &combos {
        let mut utils = Vec::new();
        let mut maxutils = Vec::new();
        let mut congs = Vec::new();
        let mut viols = Vec::new();
        let mut gps = Vec::new();
        let mut dps = Vec::new();
        let mut g_act = Vec::new();
        let mut d_act = Vec::new();

        for s in 0..num_seeds {
            let seed = make_seed(s);
            let difficulty = Level2Difficulty {
                track: Track::Track1,
                profit_threshold: -1e12,
                ..Default::default()
            };
            let mut challenge = Level2Challenge::generate_instance(seed, &difficulty).unwrap();
            scale_exogenous(&mut challenge, *exo_f);
            scale_flow_limits(&mut challenge, *lim_f);

            let metrics = solve_and_analyze(&challenge);
            utils.push(metrics.exo_flow_util_mean * 100.0);
            maxutils.push(metrics.exo_flow_util_max * 100.0);
            congs.push(metrics.exo_congestion_frac * 100.0);
            viols.push(metrics.exo_violation_frac * 100.0);
            if let Some(p) = metrics.greedy_profit { gps.push(p); }
            if let Some(p) = metrics.decomp_profit { dps.push(p); }
            g_act.push(metrics.greedy_active_frac * 100.0);
            d_act.push(metrics.decomp_active_frac * 100.0);
        }

        let (um, _) = stats(&utils);
        let (mum, _) = stats(&maxutils);
        let (cm, _) = stats(&congs);
        let (vm, _) = stats(&viols);
        let (gm, _) = stats(&gps);
        let (dm, _) = stats(&dps);
        let (gam, _) = stats(&g_act);
        let (dam, _) = stats(&d_act);
        let gap = if dm.abs() > 1e-6 && !gps.is_empty() {
            format!("{:.0}", (1.0 - gm / dm) * 100.0)
        } else { "N/A".to_string() };

        println!("  {:>4.0}x {:>4.2}x | {:>6.1} {:>6.1} {:>6.2} {:>6.2} | {:>9.0} {:>9.0} | {:>5} {:>5.1} {:>5.1}",
            exo_f, lim_f, um, mum, cm, vm, gm, dm, gap, gam, dam);
    }
    println!();
}

// ============================================================================
// Experiment 4: Apply best combo across all 5 tracks
// ============================================================================

fn experiment_all_tracks_tuned(exo_factor: f64, limit_factor: f64) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 4: All Tracks with tuned parameters                 ║");
    println!("║  exo_scale={:.0}x, limit_scale={:.2}x                           ", exo_factor, limit_factor);
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    let num_seeds = 8;

    println!("  {:>7} | {:>4} {:>4} {:>4} {:>4} | {:>7} {:>7} {:>7} | {:>9} {:>6} | {:>9} {:>6} | {:>5} {:>6}",
        "Track", "n", "L", "m", "H", "util%", "cong%", "viol%", "Gr $", "GrF%", "De $", "DeF%", "gap%", "V ms");
    println!("  {}", "-".repeat(115));

    for track in Track::all() {
        let params = track.parameters();
        let mut utils = Vec::new();
        let mut congs = Vec::new();
        let mut viols = Vec::new();
        let mut gps = Vec::new();
        let mut dps = Vec::new();
        let mut g_feas_count = 0usize;
        let mut d_feas_count = 0usize;
        let mut vtimes = Vec::new();
        let mut total = 0usize;

        for s in 0..num_seeds {
            let seed = make_seed(s);
            let difficulty = Level2Difficulty {
                track,
                profit_threshold: -1e12,
                ..Default::default()
            };
            let mut challenge = match Level2Challenge::generate_instance(seed, &difficulty) {
                Ok(c) => c,
                Err(_) => continue,
            };
            scale_exogenous(&mut challenge, exo_factor);
            scale_flow_limits(&mut challenge, limit_factor);

            total += 1;
            let (u, _, c, v) = analyze_instance(&challenge);
            utils.push(u * 100.0);
            congs.push(c * 100.0);
            viols.push(v * 100.0);

            // Greedy
            if let Ok(Some(sol)) = level2_greedy::solve_challenge(&challenge) {
                if let Ok(p) = challenge.verify_solution(&sol) {
                    gps.push(p);
                    g_feas_count += 1;
                }
            }

            // Decomposition
            if let Ok(Some(sol)) = level2_decomposition::solve_challenge(&challenge) {
                let t0 = Instant::now();
                if let Ok(p) = challenge.verify_solution(&sol) {
                    vtimes.push(t0.elapsed().as_micros() as f64 / 1000.0);
                    dps.push(p);
                    d_feas_count += 1;
                }
            }
        }

        let (um, _) = stats(&utils);
        let (cm, _) = stats(&congs);
        let (vm, _) = stats(&viols);
        let (gm, _) = stats(&gps);
        let (dm, _) = stats(&dps);
        let (vtm, _) = stats(&vtimes);

        let gf = if total > 0 { g_feas_count as f64 / total as f64 * 100.0 } else { 0.0 };
        let df = if total > 0 { d_feas_count as f64 / total as f64 * 100.0 } else { 0.0 };
        let gap = if dm.abs() > 1e-6 && !gps.is_empty() {
            format!("{:.0}", (1.0 - gm / dm) * 100.0)
        } else { "N/A".to_string() };

        println!("  {:>7} | {:>4} {:>4} {:>4} {:>4} | {:>6.1} {:>6.2} {:>6.2} | {:>9.0} {:>5.0}% | {:>9.0} {:>5.0}% | {:>5} {:>5.1}",
            format!("{:?}", track), params.num_nodes, params.num_lines, params.num_batteries, params.num_steps,
            um, cm, vm, gm, gf, dm, df, gap, vtm);
    }
    println!();
}

// ============================================================================
// Experiment 5: F_base sweep (what if nominal limit were lower?)
// ============================================================================

fn experiment_fbase_equivalents() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 5: Effective F_base sweep (Track 2, 5 seeds)        ║");
    println!("║  Testing what F_base *should* be for meaningful congestion      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Current F_base = 100 MW. We simulate lower F_base by scaling limits down.
    // After γ_cong=0.8, effective limit = F_base * 0.8 * (0.8-1.2)
    // So limit_factor effectively changes F_base.
    let fbase_equiv = [100.0, 50.0, 30.0, 20.0, 15.0, 10.0, 8.0, 5.0];
    let num_seeds = 5;

    println!("  {:>8} | {:>8} {:>8} {:>8} | {:>10} {:>6} | {:>10} {:>6} | {:>5}",
        "F_base", "util%", "cong%", "viol%", "Gr $", "GrF%", "De $", "DeF%", "gap%");
    println!("  {}", "-".repeat(90));

    for &fbase in &fbase_equiv {
        let limit_factor = fbase / 100.0; // Since nominal is 100 MW

        let mut utils = Vec::new();
        let mut congs = Vec::new();
        let mut viols = Vec::new();
        let mut gps = Vec::new();
        let mut dps = Vec::new();
        let mut g_feas = 0;
        let mut d_feas = 0;
        let mut total = 0;

        for s in 0..num_seeds {
            let seed = make_seed(s);
            let difficulty = Level2Difficulty {
                track: Track::Track2,
                profit_threshold: -1e12,
                ..Default::default()
            };
            let mut challenge = match Level2Challenge::generate_instance(seed, &difficulty) {
                Ok(c) => c,
                Err(_) => continue,
            };
            scale_flow_limits(&mut challenge, limit_factor);

            total += 1;
            let (u, _, c, v) = analyze_instance(&challenge);
            utils.push(u * 100.0);
            congs.push(c * 100.0);
            viols.push(v * 100.0);

            if let Ok(Some(sol)) = level2_greedy::solve_challenge(&challenge) {
                if let Ok(p) = challenge.verify_solution(&sol) { gps.push(p); g_feas += 1; }
            }
            if let Ok(Some(sol)) = level2_decomposition::solve_challenge(&challenge) {
                if let Ok(p) = challenge.verify_solution(&sol) { dps.push(p); d_feas += 1; }
            }
        }

        let (um, _) = stats(&utils);
        let (cm, _) = stats(&congs);
        let (vm, _) = stats(&viols);
        let (gm, _) = stats(&gps);
        let (dm, _) = stats(&dps);
        let gf = g_feas as f64 / total.max(1) as f64 * 100.0;
        let df = d_feas as f64 / total.max(1) as f64 * 100.0;
        let gap = if dm.abs() > 1e-6 && !gps.is_empty() {
            format!("{:.0}", (1.0 - gm / dm) * 100.0)
        } else { "N/A".to_string() };

        println!("  {:>7.0} | {:>7.1} {:>7.2} {:>7.2} | {:>10.0} {:>5.0}% | {:>10.0} {:>5.0}% | {:>5}",
            fbase, um, cm, vm, gm, gf, dm, df, gap);
    }
    println!();
}

// ============================================================================
// Experiment 6: p_base equivalent sweep (via exo scaling, Track 2)
// ============================================================================

fn experiment_pbase_equivalents() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 6: Effective p_base sweep (Track 2, 5 seeds)        ║");
    println!("║  Testing what p_base *should* be for meaningful congestion      ║");
    println!("╚══════════════════════════════════════════════════════════════════╝\n");

    // Current p_base = 50 MW. We simulate higher p_base by scaling exo up.
    let pbase_equiv = [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0];
    let num_seeds = 5;

    println!("  {:>8} | {:>8} {:>8} {:>8} | {:>10} {:>6} | {:>10} {:>6} | {:>5}",
        "p_base", "util%", "cong%", "viol%", "Gr $", "GrF%", "De $", "DeF%", "gap%");
    println!("  {}", "-".repeat(90));

    for &pbase in &pbase_equiv {
        let exo_factor = pbase / 50.0;

        let mut utils = Vec::new();
        let mut congs = Vec::new();
        let mut viols = Vec::new();
        let mut gps = Vec::new();
        let mut dps = Vec::new();
        let mut g_feas = 0;
        let mut d_feas = 0;
        let mut total = 0;

        for s in 0..num_seeds {
            let seed = make_seed(s);
            let difficulty = Level2Difficulty {
                track: Track::Track2,
                profit_threshold: -1e12,
                ..Default::default()
            };
            let mut challenge = match Level2Challenge::generate_instance(seed, &difficulty) {
                Ok(c) => c,
                Err(_) => continue,
            };
            scale_exogenous(&mut challenge, exo_factor);

            total += 1;
            let (u, _, c, v) = analyze_instance(&challenge);
            utils.push(u * 100.0);
            congs.push(c * 100.0);
            viols.push(v * 100.0);

            if let Ok(Some(sol)) = level2_greedy::solve_challenge(&challenge) {
                if let Ok(p) = challenge.verify_solution(&sol) { gps.push(p); g_feas += 1; }
            }
            if let Ok(Some(sol)) = level2_decomposition::solve_challenge(&challenge) {
                if let Ok(p) = challenge.verify_solution(&sol) { dps.push(p); d_feas += 1; }
            }
        }

        let (um, _) = stats(&utils);
        let (cm, _) = stats(&congs);
        let (vm, _) = stats(&viols);
        let (gm, _) = stats(&gps);
        let (dm, _) = stats(&dps);
        let gf = g_feas as f64 / total.max(1) as f64 * 100.0;
        let df = d_feas as f64 / total.max(1) as f64 * 100.0;
        let gap = if dm.abs() > 1e-6 && !gps.is_empty() {
            format!("{:.0}", (1.0 - gm / dm) * 100.0)
        } else { "N/A".to_string() };

        println!("  {:>7.0} | {:>7.1} {:>7.2} {:>7.2} | {:>10.0} {:>5.0}% | {:>10.0} {:>5.0}% | {:>5}",
            pbase, um, cm, vm, gm, gf, dm, df, gap);
    }
    println!();
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("\n{}", "=".repeat(70));
    println!("  Congestion Tuning Experiments");
    println!("  Finding parameter ranges that produce meaningful congestion");
    println!("{}\n", "=".repeat(70));

    let total_start = Instant::now();

    experiment_exo_scaling();
    experiment_flow_limit_scaling();
    experiment_combined_scaling();

    // Based on results from above, test best combos across all tracks
    // (Run these after reviewing initial results)
    experiment_fbase_equivalents();
    experiment_pbase_equivalents();

    // Apply the most promising combo to all 5 tracks
    println!("████████████████████████████████████████████████████████████████████");
    println!("  Applying tuned parameters to all 5 tracks");
    println!("████████████████████████████████████████████████████████████████████\n");

    // Test a few promising combos
    experiment_all_tracks_tuned(5.0, 0.3);
    experiment_all_tracks_tuned(3.0, 0.5);
    experiment_all_tracks_tuned(8.0, 0.2);

    let total_elapsed = total_start.elapsed();
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Total experiment time: {:.1?}", total_elapsed);
    println!("══════════════════════════════════════════════════════════════════");
}
