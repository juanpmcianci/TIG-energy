/*!
 * Example runner for TIG Energy Arbitrage V2 Challenge
 *
 * Demonstrates both Level 1 (single-asset) and Level 2 (portfolio) challenges.
 * Level 2 now includes the 5-track system as specified in tig_level_2_spec.tex.
 *
 * Run with: cargo run --example runner_v2 --release
 */

use tig_challenges::energy_arbitrage_v2::{
    Level1Challenge, Level1Difficulty, Level1Solution,
    Level2Challenge, Level2Difficulty, Level2Solution,
    Track, constants,
};
use tig_algorithms::energy_arbitrage_v2::{
    level1_greedy, level1_dp,
    level2_greedy, level2_decomposition,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     TIG Energy Arbitrage Challenge V2 - Example Runner       ║");
    println!("║     Implements tig_level_2_spec.tex with 5 Tracks            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    run_level1_examples();
    println!("\n{}\n", "═".repeat(66));
    run_level2_track_examples();
}

fn run_level1_examples() {
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│  LEVEL 1: Single-Asset Temporal Arbitrage                      │");
    println!("│  Action-committed pricing prevents lookahead exploitation      │");
    println!("└────────────────────────────────────────────────────────────────┘\n");

    let difficulties = vec![
        ("Easy", Level1Difficulty {
            num_steps: 24,
            volatility: 0.15,
            tail_index: 4.0,
            transaction_cost: 0.3,
            degradation_cost: 0.5,
            profit_threshold: -1000.0, // Relaxed for demo
        }),
        ("Medium", Level1Difficulty {
            num_steps: 24,
            volatility: 0.25,
            tail_index: 3.0,
            transaction_cost: 0.5,
            degradation_cost: 1.0,
            profit_threshold: -1000.0,
        }),
        ("Hard", Level1Difficulty {
            num_steps: 48,
            volatility: 0.35,
            tail_index: 2.5,
            transaction_cost: 0.7,
            degradation_cost: 1.5,
            profit_threshold: -1000.0,
        }),
    ];

    for (name, difficulty) in &difficulties {
        println!("--- Level 1: {} Difficulty ---", name);
        println!("  Steps: {}, Volatility: {:.0}%, Tail Index: {:.1}",
            difficulty.num_steps,
            difficulty.volatility * 100.0,
            difficulty.tail_index);
        println!("  Transaction cost: ${:.2}/MWh, Degradation: ${:.2}/MWh\n",
            difficulty.transaction_cost,
            difficulty.degradation_cost);

        let seed = [42u8; 32];
        let challenge = match Level1Challenge::generate_instance(seed, difficulty) {
            Ok(c) => c,
            Err(e) => {
                println!("  Failed to generate instance: {}\n", e);
                continue;
            }
        };

        // Print some instance info
        let da_min = challenge.day_ahead_prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let da_max = challenge.day_ahead_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let da_avg: f64 = challenge.day_ahead_prices.iter().sum::<f64>() / challenge.day_ahead_prices.len() as f64;
        println!("  Day-ahead prices: min=${:.1}, avg=${:.1}, max=${:.1}", da_min, da_avg, da_max);
        println!("  Battery: {:.0} MWh, {:.0} MW charge/discharge, {:.0}% efficiency\n",
            challenge.battery.capacity_mwh,
            challenge.battery.power_charge_mw,
            challenge.battery.efficiency_charge * 100.0);

        // Test solvers
        let solvers: Vec<(&str, fn(&Level1Challenge) -> anyhow::Result<Option<Level1Solution>>)> = vec![
            ("Greedy", level1_greedy::solve_challenge),
            ("DP-MPC", level1_dp::solve_challenge),
        ];

        for (solver_name, solver_fn) in &solvers {
            let start = std::time::Instant::now();

            match solver_fn(&challenge) {
                Ok(Some(solution)) => {
                    let elapsed = start.elapsed();
                    match challenge.verify_solution(&solution) {
                        Ok(profit) => {
                            println!("  {}: PASS", solver_name);
                            println!("    Total Profit: ${:.2}", profit);
                            println!("    Time: {:.2?}", elapsed);

                            // Show some action statistics
                            let charges: f64 = solution.transcript.iter()
                                .map(|e| e.action.charge_mw)
                                .sum();
                            let discharges: f64 = solution.transcript.iter()
                                .map(|e| e.action.discharge_mw)
                                .sum();
                            println!("    Total charged: {:.1} MWh, discharged: {:.1} MWh",
                                charges, discharges);
                        }
                        Err(e) => {
                            println!("  {}: VERIFICATION FAILED - {}", solver_name, e);
                        }
                    }
                }
                Ok(None) => {
                    println!("  {}: No solution found", solver_name);
                }
                Err(e) => {
                    println!("  {}: Error - {}", solver_name, e);
                }
            }
        }

        println!();
    }
}

fn run_level2_track_examples() {
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│  LEVEL 2: Portfolio Arbitrage on Constrained Network           │");
    println!("│  Five tracks with increasing difficulty per spec               │");
    println!("└────────────────────────────────────────────────────────────────┘\n");

    println!("Default constants from spec:");
    println!("  Δt = {} hours (15 min)", constants::DELTA_T);
    println!("  η^c = η^d = {:.0}%", constants::ETA_CHARGE * 100.0);
    println!("  κ_tx = ${:.2}/MWh, κ_deg = ${:.2}", constants::KAPPA_TX, constants::KAPPA_DEG);
    println!("  ρ_sp = {:.2} (spatial correlation)", constants::RHO_SPATIAL);
    println!("  τ_cong = {:.2} (congestion threshold)", constants::TAU_CONG);
    println!("  λ_min = ${:.0}, λ_max = ${:.0}\n", constants::LAMBDA_MIN, constants::LAMBDA_MAX);

    // Run shorter versions of each track for demo purposes
    for track in Track::all() {
        let params = track.parameters();

        println!("═══════════════════════════════════════════════════════════════");
        println!("  TRACK {:?}", track);
        println!("═══════════════════════════════════════════════════════════════");
        println!("  Network: {} nodes, {} lines", params.num_nodes, params.num_lines);
        println!("  Batteries: {}, Heterogeneity: {:.0}%", params.num_batteries, params.heterogeneity * 100.0);
        println!("  Horizon: {} steps ({:.1} hours)", params.num_steps, params.num_steps as f64 * constants::DELTA_T);
        println!("  γ_cong = {:.2} (line limit scaling)", params.gamma_cong);
        println!("  σ = {:.2} (volatility), α = {:.1} (tail)", params.sigma, params.alpha);
        println!("  ρ_jump = {:.2} (jump probability)\n", params.rho_jump);

        // Use shorter horizon for demo
        let demo_steps = 24.min(params.num_steps);

        let difficulty = Level2Difficulty {
            track,
            num_steps: Some(demo_steps),
            profit_threshold: -100000.0, // Relaxed for demo
            ..Default::default()
        };

        let seed = [42u8; 32];
        let challenge = match Level2Challenge::generate_instance(seed, &difficulty) {
            Ok(c) => c,
            Err(e) => {
                println!("  Failed to generate instance: {}\n", e);
                continue;
            }
        };

        // Print network info
        println!("  Generated network: {} nodes, {} lines",
                 challenge.network.num_nodes, challenge.network.num_lines);
        println!("  Flow limits: {:.1} - {:.1} MW",
                 challenge.network.flow_limits.iter().cloned().fold(f64::INFINITY, f64::min),
                 challenge.network.flow_limits.iter().cloned().fold(0.0, f64::max));

        // Print battery info
        let capacities: Vec<f64> = challenge.batteries.iter().map(|b| b.spec.capacity_mwh).collect();
        let min_cap = capacities.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_cap = capacities.iter().cloned().fold(0.0, f64::max);
        println!("  Battery capacities: {:.1} - {:.1} MWh (heterogeneity factor: {:.2}x)",
                 min_cap, max_cap, max_cap / min_cap);

        // Print price info
        let all_prices: Vec<f64> = challenge.day_ahead_prices.iter()
            .flat_map(|p| p.iter().cloned())
            .collect();
        let price_min = all_prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let price_max = all_prices.iter().cloned().fold(0.0, f64::max);
        let price_avg: f64 = all_prices.iter().sum::<f64>() / all_prices.len() as f64;
        println!("  DA prices: ${:.1} - ${:.1} (avg ${:.1})\n", price_min, price_max, price_avg);

        // Test solvers
        let solvers: Vec<(&str, fn(&Level2Challenge) -> anyhow::Result<Option<Level2Solution>>)> = vec![
            ("Greedy", level2_greedy::solve_challenge),
            ("Decomposition", level2_decomposition::solve_challenge),
        ];

        for (solver_name, solver_fn) in &solvers {
            let start = std::time::Instant::now();

            match solver_fn(&challenge) {
                Ok(Some(solution)) => {
                    let elapsed = start.elapsed();
                    match challenge.verify_solution(&solution) {
                        Ok(profit) => {
                            println!("  {}: PASS", solver_name);
                            println!("    Portfolio Profit: ${:.2}", profit);
                            println!("    Time: {:.2?}", elapsed);

                            // Show aggregate statistics
                            let mut total_charge = 0.0;
                            let mut total_discharge = 0.0;
                            for action in &solution.schedule {
                                for sa in &action.actions {
                                    let (c, d) = sa.decompose();
                                    total_charge += c;
                                    total_discharge += d;
                                }
                            }
                            println!("    Total charged: {:.1} MW, discharged: {:.1} MW",
                                     total_charge, total_discharge);
                        }
                        Err(e) => {
                            println!("  {}: VERIFICATION FAILED - {}", solver_name, e);
                        }
                    }
                }
                Ok(None) => {
                    println!("  {}: No solution found", solver_name);
                }
                Err(e) => {
                    println!("  {}: Error - {}", solver_name, e);
                }
            }
        }

        println!();
    }

    // Summary of track ladder
    println!("═══════════════════════════════════════════════════════════════");
    println!("  TRACK LADDER SUMMARY (from spec)");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Track 1: Correctness baseline - small network, nominal limits");
    println!("  Track 2: Meaningful congestion - tighter limits, more stochasticity");
    println!("  Track 3: Multi-day horizon - larger scale, frequent spikes");
    println!("  Track 4: Dense network - frequent congestion, heavy tails");
    println!("  Track 5: Capstone - largest scale, tightest limits, heaviest tails\n");
}
