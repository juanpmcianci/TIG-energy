/*!
 * Example runner for TIG Energy Arbitrage V2 Challenge
 *
 * Demonstrates both Level 1 (single-asset) and Level 2 (portfolio) challenges.
 *
 * Run with: cargo run --example runner_v2 --release
 */

use tig_challenges::energy_arbitrage_v2::{
    Level1Challenge, Level1Difficulty, Level1Solution,
    Level2Challenge, Level2Difficulty, Level2Solution,
};
use tig_algorithms::energy_arbitrage_v2::{
    level1_greedy, level1_dp,
    level2_greedy, level2_decomposition,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     TIG Energy Arbitrage Challenge V2 - Example Runner       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    run_level1_examples();
    println!("\n{}\n", "═".repeat(66));
    run_level2_examples();
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

fn run_level2_examples() {
    println!("┌────────────────────────────────────────────────────────────────┐");
    println!("│  LEVEL 2: Portfolio Arbitrage on Constrained Network           │");
    println!("│  Spatial arbitrage with DC power flow constraints              │");
    println!("└────────────────────────────────────────────────────────────────┘\n");

    let difficulties = vec![
        ("Easy", Level2Difficulty {
            num_steps: 24,
            num_nodes: 4,
            num_batteries: 2,
            volatility: 0.15,
            tail_index: 4.0,
            congestion_factor: 1.0,  // No congestion
            heterogeneity: 0.1,
            congestion_premium: 5.0,
            profit_threshold: -10000.0,
        }),
        ("Medium", Level2Difficulty {
            num_steps: 24,
            num_nodes: 5,
            num_batteries: 3,
            volatility: 0.25,
            tail_index: 3.0,
            congestion_factor: 0.7,
            heterogeneity: 0.3,
            congestion_premium: 10.0,
            profit_threshold: -10000.0,
        }),
        ("Hard", Level2Difficulty {
            num_steps: 48,
            num_nodes: 8,
            num_batteries: 5,
            volatility: 0.35,
            tail_index: 2.5,
            congestion_factor: 0.5,  // High congestion
            heterogeneity: 0.5,
            congestion_premium: 20.0,
            profit_threshold: -10000.0,
        }),
    ];

    for (name, difficulty) in &difficulties {
        println!("--- Level 2: {} Difficulty ---", name);
        println!("  Network: {} nodes, {} lines (ring topology)", difficulty.num_nodes, difficulty.num_nodes);
        println!("  Batteries: {}, Heterogeneity: {:.0}%", difficulty.num_batteries, difficulty.heterogeneity * 100.0);
        println!("  Volatility: {:.0}%, Congestion Factor: {:.0}%\n",
            difficulty.volatility * 100.0,
            difficulty.congestion_factor * 100.0);

        let seed = [42u8; 32];
        let challenge = match Level2Challenge::generate_instance(seed, difficulty) {
            Ok(c) => c,
            Err(e) => {
                println!("  Failed to generate instance: {}\n", e);
                continue;
            }
        };

        // Print network info
        println!("  Network topology: Ring with {} nodes", challenge.network.num_nodes);
        println!("  Line flow limits: {:.1} MW each", challenge.network.flow_limits[0]);
        println!("  Battery placements:");
        for (b, placed) in challenge.batteries.iter().enumerate() {
            println!("    Battery {}: Node {}, {:.0} MWh, {:.0}/{:.0} MW",
                b, placed.node,
                placed.spec.capacity_mwh,
                placed.spec.power_charge_mw,
                placed.spec.power_discharge_mw);
        }
        println!();

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

                            // Show per-battery statistics
                            for b in 0..challenge.batteries.len() {
                                let charges: f64 = solution.schedule.iter()
                                    .map(|s| s.battery_actions[b].charge_mw)
                                    .sum();
                                let discharges: f64 = solution.schedule.iter()
                                    .map(|s| s.battery_actions[b].discharge_mw)
                                    .sum();
                                println!("    Battery {}: charged {:.1} MWh, discharged {:.1} MWh",
                                    b, charges, discharges);
                            }
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
