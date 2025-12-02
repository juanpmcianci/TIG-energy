/*!
 * Example runner for TIG Energy Arbitrage Challenge
 * 
 * Run with: cargo run --example runner --release
 */

use tig_challenges::energy_arbitrage::{Challenge, Difficulty};
use tig_algorithms::energy_arbitrage::{greedy, mpc_dp, sddp};

fn main() {
    println!("=== TIG Energy Arbitrage Challenge ===\n");
    
    // Define difficulty levels
    let difficulties = vec![
        ("Easy", Difficulty {
            num_steps: 24,
            num_scenarios: 50,
            volatility_percent: 15,
            tail_risk_percent: 3,
            better_than_baseline: 5,
        }),
        ("Medium", Difficulty {
            num_steps: 24,
            num_scenarios: 100,
            volatility_percent: 25,
            tail_risk_percent: 5,
            better_than_baseline: 15,
        }),
        ("Hard", Difficulty {
            num_steps: 48,
            num_scenarios: 200,
            volatility_percent: 35,
            tail_risk_percent: 10,
            better_than_baseline: 25,
        }),
    ];
    
    for (name, difficulty) in &difficulties {
        println!("--- Difficulty: {} ---", name);
        println!("  Steps: {}, Scenarios: {}, Volatility: {}%, Tail: {}%",
            difficulty.num_steps, difficulty.num_scenarios,
            difficulty.volatility_percent, difficulty.tail_risk_percent);
        println!("  Required improvement over baseline: {}%\n", difficulty.better_than_baseline);
        
        // Generate instance
        let seed = [42u8; 32];
        let challenge = match Challenge::generate_instance(seed, difficulty) {
            Ok(c) => c,
            Err(e) => {
                println!("  Failed to generate instance: {}\n", e);
                continue;
            }
        };
        
        println!("  Baseline profit: ${:.2}", challenge.baseline_profit);
        println!("  Required profit: ${:.2}\n", 
            challenge.baseline_profit * (1.0 + difficulty.better_than_baseline as f64 / 100.0));
        
        // Test each solver
        let solvers: Vec<(&str, fn(&Challenge) -> anyhow::Result<Option<tig_challenges::energy_arbitrage::Solution>>)> = vec![
            ("Greedy", greedy::solve_challenge),
            ("MPC-DP", mpc_dp::solve_challenge),
            ("SDDP", sddp::solve_challenge),
        ];
        
        for (solver_name, solver_fn) in &solvers {
            let start = std::time::Instant::now();
            
            match solver_fn(&challenge) {
                Ok(Some(solution)) => {
                    let elapsed = start.elapsed();
                    match challenge.verify_solution(&solution) {
                        Ok(profit) => {
                            let improvement = (profit / challenge.baseline_profit - 1.0) * 100.0;
                            println!("  {}: PASS", solver_name);
                            println!("    Profit: ${:.2} ({:+.1}% vs baseline)", profit, improvement);
                            println!("    Time: {:.2?}", elapsed);
                        }
                        Err(e) => {
                            println!("  {}: FAIL - {}", solver_name, e);
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
