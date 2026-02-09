/*!
 * Example runner for TIG Energy Arbitrage Network Challenge
 *
 * Demonstrates the portfolio arbitrage challenge with 5 tracks
 * as specified in tig_level_2_spec.tex.
 *
 * Run with: cargo run --example runner_v2 --release
 */

use tig_challenges::energy_arbitrage_v2::{
    Level2Challenge, Level2Difficulty, Level2Solution,
    Track, constants,
};
use tig_algorithms::energy_arbitrage_v2::{
    level2_greedy, level2_decomposition,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     TIG Energy Arbitrage Network Challenge - Runner          ║");
    println!("║     Portfolio Arbitrage with 5 Tracks                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Default constants from spec:");
    println!("  Δt = {} hours (15 min)", constants::DELTA_T);
    println!("  η^c = η^d = {:.0}%", constants::ETA_CHARGE * 100.0);
    println!("  κ_tx = ${:.2}/MWh, κ_deg = ${:.2}", constants::KAPPA_TX, constants::KAPPA_DEG);
    println!("  ρ_sp = {:.2} (spatial correlation)", constants::RHO_SPATIAL);
    println!("  τ_cong = {:.2} (congestion threshold)", constants::TAU_CONG);
    println!("  λ_min = ${:.0}, λ_max = ${:.0}\n", constants::LAMBDA_MIN, constants::LAMBDA_MAX);

    // Run each track
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
                        Ok(better) => {
                            println!("  {}: PASS", solver_name);
                            println!("    Better-than-baseline: {:.6}", better);
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
