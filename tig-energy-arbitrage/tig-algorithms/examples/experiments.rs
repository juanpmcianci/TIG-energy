/*!
 * Parameter Range & Difficulty Progression Experiments
 *
 * Tests how different parameter settings affect:
 * - Solver profit
 * - Solve time
 * - Greedy-to-DP gap (optimality gap proxy)
 * - Solution feasibility rates
 * - Profit variance across seeds
 *
 * Run with: cargo run --example experiments --release
 */

use tig_challenges::energy_arbitrage_v2::{
    Level1Challenge, Level1Difficulty, Level1Solution,
    Level2Challenge, Level2Difficulty, Level2Solution,
};
use tig_algorithms::energy_arbitrage_v2::{
    level1_greedy, level1_dp,
    level2_greedy, level2_decomposition,
};
use std::time::Instant;

// ============================================================================
// Experiment Infrastructure
// ============================================================================

#[derive(Debug, Clone)]
struct L1Result {
    seed_idx: usize,
    greedy_profit: Option<f64>,
    greedy_time_us: u128,
    dp_profit: Option<f64>,
    dp_time_us: u128,
}

#[derive(Debug, Clone)]
struct L2Result {
    seed_idx: usize,
    greedy_profit: Option<f64>,
    greedy_time_us: u128,
    decomp_profit: Option<f64>,
    decomp_time_us: u128,
}

fn make_seed(idx: usize) -> [u8; 32] {
    let mut seed = [0u8; 32];
    let bytes = (idx as u64).to_le_bytes();
    seed[..8].copy_from_slice(&bytes);
    // Add some entropy mixing
    seed[8] = 0xDE;
    seed[9] = 0xAD;
    seed[10] = 0xBE;
    seed[11] = 0xEF;
    seed
}

fn run_l1_experiment(difficulty: &Level1Difficulty, num_seeds: usize) -> Vec<L1Result> {
    let mut results = Vec::new();
    for s in 0..num_seeds {
        let seed = make_seed(s);
        let challenge = match Level1Challenge::generate_instance(seed, difficulty) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Greedy
        let t0 = Instant::now();
        let greedy_sol = level1_greedy::solve_challenge(&challenge);
        let greedy_time = t0.elapsed().as_micros();
        let greedy_profit = greedy_sol.ok().flatten().and_then(|sol| {
            challenge.verify_solution(&sol).ok()
        });

        // DP
        let t0 = Instant::now();
        let dp_sol = level1_dp::solve_challenge(&challenge);
        let dp_time = t0.elapsed().as_micros();
        let dp_profit = dp_sol.ok().flatten().and_then(|sol| {
            challenge.verify_solution(&sol).ok()
        });

        results.push(L1Result {
            seed_idx: s,
            greedy_profit,
            greedy_time_us: greedy_time,
            dp_profit,
            dp_time_us: dp_time,
        });
    }
    results
}

fn run_l2_experiment(difficulty: &Level2Difficulty, num_seeds: usize) -> Vec<L2Result> {
    let mut results = Vec::new();
    for s in 0..num_seeds {
        let seed = make_seed(s);
        let challenge = match Level2Challenge::generate_instance(seed, difficulty) {
            Ok(c) => c,
            Err(_) => continue,
        };

        // Greedy
        let t0 = Instant::now();
        let greedy_sol = level2_greedy::solve_challenge(&challenge);
        let greedy_time = t0.elapsed().as_micros();
        let greedy_profit = greedy_sol.ok().flatten().and_then(|sol| {
            challenge.verify_solution(&sol).ok()
        });

        // Decomposition
        let t0 = Instant::now();
        let decomp_sol = level2_decomposition::solve_challenge(&challenge);
        let decomp_time = t0.elapsed().as_micros();
        let decomp_profit = decomp_sol.ok().flatten().and_then(|sol| {
            challenge.verify_solution(&sol).ok()
        });

        results.push(L2Result {
            seed_idx: s,
            greedy_profit,
            greedy_time_us: greedy_time,
            decomp_profit,
            decomp_time_us: decomp_time,
        });
    }
    results
}

// ============================================================================
// Statistics helpers
// ============================================================================

struct Stats {
    count: usize,
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    feasible_rate: f64,
}

fn compute_stats(values: &[Option<f64>]) -> Stats {
    let feasible: Vec<f64> = values.iter().filter_map(|v| *v).collect();
    let n = feasible.len();
    let total = values.len();
    if n == 0 {
        return Stats { count: 0, mean: 0.0, std: 0.0, min: 0.0, max: 0.0, feasible_rate: 0.0 };
    }
    let mean = feasible.iter().sum::<f64>() / n as f64;
    let var = feasible.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    Stats {
        count: n,
        mean,
        std: var.sqrt(),
        min: feasible.iter().cloned().fold(f64::INFINITY, f64::min),
        max: feasible.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        feasible_rate: n as f64 / total as f64,
    }
}

fn print_stats(label: &str, stats: &Stats) {
    if stats.count == 0 {
        println!("    {}: NO FEASIBLE SOLUTIONS", label);
    } else {
        println!("    {}: mean=${:.1}, std=${:.1}, min=${:.1}, max=${:.1}, feasible={:.0}%",
            label, stats.mean, stats.std, stats.min, stats.max, stats.feasible_rate * 100.0);
    }
}

fn print_time_stats(label: &str, times_us: &[u128]) {
    if times_us.is_empty() { return; }
    let mean = times_us.iter().sum::<u128>() as f64 / times_us.len() as f64;
    if mean > 1000.0 {
        println!("    {} avg time: {:.2} ms", label, mean / 1000.0);
    } else {
        println!("    {} avg time: {:.0} us", label, mean);
    }
}

// ============================================================================
// Experiment 1: Level 1 - Volatility sweep
// ============================================================================

fn experiment_l1_volatility() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 1: Level 1 — Volatility Sweep                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let volatilities = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50];
    let num_seeds = 20;

    println!("  {:>8} | {:>12} {:>12} | {:>12} {:>12} | {:>8}",
        "sigma", "Greedy mean", "Greedy feas", "DP mean", "DP feas", "Gap");
    println!("  {}", "-".repeat(80));

    for &vol in &volatilities {
        let diff = Level1Difficulty {
            num_steps: 24,
            volatility: vol,
            tail_index: 3.5,
            transaction_cost: 0.5,
            degradation_cost: 1.0,
            profit_threshold: -1e9,
        };

        let results = run_l1_experiment(&diff, num_seeds);
        let greedy_stats = compute_stats(&results.iter().map(|r| r.greedy_profit).collect::<Vec<_>>());
        let dp_stats = compute_stats(&results.iter().map(|r| r.dp_profit).collect::<Vec<_>>());

        let gap = if dp_stats.mean.abs() > 1e-6 && greedy_stats.count > 0 {
            1.0 - greedy_stats.mean / dp_stats.mean
        } else {
            f64::NAN
        };

        println!("  {:>7.0}% | {:>11.1} {:>11.0}% | {:>11.1} {:>11.0}% | {:>7.1}%",
            vol * 100.0,
            greedy_stats.mean, greedy_stats.feasible_rate * 100.0,
            dp_stats.mean, dp_stats.feasible_rate * 100.0,
            gap * 100.0);
    }
    println!();
}

// ============================================================================
// Experiment 2: Level 1 - Tail index sweep
// ============================================================================

fn experiment_l1_tail_index() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 2: Level 1 — Tail Index Sweep                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let tail_indices = [2.2, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0];
    let num_seeds = 20;

    println!("  {:>8} | {:>12} {:>12} | {:>12} {:>12} | {:>8}",
        "alpha", "Greedy mean", "Greedy std", "DP mean", "DP std", "Gap");
    println!("  {}", "-".repeat(80));

    for &alpha in &tail_indices {
        let diff = Level1Difficulty {
            num_steps: 24,
            volatility: 0.20,
            tail_index: alpha,
            transaction_cost: 0.5,
            degradation_cost: 1.0,
            profit_threshold: -1e9,
        };

        let results = run_l1_experiment(&diff, num_seeds);
        let greedy_stats = compute_stats(&results.iter().map(|r| r.greedy_profit).collect::<Vec<_>>());
        let dp_stats = compute_stats(&results.iter().map(|r| r.dp_profit).collect::<Vec<_>>());

        let gap = if dp_stats.mean.abs() > 1e-6 && greedy_stats.count > 0 {
            1.0 - greedy_stats.mean / dp_stats.mean
        } else {
            f64::NAN
        };

        println!("  {:>8.1} | {:>11.1} {:>11.1} | {:>11.1} {:>11.1} | {:>7.1}%",
            alpha,
            greedy_stats.mean, greedy_stats.std,
            dp_stats.mean, dp_stats.std,
            gap * 100.0);
    }
    println!();
}

// ============================================================================
// Experiment 3: Level 1 - Transaction & degradation cost sweep
// ============================================================================

fn experiment_l1_frictions() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 3: Level 1 — Friction Cost Sweep                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let tx_costs = [0.0, 0.2, 0.5, 1.0, 2.0, 3.0];
    let deg_costs = [0.0, 0.5, 1.0, 2.0, 5.0];
    let num_seeds = 15;

    println!("  {:>6} {:>6} | {:>12} | {:>12} | {:>8}",
        "tx_c", "deg_c", "Greedy mean", "DP mean", "Gap");
    println!("  {}", "-".repeat(60));

    for &tx in &tx_costs {
        for &deg in &[0.5, 2.0] {  // Reduced combos for speed
            let diff = Level1Difficulty {
                num_steps: 24,
                volatility: 0.20,
                tail_index: 3.0,
                transaction_cost: tx,
                degradation_cost: deg,
                profit_threshold: -1e9,
            };

            let results = run_l1_experiment(&diff, num_seeds);
            let greedy_stats = compute_stats(&results.iter().map(|r| r.greedy_profit).collect::<Vec<_>>());
            let dp_stats = compute_stats(&results.iter().map(|r| r.dp_profit).collect::<Vec<_>>());

            let gap = if dp_stats.mean.abs() > 1e-6 && greedy_stats.count > 0 {
                1.0 - greedy_stats.mean / dp_stats.mean
            } else {
                f64::NAN
            };

            println!("  {:>6.1} {:>6.1} | {:>11.1} | {:>11.1} | {:>7.1}%",
                tx, deg, greedy_stats.mean, dp_stats.mean, gap * 100.0);
        }
    }
    println!();
}

// ============================================================================
// Experiment 4: Level 1 - Num steps (horizon) sweep
// ============================================================================

fn experiment_l1_horizon() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 4: Level 1 — Horizon Length Sweep               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let steps = [12, 24, 48, 72, 96];
    let num_seeds = 15;

    println!("  {:>6} | {:>12} {:>10} | {:>12} {:>10} | {:>8}",
        "steps", "Greedy mean", "Greedy ms", "DP mean", "DP ms", "Gap");
    println!("  {}", "-".repeat(75));

    for &n in &steps {
        let diff = Level1Difficulty {
            num_steps: n,
            volatility: 0.20,
            tail_index: 3.0,
            transaction_cost: 0.5,
            degradation_cost: 1.0,
            profit_threshold: -1e9,
        };

        let results = run_l1_experiment(&diff, num_seeds);
        let greedy_stats = compute_stats(&results.iter().map(|r| r.greedy_profit).collect::<Vec<_>>());
        let dp_stats = compute_stats(&results.iter().map(|r| r.dp_profit).collect::<Vec<_>>());
        let greedy_time = results.iter().map(|r| r.greedy_time_us).sum::<u128>() as f64 / results.len() as f64 / 1000.0;
        let dp_time = results.iter().map(|r| r.dp_time_us).sum::<u128>() as f64 / results.len() as f64 / 1000.0;

        let gap = if dp_stats.mean.abs() > 1e-6 && greedy_stats.count > 0 {
            1.0 - greedy_stats.mean / dp_stats.mean
        } else {
            f64::NAN
        };

        println!("  {:>6} | {:>11.1} {:>9.2} | {:>11.1} {:>9.2} | {:>7.1}%",
            n, greedy_stats.mean, greedy_time, dp_stats.mean, dp_time, gap * 100.0);
    }
    println!();
}

// ============================================================================
// Experiment 5: Level 2 - Congestion factor sweep
// ============================================================================

fn experiment_l2_congestion() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 5: Level 2 — Congestion Factor Sweep            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let congestion_factors = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3];
    let num_seeds = 15;

    println!("  {:>8} | {:>12} {:>10} | {:>12} {:>10} | {:>8}",
        "gamma", "Greedy mean", "Gr feas%", "Decomp mean", "De feas%", "Gap");
    println!("  {}", "-".repeat(80));

    for &gamma in &congestion_factors {
        let diff = Level2Difficulty {
            num_steps: 24,
            num_nodes: 6,
            num_batteries: 3,
            volatility: 0.20,
            tail_index: 3.0,
            congestion_factor: gamma,
            heterogeneity: 0.3,
            congestion_premium: 10.0,
            profit_threshold: -1e9,
        };

        let results = run_l2_experiment(&diff, num_seeds);
        let greedy_stats = compute_stats(&results.iter().map(|r| r.greedy_profit).collect::<Vec<_>>());
        let decomp_stats = compute_stats(&results.iter().map(|r| r.decomp_profit).collect::<Vec<_>>());

        let gap = if decomp_stats.mean.abs() > 1e-6 && greedy_stats.count > 0 && decomp_stats.count > 0 {
            1.0 - greedy_stats.mean / decomp_stats.mean
        } else {
            f64::NAN
        };

        println!("  {:>7.0}% | {:>11.1} {:>9.0}% | {:>11.1} {:>9.0}% | {:>7.1}%",
            gamma * 100.0,
            greedy_stats.mean, greedy_stats.feasible_rate * 100.0,
            decomp_stats.mean, decomp_stats.feasible_rate * 100.0,
            gap * 100.0);
    }
    println!();
}

// ============================================================================
// Experiment 6: Level 2 - Network scale sweep
// ============================================================================

fn experiment_l2_scale() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 6: Level 2 — Network Scale Sweep                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let configs: Vec<(usize, usize, &str)> = vec![
        (4, 2, "Tiny"),
        (6, 3, "Small"),
        (10, 5, "Medium"),
        (15, 8, "Large"),
        (20, 10, "XL"),
    ];
    let num_seeds = 10;

    println!("  {:>8} {:>5} {:>5} | {:>12} {:>10} | {:>12} {:>10}",
        "Config", "Nodes", "Bats", "Greedy mean", "Gr ms", "Decomp mean", "De ms");
    println!("  {}", "-".repeat(80));

    for (nodes, bats, label) in &configs {
        let diff = Level2Difficulty {
            num_steps: 24,
            num_nodes: *nodes,
            num_batteries: *bats,
            volatility: 0.20,
            tail_index: 3.0,
            congestion_factor: 0.7,
            heterogeneity: 0.3,
            congestion_premium: 10.0,
            profit_threshold: -1e9,
        };

        let results = run_l2_experiment(&diff, num_seeds);
        let greedy_stats = compute_stats(&results.iter().map(|r| r.greedy_profit).collect::<Vec<_>>());
        let decomp_stats = compute_stats(&results.iter().map(|r| r.decomp_profit).collect::<Vec<_>>());
        let greedy_time = results.iter().map(|r| r.greedy_time_us).sum::<u128>() as f64 / results.len() as f64 / 1000.0;
        let decomp_time = results.iter().map(|r| r.decomp_time_us).sum::<u128>() as f64 / results.len() as f64 / 1000.0;

        println!("  {:>8} {:>5} {:>5} | {:>11.1} {:>9.2} | {:>11.1} {:>9.2}",
            label, nodes, bats,
            greedy_stats.mean, greedy_time,
            decomp_stats.mean, decomp_time);
    }
    println!();
}

// ============================================================================
// Experiment 7: Level 2 - Heterogeneity sweep
// ============================================================================

fn experiment_l2_heterogeneity() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 7: Level 2 — Battery Heterogeneity Sweep        ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let heterogeneities = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0];
    let num_seeds = 15;

    println!("  {:>8} | {:>12} | {:>12} | {:>8}",
        "h", "Greedy mean", "Decomp mean", "Gap");
    println!("  {}", "-".repeat(55));

    for &h in &heterogeneities {
        let diff = Level2Difficulty {
            num_steps: 24,
            num_nodes: 6,
            num_batteries: 3,
            volatility: 0.20,
            tail_index: 3.0,
            congestion_factor: 0.7,
            heterogeneity: h,
            congestion_premium: 10.0,
            profit_threshold: -1e9,
        };

        let results = run_l2_experiment(&diff, num_seeds);
        let greedy_stats = compute_stats(&results.iter().map(|r| r.greedy_profit).collect::<Vec<_>>());
        let decomp_stats = compute_stats(&results.iter().map(|r| r.decomp_profit).collect::<Vec<_>>());

        let gap = if decomp_stats.mean.abs() > 1e-6 && greedy_stats.count > 0 && decomp_stats.count > 0 {
            1.0 - greedy_stats.mean / decomp_stats.mean
        } else {
            f64::NAN
        };

        println!("  {:>7.0}% | {:>11.1} | {:>11.1} | {:>7.1}%",
            h * 100.0, greedy_stats.mean, decomp_stats.mean, gap * 100.0);
    }
    println!();
}

// ============================================================================
// Experiment 8: Track-based difficulty progression (Level 2)
// ============================================================================

fn experiment_l2_tracks() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 8: Level 2 — Track Difficulty Progression       ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Track definitions from tig_level_2_spec
    let tracks: Vec<(&str, Level2Difficulty)> = vec![
        ("Track 1", Level2Difficulty {
            num_steps: 96,
            num_nodes: 20,
            num_batteries: 10,
            volatility: 0.10,
            tail_index: 4.0,
            congestion_factor: 1.0,
            heterogeneity: 0.2,
            congestion_premium: 5.0,
            profit_threshold: -1e9,
        }),
        ("Track 2", Level2Difficulty {
            num_steps: 96,
            num_nodes: 40,
            num_batteries: 20,
            volatility: 0.15,
            tail_index: 3.5,
            congestion_factor: 0.8,
            heterogeneity: 0.4,
            congestion_premium: 8.0,
            profit_threshold: -1e9,
        }),
        ("Track 3", Level2Difficulty {
            num_steps: 192,
            num_nodes: 80,
            num_batteries: 40,
            volatility: 0.20,
            tail_index: 3.0,
            congestion_factor: 0.6,
            heterogeneity: 0.6,
            congestion_premium: 12.0,
            profit_threshold: -1e9,
        }),
        ("Track 4", Level2Difficulty {
            num_steps: 192,
            num_nodes: 100,
            num_batteries: 60,
            volatility: 0.25,
            tail_index: 2.7,
            congestion_factor: 0.5,
            heterogeneity: 0.8,
            congestion_premium: 18.0,
            profit_threshold: -1e9,
        }),
        ("Track 5", Level2Difficulty {
            num_steps: 192,
            num_nodes: 150,
            num_batteries: 100,
            volatility: 0.30,
            tail_index: 2.5,
            congestion_factor: 0.4,
            heterogeneity: 1.0,
            congestion_premium: 25.0,
            profit_threshold: -1e9,
        }),
    ];

    let num_seeds = 5; // Fewer seeds since tracks get large

    for (name, diff) in &tracks {
        println!("--- {} ---", name);
        println!("  N={}, L={}, B={}, T={}, sigma={:.0}%, alpha={:.1}, gamma={:.0}%, h={:.0}%",
            diff.num_nodes, diff.num_nodes, diff.num_batteries, diff.num_steps,
            diff.volatility * 100.0, diff.tail_index,
            diff.congestion_factor * 100.0, diff.heterogeneity * 100.0);

        let results = run_l2_experiment(diff, num_seeds);
        let greedy_stats = compute_stats(&results.iter().map(|r| r.greedy_profit).collect::<Vec<_>>());
        let decomp_stats = compute_stats(&results.iter().map(|r| r.decomp_profit).collect::<Vec<_>>());
        let greedy_time = results.iter().map(|r| r.greedy_time_us).sum::<u128>() as f64 / results.len().max(1) as f64 / 1000.0;
        let decomp_time = results.iter().map(|r| r.decomp_time_us).sum::<u128>() as f64 / results.len().max(1) as f64 / 1000.0;

        print_stats("Greedy", &greedy_stats);
        println!("    Greedy avg time: {:.2} ms", greedy_time);
        print_stats("Decomp", &decomp_stats);
        println!("    Decomp avg time: {:.2} ms", decomp_time);

        if greedy_stats.count > 0 && decomp_stats.count > 0 && decomp_stats.mean.abs() > 1e-6 {
            let gap = 1.0 - greedy_stats.mean / decomp_stats.mean;
            println!("    Greedy-Decomp Gap: {:.1}%", gap * 100.0);
        }

        // Profit CV
        if decomp_stats.count > 1 && decomp_stats.mean.abs() > 1e-6 {
            let cv = decomp_stats.std / decomp_stats.mean.abs();
            println!("    Decomp Profit CV: {:.3}", cv);
        }

        println!();
    }
}

// ============================================================================
// Experiment 9: Level 1 - Combined difficulty ladder
// ============================================================================

fn experiment_l1_difficulty_ladder() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  EXPERIMENT 9: Level 1 — Combined Difficulty Ladder         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let ladder: Vec<(&str, Level1Difficulty)> = vec![
        ("D1-Easy", Level1Difficulty {
            num_steps: 24, volatility: 0.10, tail_index: 4.5,
            transaction_cost: 0.2, degradation_cost: 0.3, profit_threshold: -1e9,
        }),
        ("D2-Med", Level1Difficulty {
            num_steps: 24, volatility: 0.20, tail_index: 3.5,
            transaction_cost: 0.5, degradation_cost: 1.0, profit_threshold: -1e9,
        }),
        ("D3-Hard", Level1Difficulty {
            num_steps: 48, volatility: 0.30, tail_index: 3.0,
            transaction_cost: 0.7, degradation_cost: 1.5, profit_threshold: -1e9,
        }),
        ("D4-VHard", Level1Difficulty {
            num_steps: 48, volatility: 0.40, tail_index: 2.5,
            transaction_cost: 1.0, degradation_cost: 2.0, profit_threshold: -1e9,
        }),
        ("D5-Extreme", Level1Difficulty {
            num_steps: 96, volatility: 0.50, tail_index: 2.2,
            transaction_cost: 1.5, degradation_cost: 3.0, profit_threshold: -1e9,
        }),
    ];

    let num_seeds = 20;

    println!("  {:>10} | {:>10} {:>10} {:>8} | {:>10} {:>10} {:>8} | {:>6}",
        "Level", "Gr mean", "Gr std", "Gr ms", "DP mean", "DP std", "DP ms", "Gap");
    println!("  {}", "-".repeat(90));

    for (name, diff) in &ladder {
        let results = run_l1_experiment(diff, num_seeds);
        let greedy_stats = compute_stats(&results.iter().map(|r| r.greedy_profit).collect::<Vec<_>>());
        let dp_stats = compute_stats(&results.iter().map(|r| r.dp_profit).collect::<Vec<_>>());
        let greedy_time = results.iter().map(|r| r.greedy_time_us).sum::<u128>() as f64 / results.len() as f64 / 1000.0;
        let dp_time = results.iter().map(|r| r.dp_time_us).sum::<u128>() as f64 / results.len() as f64 / 1000.0;

        let gap = if dp_stats.mean.abs() > 1e-6 && greedy_stats.count > 0 {
            1.0 - greedy_stats.mean / dp_stats.mean
        } else {
            f64::NAN
        };

        println!("  {:>10} | {:>9.1} {:>9.1} {:>7.2} | {:>9.1} {:>9.1} {:>7.2} | {:>5.1}%",
            name,
            greedy_stats.mean, greedy_stats.std, greedy_time,
            dp_stats.mean, dp_stats.std, dp_time,
            gap * 100.0);
    }
    println!();
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("\n{}", "=".repeat(66));
    println!("  TIG Energy Arbitrage — Parameter & Difficulty Experiments");
    println!("{}\n", "=".repeat(66));

    let total_start = Instant::now();

    // Level 1 experiments
    experiment_l1_volatility();
    experiment_l1_tail_index();
    experiment_l1_frictions();
    experiment_l1_horizon();
    experiment_l1_difficulty_ladder();

    // Level 2 experiments
    experiment_l2_congestion();
    experiment_l2_scale();
    experiment_l2_heterogeneity();
    experiment_l2_tracks();

    let total_elapsed = total_start.elapsed();
    println!("══════════════════════════════════════════════════════════════════");
    println!("  Total experiment time: {:.1?}", total_elapsed);
    println!("══════════════════════════════════════════════════════════════════");
}
