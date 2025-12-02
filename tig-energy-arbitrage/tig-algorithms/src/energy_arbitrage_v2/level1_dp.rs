/*!
 * Level 1 Dynamic Programming Solver
 *
 * Model Predictive Control with DP lookahead for single-asset arbitrage.
 * Uses Monte Carlo sampling to estimate expected value under action commitment.
 */

use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use tig_challenges::energy_arbitrage_v2::{
    Action, Level1Challenge, Level1Solution, TranscriptEntry,
};

/// Configuration for DP solver
#[derive(Debug, Clone)]
pub struct DPConfig {
    /// Number of SOC discretization points
    pub soc_grid_size: usize,
    /// Number of Monte Carlo samples for value estimation
    pub num_samples: usize,
    /// Lookahead horizon for MPC
    pub lookahead: usize,
    /// Number of action discretization points
    pub action_grid_size: usize,
}

impl Default for DPConfig {
    fn default() -> Self {
        Self {
            soc_grid_size: 51,
            num_samples: 50,
            lookahead: 6,
            action_grid_size: 11,
        }
    }
}

/// Solve Level 1 challenge using DP with MPC
pub fn solve_challenge(challenge: &Level1Challenge) -> Result<Option<Level1Solution>> {
    solve_with_config(challenge, &DPConfig::default())
}

/// Solve with custom configuration
pub fn solve_with_config(
    challenge: &Level1Challenge,
    config: &DPConfig,
) -> Result<Option<Level1Solution>> {
    let t = challenge.difficulty.num_steps;
    let battery = &challenge.battery;

    // Build SOC grid
    let soc_grid: Vec<f64> = (0..config.soc_grid_size)
        .map(|i| {
            battery.soc_min_mwh
                + i as f64 * (battery.soc_max_mwh - battery.soc_min_mwh)
                    / (config.soc_grid_size - 1) as f64
        })
        .collect();

    // Build action grid (charge and discharge separately)
    let action_grid: Vec<Action> = {
        let mut actions = vec![Action::idle()];
        for i in 1..=config.action_grid_size {
            let frac = i as f64 / config.action_grid_size as f64;
            actions.push(Action::charge(battery.power_charge_mw * frac));
            actions.push(Action::discharge(battery.power_discharge_mw * frac));
        }
        actions
    };

    let mut transcript = Vec::with_capacity(t);
    let mut current_seed = challenge.seed;
    let mut soc = battery.soc_initial_mwh;

    for step in 0..t {
        let da_price = challenge.day_ahead_prices[step];
        let rt_price = challenge.generate_rt_price(da_price, &current_seed);

        // Find best action using lookahead
        let horizon = (step + config.lookahead).min(t);
        let best_action = find_best_action(
            challenge,
            config,
            step,
            horizon,
            soc,
            &current_seed,
            &soc_grid,
            &action_grid,
        );

        // Apply action
        let new_soc = challenge.apply_action(soc, &best_action, 1.0);
        let profit = challenge.compute_step_profit(&best_action, rt_price, 1.0);

        transcript.push(TranscriptEntry {
            time_step: step,
            action: best_action.clone(),
            soc_mwh: new_soc,
            seed: current_seed,
            rt_price,
            profit,
        });

        current_seed = challenge.commit_action(&current_seed, &best_action, step, new_soc);
        soc = new_soc;
    }

    Ok(Some(Level1Solution { transcript }))
}

/// Find best action using DP lookahead with Monte Carlo sampling
fn find_best_action(
    challenge: &Level1Challenge,
    config: &DPConfig,
    current_step: usize,
    horizon: usize,
    current_soc: f64,
    current_seed: &[u8; 32],
    soc_grid: &[f64],
    action_grid: &[Action],
) -> Action {
    let battery = &challenge.battery;
    let steps_ahead = horizon - current_step;

    if steps_ahead == 0 {
        return Action::idle();
    }

    // Evaluate each candidate action
    let mut best_action = Action::idle();
    let mut best_value = f64::NEG_INFINITY;

    for action in action_grid {
        // Check feasibility
        if !is_feasible(action, current_soc, battery) {
            continue;
        }

        // Compute immediate reward (deterministic given current seed)
        let da_price = challenge.day_ahead_prices[current_step];
        let rt_price = challenge.generate_rt_price(da_price, current_seed);
        let immediate_reward = challenge.compute_step_profit(action, rt_price, 1.0);

        // Compute expected future value via Monte Carlo
        let new_soc = challenge.apply_action(current_soc, action, 1.0);
        let new_seed = challenge.commit_action(current_seed, action, current_step, new_soc);

        let future_value = if steps_ahead > 1 {
            estimate_future_value(
                challenge,
                config,
                current_step + 1,
                horizon,
                new_soc,
                &new_seed,
                soc_grid,
                action_grid,
            )
        } else {
            0.0
        };

        let total_value = immediate_reward + future_value;
        if total_value > best_value {
            best_value = total_value;
            best_action = action.clone();
        }
    }

    best_action
}

/// Estimate future value using simplified DP
fn estimate_future_value(
    challenge: &Level1Challenge,
    config: &DPConfig,
    start_step: usize,
    horizon: usize,
    start_soc: f64,
    start_seed: &[u8; 32],
    _soc_grid: &[f64],
    _action_grid: &[Action],
) -> f64 {
    let steps = horizon - start_step;

    if steps == 0 {
        return 0.0;
    }

    // Simplified: use average expected prices and greedy lookahead
    let mut value = 0.0;
    let mut soc = start_soc;
    let mut seed = *start_seed;

    // Use multiple samples to estimate expected value
    let mut rng = SmallRng::from_seed(seed);

    for step in start_step..horizon {
        let da_price = challenge.day_ahead_prices[step];

        // Sample a few scenarios and average
        let mut step_value = 0.0;
        let sample_count = config.num_samples.min(10);

        for _ in 0..sample_count {
            // Perturb seed slightly for different scenarios
            let sample_seed: [u8; 32] = rng.gen();
            let rt_price = challenge.generate_rt_price(da_price, &sample_seed);

            // Find best greedy action for this scenario
            let action = greedy_action(challenge, soc, rt_price, da_price);
            let profit = challenge.compute_step_profit(&action, rt_price, 1.0);
            step_value += profit;
        }

        value += step_value / sample_count as f64;

        // Advance state using greedy action on expected price
        let expected_price = da_price; // Simplified
        let action = greedy_action(challenge, soc, expected_price, da_price);
        soc = challenge.apply_action(soc, &action, 1.0);
        seed = challenge.commit_action(&seed, &action, step, soc);
    }

    value
}

/// Check if action is feasible given current SOC
fn is_feasible(action: &Action, soc: f64, battery: &tig_challenges::energy_arbitrage_v2::BatterySpec) -> bool {
    let new_soc = soc
        + battery.efficiency_charge * action.charge_mw
        - action.discharge_mw / battery.efficiency_discharge;

    new_soc >= battery.soc_min_mwh - 1e-6 && new_soc <= battery.soc_max_mwh + 1e-6
}

/// Simple greedy action selection
fn greedy_action(
    challenge: &Level1Challenge,
    soc: f64,
    rt_price: f64,
    da_price: f64,
) -> Action {
    let battery = &challenge.battery;

    if rt_price < 0.95 * da_price {
        let headroom = battery.soc_max_mwh - soc;
        let max_charge = headroom / battery.efficiency_charge;
        let charge_power = max_charge.min(battery.power_charge_mw);
        if charge_power > 0.1 {
            return Action::charge(charge_power);
        }
    } else if rt_price > 1.05 * da_price {
        let available = soc - battery.soc_min_mwh;
        let max_discharge = available * battery.efficiency_discharge;
        let discharge_power = max_discharge.min(battery.power_discharge_mw);
        if discharge_power > 0.1 {
            return Action::discharge(discharge_power);
        }
    }

    Action::idle()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tig_challenges::energy_arbitrage_v2::Level1Difficulty;

    #[test]
    fn test_dp_solver() {
        let difficulty = Level1Difficulty {
            num_steps: 12, // Shorter for faster test
            ..Default::default()
        };
        let seed = [42u8; 32];
        let challenge = Level1Challenge::generate_instance(seed, &difficulty).unwrap();

        let config = DPConfig {
            soc_grid_size: 21,
            num_samples: 10,
            lookahead: 4,
            action_grid_size: 5,
        };

        let solution = solve_with_config(&challenge, &config).unwrap().unwrap();
        assert_eq!(solution.transcript.len(), 12);
    }
}
