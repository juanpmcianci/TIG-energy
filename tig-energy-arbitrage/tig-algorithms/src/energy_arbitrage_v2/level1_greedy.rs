/*!
 * Level 1 Greedy Solver
 *
 * Simple threshold-based policy for single-asset temporal arbitrage.
 * Serves as a baseline algorithm.
 */

use anyhow::Result;
use tig_challenges::energy_arbitrage_v2::{
    Action, Level1Challenge, Level1Solution, TranscriptEntry,
};

/// Configuration for greedy solver
#[derive(Debug, Clone)]
pub struct GreedyConfig {
    /// Charge when RT price is below this fraction of DA price
    pub charge_threshold: f64,
    /// Discharge when RT price is above this fraction of DA price
    pub discharge_threshold: f64,
}

impl Default for GreedyConfig {
    fn default() -> Self {
        Self {
            charge_threshold: 0.95,
            discharge_threshold: 1.05,
        }
    }
}

/// Solve Level 1 challenge using greedy threshold policy
pub fn solve_challenge(challenge: &Level1Challenge) -> Result<Option<Level1Solution>> {
    solve_with_config(challenge, &GreedyConfig::default())
}

/// Solve with custom configuration
pub fn solve_with_config(
    challenge: &Level1Challenge,
    config: &GreedyConfig,
) -> Result<Option<Level1Solution>> {
    let t = challenge.difficulty.num_steps;
    let battery = &challenge.battery;

    let mut transcript = Vec::with_capacity(t);
    let mut current_seed = challenge.seed;
    let mut soc = battery.soc_initial_mwh;

    for step in 0..t {
        let da_price = challenge.day_ahead_prices[step];
        let rt_price = challenge.generate_rt_price(da_price, &current_seed);

        // Greedy decision based on price comparison
        let action = if rt_price < config.charge_threshold * da_price {
            // Price is low - charge if we have room
            let headroom = battery.soc_max_mwh - soc;
            let max_charge = headroom / battery.efficiency_charge;
            let charge_power = max_charge.min(battery.power_charge_mw);
            if charge_power > 0.1 {
                Action::charge(charge_power)
            } else {
                Action::idle()
            }
        } else if rt_price > config.discharge_threshold * da_price {
            // Price is high - discharge if we have energy
            let available = soc - battery.soc_min_mwh;
            let max_discharge = available * battery.efficiency_discharge;
            let discharge_power = max_discharge.min(battery.power_discharge_mw);
            if discharge_power > 0.1 {
                Action::discharge(discharge_power)
            } else {
                Action::idle()
            }
        } else {
            Action::idle()
        };

        // Apply action
        let new_soc = challenge.apply_action(soc, &action, 1.0);
        let profit = challenge.compute_step_profit(&action, rt_price, 1.0);

        // Record transcript entry
        transcript.push(TranscriptEntry {
            time_step: step,
            action: action.clone(),
            soc_mwh: new_soc,
            seed: current_seed,
            rt_price,
            profit,
        });

        // Commit action for next step
        current_seed = challenge.commit_action(&current_seed, &action, step, new_soc);
        soc = new_soc;
    }

    Ok(Some(Level1Solution { transcript }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tig_challenges::energy_arbitrage_v2::Level1Difficulty;

    #[test]
    fn test_greedy_solver() {
        let difficulty = Level1Difficulty::default();
        let seed = [42u8; 32];
        let challenge = Level1Challenge::generate_instance(seed, &difficulty).unwrap();

        let solution = solve_challenge(&challenge).unwrap().unwrap();
        assert_eq!(solution.transcript.len(), 24);

        // Verify the solution
        let result = challenge.verify_solution(&solution);
        // Note: may fail profit threshold but should be structurally valid
        match result {
            Ok(profit) => println!("Greedy profit: {:.2}", profit),
            Err(e) => {
                // Check if it's just a threshold issue
                let err_str = e.to_string();
                if !err_str.contains("below threshold") {
                    panic!("Unexpected error: {}", e);
                }
            }
        }
    }
}
