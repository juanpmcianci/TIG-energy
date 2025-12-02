/*!
 * Level 2 Greedy Solver
 *
 * Simple heuristic for portfolio arbitrage on constrained network.
 * Each battery operates independently with flow-aware adjustments.
 */

use anyhow::Result;
use tig_challenges::energy_arbitrage_v2::{
    Action, Level2Challenge, Level2Solution, PortfolioAction,
};

/// Configuration for Level 2 greedy solver
#[derive(Debug, Clone)]
pub struct GreedyConfig {
    /// Charge threshold (fraction of DA price)
    pub charge_threshold: f64,
    /// Discharge threshold (fraction of DA price)
    pub discharge_threshold: f64,
    /// Flow safety margin (fraction of limit)
    pub flow_safety_margin: f64,
}

impl Default for GreedyConfig {
    fn default() -> Self {
        Self {
            charge_threshold: 0.95,
            discharge_threshold: 1.05,
            flow_safety_margin: 0.9,
        }
    }
}

/// Solve Level 2 challenge using greedy policy
pub fn solve_challenge(challenge: &Level2Challenge) -> Result<Option<Level2Solution>> {
    solve_with_config(challenge, &GreedyConfig::default())
}

/// Solve with custom configuration
pub fn solve_with_config(
    challenge: &Level2Challenge,
    config: &GreedyConfig,
) -> Result<Option<Level2Solution>> {
    let t = challenge.difficulty.num_steps;
    let m = challenge.batteries.len();

    // Track SOC for each battery
    let mut socs: Vec<f64> = challenge
        .batteries
        .iter()
        .map(|b| b.spec.soc_initial_mwh)
        .collect();

    let mut schedule = Vec::with_capacity(t);

    for step in 0..t {
        // Determine greedy actions for each battery
        let mut candidate_actions = Vec::with_capacity(m);

        for (b, placed) in challenge.batteries.iter().enumerate() {
            let node = placed.node;
            let battery = &placed.spec;
            let da_price = challenge.day_ahead_prices[node][step];
            let soc = socs[b];

            // Simple greedy decision
            let action = if da_price < config.charge_threshold * average_da_price(challenge, step) {
                // Low price node - charge
                let headroom = battery.soc_max_mwh - soc;
                let max_charge = headroom / battery.efficiency_charge;
                let charge_power = max_charge.min(battery.power_charge_mw);
                if charge_power > 0.1 {
                    Action::charge(charge_power)
                } else {
                    Action::idle()
                }
            } else if da_price > config.discharge_threshold * average_da_price(challenge, step) {
                // High price node - discharge
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

            candidate_actions.push(action);
        }

        // Check flow constraints and adjust if needed
        let portfolio_action = adjust_for_flow_limits(challenge, &candidate_actions, config);

        // Update SOCs
        for (b, placed) in challenge.batteries.iter().enumerate() {
            let action = &portfolio_action.battery_actions[b];
            socs[b] = socs[b]
                + placed.spec.efficiency_charge * action.charge_mw
                - action.discharge_mw / placed.spec.efficiency_discharge;
            socs[b] = socs[b].clamp(placed.spec.soc_min_mwh, placed.spec.soc_max_mwh);
        }

        schedule.push(portfolio_action);
    }

    Ok(Some(Level2Solution { schedule }))
}

/// Compute average DA price across all nodes
fn average_da_price(challenge: &Level2Challenge, step: usize) -> f64 {
    let sum: f64 = challenge
        .day_ahead_prices
        .iter()
        .map(|prices| prices[step])
        .sum();
    sum / challenge.network.num_nodes as f64
}

/// Adjust actions to respect flow limits
fn adjust_for_flow_limits(
    challenge: &Level2Challenge,
    candidate_actions: &[Action],
    config: &GreedyConfig,
) -> PortfolioAction {
    let mut actions = candidate_actions.to_vec();

    // Compute initial injections and flows
    let initial_portfolio = PortfolioAction {
        battery_actions: actions.clone(),
    };
    let injections = challenge.compute_injections(&initial_portfolio);
    let flows = challenge.network.compute_flows(&injections);

    // Check for violations
    let mut violation = false;
    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l] * config.flow_safety_margin;
        if flow.abs() > limit {
            violation = true;
            break;
        }
    }

    if violation {
        // Scale down all actions proportionally
        let max_ratio = flows
            .iter()
            .zip(challenge.network.flow_limits.iter())
            .map(|(&flow, &limit)| {
                let safe_limit = limit * config.flow_safety_margin;
                if flow.abs() > 1e-6 {
                    safe_limit / flow.abs()
                } else {
                    f64::INFINITY
                }
            })
            .fold(f64::INFINITY, f64::min);

        if max_ratio < 1.0 && max_ratio > 0.0 {
            for action in &mut actions {
                action.charge_mw *= max_ratio;
                action.discharge_mw *= max_ratio;
            }
        } else if max_ratio <= 0.0 {
            // Fallback: all idle
            for action in &mut actions {
                *action = Action::idle();
            }
        }
    }

    PortfolioAction {
        battery_actions: actions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tig_challenges::energy_arbitrage_v2::Level2Difficulty;

    #[test]
    fn test_level2_greedy() {
        let difficulty = Level2Difficulty::default();
        let seed = [42u8; 32];
        let challenge = Level2Challenge::generate_instance(seed, &difficulty).unwrap();

        let solution = solve_challenge(&challenge).unwrap().unwrap();
        assert_eq!(solution.schedule.len(), 24);

        // Verify solution validity (may not meet profit threshold)
        match challenge.verify_solution(&solution) {
            Ok(profit) => println!("Level 2 greedy profit: {:.2}", profit),
            Err(e) => {
                let err_str = e.to_string();
                if !err_str.contains("below threshold") {
                    panic!("Unexpected error: {}", e);
                }
            }
        }
    }
}
