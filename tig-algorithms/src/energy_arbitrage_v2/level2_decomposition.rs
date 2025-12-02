/*!
 * Level 2 Benders Decomposition Solver
 *
 * Coordinates portfolio dispatch using decomposition approach:
 * - Master problem: determines battery schedules
 * - Subproblem: checks network flow feasibility
 */

use anyhow::Result;
use tig_challenges::energy_arbitrage_v2::{
    Action, Level2Challenge, Level2Solution, PortfolioAction,
};

/// Configuration for decomposition solver
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Number of LP refinement steps per iteration
    pub refinement_steps: usize,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-4,
            refinement_steps: 5,
        }
    }
}

/// Solve Level 2 challenge using simplified Benders decomposition
pub fn solve_challenge(challenge: &Level2Challenge) -> Result<Option<Level2Solution>> {
    solve_with_config(challenge, &DecompositionConfig::default())
}

/// Solve with custom configuration
pub fn solve_with_config(
    challenge: &Level2Challenge,
    config: &DecompositionConfig,
) -> Result<Option<Level2Solution>> {
    let t = challenge.difficulty.num_steps;

    // Initialize with greedy solution
    let mut schedule = initial_schedule(challenge);

    // Iterative refinement
    for _iter in 0..config.max_iterations {
        let mut improved = false;

        for step in 0..t {
            // Try to improve this step's schedule
            let current_portfolio = &schedule[step];
            let improved_portfolio = improve_step(challenge, step, current_portfolio, &schedule);

            // Check if improvement is significant
            let old_value = evaluate_portfolio(challenge, step, current_portfolio);
            let new_value = evaluate_portfolio(challenge, step, &improved_portfolio);

            if new_value > old_value + config.tolerance {
                schedule[step] = improved_portfolio;
                improved = true;
            }
        }

        if !improved {
            break;
        }
    }

    // Final flow feasibility check and adjustment
    schedule = ensure_feasibility(challenge, schedule);

    Ok(Some(Level2Solution { schedule }))
}

/// Generate initial schedule using node-wise greedy
fn initial_schedule(challenge: &Level2Challenge) -> Vec<PortfolioAction> {
    let t = challenge.difficulty.num_steps;
    let m = challenge.batteries.len();

    let mut socs: Vec<f64> = challenge
        .batteries
        .iter()
        .map(|b| b.spec.soc_initial_mwh)
        .collect();

    let mut schedule = Vec::with_capacity(t);

    for step in 0..t {
        let mut actions = Vec::with_capacity(m);

        for (b, placed) in challenge.batteries.iter().enumerate() {
            let node = placed.node;
            let battery = &placed.spec;
            let da_price = challenge.day_ahead_prices[node][step];
            let avg_price = average_da_price(challenge, step);
            let soc = socs[b];

            // Compute safe bounds for actions
            let headroom = (battery.soc_max_mwh - soc).max(0.0);
            let max_safe_charge = (headroom / battery.efficiency_charge).min(battery.power_charge_mw).max(0.0);

            let available = (soc - battery.soc_min_mwh).max(0.0);
            let max_safe_discharge = (available * battery.efficiency_discharge).min(battery.power_discharge_mw).max(0.0);

            // Price-based action with spread consideration
            let action = if da_price < 0.9 * avg_price && max_safe_charge > 0.1 {
                // Good opportunity to charge at this node
                Action::charge(max_safe_charge)
            } else if da_price > 1.1 * avg_price && max_safe_discharge > 0.1 {
                // Good opportunity to discharge at this node
                Action::discharge(max_safe_discharge)
            } else {
                // Look at absolute price level
                let price_pctile = price_percentile(challenge, step, node);
                if price_pctile < 0.3 && max_safe_charge > 0.1 {
                    Action::charge(max_safe_charge * 0.5)
                } else if price_pctile > 0.7 && max_safe_discharge > 0.1 {
                    Action::discharge(max_safe_discharge * 0.5)
                } else {
                    Action::idle()
                }
            };

            actions.push(action.clone());

            // Update SOC with clamping
            let new_soc = soc
                + battery.efficiency_charge * action.charge_mw
                - action.discharge_mw / battery.efficiency_discharge;
            socs[b] = new_soc.clamp(battery.soc_min_mwh, battery.soc_max_mwh);
        }

        schedule.push(PortfolioAction {
            battery_actions: actions,
        });
    }

    schedule
}

/// Try to improve a single step's portfolio
fn improve_step(
    challenge: &Level2Challenge,
    step: usize,
    current: &PortfolioAction,
    full_schedule: &[PortfolioAction],
) -> PortfolioAction {
    let m = challenge.batteries.len();

    // Compute current SOCs at this step
    let socs = compute_socs_at_step(challenge, full_schedule, step);

    // Try adjustments for each battery
    let mut best_actions = current.battery_actions.clone();
    let mut best_value = evaluate_portfolio(challenge, step, current);

    for b in 0..m {
        let placed = &challenge.batteries[b];
        let battery = &placed.spec;
        let soc = socs[b];

        // Try different action levels
        for power_frac in [0.0, 0.25, 0.5, 0.75, 1.0].iter() {
            // Try charging - respect SOC upper bound
            let headroom = (battery.soc_max_mwh - soc).max(0.0);
            let max_charge = (headroom / battery.efficiency_charge).min(battery.power_charge_mw).max(0.0);
            let charge_action = Action::charge(max_charge * power_frac);

            // Try discharging - respect SOC lower bound
            let available = (soc - battery.soc_min_mwh).max(0.0);
            let max_discharge =
                (available * battery.efficiency_discharge).min(battery.power_discharge_mw).max(0.0);
            let discharge_action = Action::discharge(max_discharge * power_frac);

            for candidate_action in [Action::idle(), charge_action, discharge_action].iter() {
                // Verify SOC feasibility for this action
                let new_soc = soc
                    + battery.efficiency_charge * candidate_action.charge_mw
                    - candidate_action.discharge_mw / battery.efficiency_discharge;

                if new_soc < battery.soc_min_mwh - 1e-6 || new_soc > battery.soc_max_mwh + 1e-6 {
                    continue; // Skip infeasible
                }

                let mut trial_actions = best_actions.clone();
                trial_actions[b] = candidate_action.clone();

                let trial_portfolio = PortfolioAction {
                    battery_actions: trial_actions.clone(),
                };

                // Check flow feasibility
                let injections = challenge.compute_injections(&trial_portfolio);
                let flows = challenge.network.compute_flows(&injections);
                if challenge.network.check_flow_limits(&flows).is_some() {
                    continue; // Skip infeasible
                }

                let value = evaluate_portfolio(challenge, step, &trial_portfolio);
                if value > best_value {
                    best_value = value;
                    best_actions = trial_actions;
                }
            }
        }
    }

    PortfolioAction {
        battery_actions: best_actions,
    }
}

/// Evaluate expected profit of a portfolio action at a step
fn evaluate_portfolio(
    challenge: &Level2Challenge,
    step: usize,
    portfolio: &PortfolioAction,
) -> f64 {
    let mut profit = 0.0;

    for (b, placed) in challenge.batteries.iter().enumerate() {
        let action = &portfolio.battery_actions[b];
        let node = placed.node;
        let price = challenge.day_ahead_prices[node][step]; // Use DA as estimate

        let c = action.charge_mw;
        let d = action.discharge_mw;

        let revenue = (d - c) * price;
        let tx_cost = challenge.frictions.transaction_cost_per_mwh * (c + d);
        let dod = (d / placed.spec.capacity_mwh).powf(challenge.frictions.degradation_exponent);
        let deg_cost = challenge.frictions.degradation_cost_per_mwh * dod;

        profit += revenue - tx_cost - deg_cost;
    }

    profit
}

/// Compute SOCs at a given step based on schedule
fn compute_socs_at_step(
    challenge: &Level2Challenge,
    schedule: &[PortfolioAction],
    target_step: usize,
) -> Vec<f64> {
    let mut socs: Vec<f64> = challenge
        .batteries
        .iter()
        .map(|b| b.spec.soc_initial_mwh)
        .collect();

    for step in 0..target_step.min(schedule.len()) {
        for (b, placed) in challenge.batteries.iter().enumerate() {
            let action = &schedule[step].battery_actions[b];
            socs[b] = socs[b]
                + placed.spec.efficiency_charge * action.charge_mw
                - action.discharge_mw / placed.spec.efficiency_discharge;
            socs[b] = socs[b].clamp(placed.spec.soc_min_mwh, placed.spec.soc_max_mwh);
        }
    }

    socs
}

/// Ensure final schedule respects all constraints (SOC and flow)
fn ensure_feasibility(
    challenge: &Level2Challenge,
    mut schedule: Vec<PortfolioAction>,
) -> Vec<PortfolioAction> {
    let t = schedule.len();

    // Track SOCs and fix any violations
    let mut socs: Vec<f64> = challenge
        .batteries
        .iter()
        .map(|b| b.spec.soc_initial_mwh)
        .collect();

    for step in 0..t {
        let mut actions = schedule[step].battery_actions.clone();

        // First pass: fix SOC constraint violations
        for (b, placed) in challenge.batteries.iter().enumerate() {
            let battery = &placed.spec;
            let soc = socs[b];
            let action = &mut actions[b];

            // Compute what SOC would be after this action
            let new_soc = soc
                + battery.efficiency_charge * action.charge_mw
                - action.discharge_mw / battery.efficiency_discharge;

            // Fix if violating upper bound
            if new_soc > battery.soc_max_mwh + 1e-6 {
                // Reduce charge or increase discharge
                let excess = new_soc - battery.soc_max_mwh;
                if action.charge_mw > 0.0 {
                    let reduce_charge = (excess / battery.efficiency_charge).min(action.charge_mw);
                    action.charge_mw -= reduce_charge;
                }
            }

            // Fix if violating lower bound
            if new_soc < battery.soc_min_mwh - 1e-6 {
                // Reduce discharge or increase charge
                let deficit = battery.soc_min_mwh - new_soc;
                if action.discharge_mw > 0.0 {
                    let reduce_discharge = (deficit * battery.efficiency_discharge).min(action.discharge_mw);
                    action.discharge_mw -= reduce_discharge;
                }
            }

            // Ensure non-negative
            action.charge_mw = action.charge_mw.max(0.0);
            action.discharge_mw = action.discharge_mw.max(0.0);
        }

        // Second pass: check flow constraints
        let portfolio = PortfolioAction { battery_actions: actions.clone() };
        let injections = challenge.compute_injections(&portfolio);
        let flows = challenge.network.compute_flows(&injections);

        if let Some(_violation) = challenge.network.check_flow_limits(&flows) {
            // Scale down actions
            let max_ratio = flows
                .iter()
                .zip(challenge.network.flow_limits.iter())
                .map(|(&flow, &limit)| {
                    if flow.abs() > 1e-6 {
                        (limit * 0.95) / flow.abs()
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
            }
        }

        // Update schedule with fixed actions
        schedule[step] = PortfolioAction { battery_actions: actions.clone() };

        // Update SOCs for next iteration
        for (b, placed) in challenge.batteries.iter().enumerate() {
            let battery = &placed.spec;
            let action = &actions[b];
            let new_soc = socs[b]
                + battery.efficiency_charge * action.charge_mw
                - action.discharge_mw / battery.efficiency_discharge;
            socs[b] = new_soc.clamp(battery.soc_min_mwh, battery.soc_max_mwh);
        }
    }

    schedule
}

/// Compute average DA price across nodes
fn average_da_price(challenge: &Level2Challenge, step: usize) -> f64 {
    let sum: f64 = challenge
        .day_ahead_prices
        .iter()
        .map(|prices| prices[step])
        .sum();
    sum / challenge.network.num_nodes as f64
}

/// Compute price percentile for a node at a step
fn price_percentile(challenge: &Level2Challenge, step: usize, node: usize) -> f64 {
    let node_price = challenge.day_ahead_prices[node][step];
    let t = challenge.difficulty.num_steps;

    // Compare to all prices at this node
    let mut prices: Vec<f64> = challenge.day_ahead_prices[node].clone();
    prices.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let rank = prices
        .iter()
        .position(|&p| p >= node_price)
        .unwrap_or(t);

    rank as f64 / t as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tig_challenges::energy_arbitrage_v2::Level2Difficulty;

    #[test]
    fn test_decomposition_solver() {
        let difficulty = Level2Difficulty {
            num_steps: 12, // Shorter for test
            ..Default::default()
        };
        let seed = [42u8; 32];
        let challenge = Level2Challenge::generate_instance(seed, &difficulty).unwrap();

        let config = DecompositionConfig {
            max_iterations: 10,
            ..Default::default()
        };

        let solution = solve_with_config(&challenge, &config).unwrap().unwrap();
        assert_eq!(solution.schedule.len(), 12);
    }
}
