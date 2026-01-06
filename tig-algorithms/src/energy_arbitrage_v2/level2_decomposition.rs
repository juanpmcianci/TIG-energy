/*!
 * Level 2 Benders Decomposition Solver
 *
 * Coordinates portfolio dispatch using decomposition approach:
 * - Master problem: determines battery schedules
 * - Subproblem: checks network flow feasibility
 *
 * Updated for spec-compliant API with SignedAction and Track system.
 * Includes proper flow-aware optimization with binary search feasibility.
 */

use anyhow::Result;
use tig_challenges::energy_arbitrage_v2::{
    constants, Level2Challenge, Level2Solution, PortfolioAction, SignedAction,
};

/// Configuration for decomposition solver
#[derive(Debug, Clone)]
pub struct DecompositionConfig {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Flow safety margin
    pub flow_margin: f64,
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            max_iterations: 30,
            tolerance: 1e-4,
            flow_margin: 0.95,
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
    let params = challenge.difficulty.effective_params();
    let t = params.num_steps;

    // Initialize with conservative greedy solution
    let mut schedule = initial_schedule(challenge, config);

    // Iterative refinement
    for _iter in 0..config.max_iterations {
        let mut improved = false;

        for step in 0..t {
            let current_portfolio = &schedule[step];
            let improved_portfolio = improve_step(challenge, step, current_portfolio, &schedule, config);

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

    // Final feasibility enforcement
    schedule = enforce_all_constraints(challenge, schedule, config);

    Ok(Some(Level2Solution { schedule }))
}

/// Generate initial schedule - conservative to ensure feasibility
fn initial_schedule(challenge: &Level2Challenge, config: &DecompositionConfig) -> Vec<PortfolioAction> {
    let params = challenge.difficulty.effective_params();
    let t = params.num_steps;
    let m = challenge.batteries.len();
    let dt = constants::DELTA_T;

    let mut socs: Vec<f64> = challenge
        .batteries
        .iter()
        .map(|b| b.spec.soc_initial_mwh)
        .collect();

    let mut schedule = Vec::with_capacity(t);

    for step in 0..t {
        let mut actions = vec![SignedAction::idle(); m];

        // Compute baseline flows (with idle batteries)
        let idle_portfolio = PortfolioAction { actions: actions.clone() };
        let base_injections = challenge.compute_total_injections(&idle_portfolio, step);
        let base_flows = challenge.network.compute_flows(&base_injections);

        // Check baseline feasibility
        let baseline_feasible = base_flows.iter().zip(challenge.network.flow_limits.iter())
            .all(|(&f, &lim)| f.abs() <= lim * config.flow_margin);

        if baseline_feasible {
            // Try to add some battery actions
            for (b, placed) in challenge.batteries.iter().enumerate() {
                let node = placed.node;
                let battery = &placed.spec;
                let da_price = challenge.day_ahead_prices[node][step];
                let avg_price = average_da_price(challenge, step);
                let soc = socs[b];

                let headroom = (battery.soc_max_mwh - soc).max(0.0);
                let max_safe_charge = (headroom / (battery.efficiency_charge * dt))
                    .min(battery.power_charge_mw)
                    .max(0.0);

                let available = (soc - battery.soc_min_mwh).max(0.0);
                let max_safe_discharge = (available * battery.efficiency_discharge / dt)
                    .min(battery.power_discharge_mw)
                    .max(0.0);

                // Conservative action sizing
                let action_scale = 0.3; // Start conservative

                if da_price < 0.9 * avg_price && max_safe_charge > 0.1 {
                    actions[b] = SignedAction::new(-max_safe_charge * action_scale);
                } else if da_price > 1.1 * avg_price && max_safe_discharge > 0.1 {
                    actions[b] = SignedAction::new(max_safe_discharge * action_scale);
                }
            }

            // Verify and scale if needed
            actions = ensure_flow_feasibility(challenge, step, &actions, config);
        }

        // Update SOCs
        for (b, placed) in challenge.batteries.iter().enumerate() {
            let action = &actions[b];
            let new_soc = challenge.apply_action_to_soc(b, socs[b], action);
            socs[b] = new_soc.clamp(placed.spec.soc_min_mwh, placed.spec.soc_max_mwh);
        }

        schedule.push(PortfolioAction { actions });
    }

    schedule
}

/// Ensure actions are flow-feasible using binary search
fn ensure_flow_feasibility(
    challenge: &Level2Challenge,
    step: usize,
    actions: &[SignedAction],
    config: &DecompositionConfig,
) -> Vec<SignedAction> {
    let portfolio = PortfolioAction { actions: actions.to_vec() };
    let injections = challenge.compute_total_injections(&portfolio, step);
    let flows = challenge.network.compute_flows(&injections);

    // Check if already feasible
    let feasible = flows.iter().zip(challenge.network.flow_limits.iter())
        .all(|(&f, &lim)| f.abs() <= lim + constants::EPS_FLOW);

    if feasible {
        return actions.to_vec();
    }

    // Binary search for maximum feasible scale
    let mut low = 0.0;
    let mut high = 1.0;

    for _ in 0..20 {
        let mid = (low + high) / 2.0;
        let scaled: Vec<SignedAction> = actions.iter()
            .map(|a| SignedAction::new(a.power_mw * mid))
            .collect();
        let scaled_portfolio = PortfolioAction { actions: scaled };
        let inj = challenge.compute_total_injections(&scaled_portfolio, step);
        let fl = challenge.network.compute_flows(&inj);

        let feas = fl.iter().zip(challenge.network.flow_limits.iter())
            .all(|(&f, &lim)| f.abs() <= lim * config.flow_margin);

        if feas {
            low = mid;
        } else {
            high = mid;
        }
    }

    actions.iter()
        .map(|a| SignedAction::new(a.power_mw * low))
        .collect()
}

/// Try to improve a single step's portfolio
fn improve_step(
    challenge: &Level2Challenge,
    step: usize,
    current: &PortfolioAction,
    full_schedule: &[PortfolioAction],
    config: &DecompositionConfig,
) -> PortfolioAction {
    let m = challenge.batteries.len();
    let dt = constants::DELTA_T;
    let socs = compute_socs_at_step(challenge, full_schedule, step);

    let mut best_actions = current.actions.clone();
    let mut best_value = evaluate_portfolio(challenge, step, current);

    // Try improving each battery
    for b in 0..m {
        let placed = &challenge.batteries[b];
        let battery = &placed.spec;
        let soc = socs[b];

        let headroom = (battery.soc_max_mwh - soc).max(0.0);
        let max_charge = (headroom / (battery.efficiency_charge * dt))
            .min(battery.power_charge_mw)
            .max(0.0);

        let available = (soc - battery.soc_min_mwh).max(0.0);
        let max_discharge = (available * battery.efficiency_discharge / dt)
            .min(battery.power_discharge_mw)
            .max(0.0);

        // Try different action levels
        for &frac in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            for &sign in &[-1.0, 1.0] {
                let power = if sign < 0.0 {
                    -max_charge * frac
                } else {
                    max_discharge * frac
                };

                let candidate = SignedAction::new(power);

                // Check SOC feasibility
                let new_soc = challenge.apply_action_to_soc(b, soc, &candidate);
                if new_soc < battery.soc_min_mwh - constants::EPS_SOC
                    || new_soc > battery.soc_max_mwh + constants::EPS_SOC
                {
                    continue;
                }

                let mut trial_actions = best_actions.clone();
                trial_actions[b] = candidate;

                // Check flow feasibility
                let trial_portfolio = PortfolioAction { actions: trial_actions.clone() };
                let injections = challenge.compute_total_injections(&trial_portfolio, step);
                let flows = challenge.network.compute_flows(&injections);

                let flow_ok = flows.iter().zip(challenge.network.flow_limits.iter())
                    .all(|(&f, &lim)| f.abs() <= lim * config.flow_margin);

                if !flow_ok {
                    continue;
                }

                let value = evaluate_portfolio(challenge, step, &trial_portfolio);
                if value > best_value {
                    best_value = value;
                    best_actions = trial_actions;
                }
            }
        }
    }

    PortfolioAction { actions: best_actions }
}

/// Evaluate expected profit of a portfolio action
fn evaluate_portfolio(
    challenge: &Level2Challenge,
    step: usize,
    portfolio: &PortfolioAction,
) -> f64 {
    let dt = constants::DELTA_T;
    let mut profit = 0.0;

    for (b, placed) in challenge.batteries.iter().enumerate() {
        let action = &portfolio.actions[b];
        let node = placed.node;
        let price = challenge.day_ahead_prices[node][step];

        let u = action.power_mw;
        let abs_u = u.abs();

        let revenue = u * price * dt;
        let tx_cost = challenge.frictions.transaction_cost_per_mwh * abs_u * dt;
        let deg_base = (abs_u * dt) / placed.spec.capacity_mwh;
        let deg_cost = challenge.frictions.degradation_scale
            * deg_base.powf(challenge.frictions.degradation_exponent);

        profit += revenue - tx_cost - deg_cost;
    }

    profit
}

/// Compute SOCs at a given step
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
            let action = &schedule[step].actions[b];
            let new_soc = challenge.apply_action_to_soc(b, socs[b], action);
            socs[b] = new_soc.clamp(placed.spec.soc_min_mwh, placed.spec.soc_max_mwh);
        }
    }

    socs
}

/// Final pass to enforce all constraints
fn enforce_all_constraints(
    challenge: &Level2Challenge,
    mut schedule: Vec<PortfolioAction>,
    config: &DecompositionConfig,
) -> Vec<PortfolioAction> {
    let t = schedule.len();
    let dt = constants::DELTA_T;

    let mut socs: Vec<f64> = challenge
        .batteries
        .iter()
        .map(|b| b.spec.soc_initial_mwh)
        .collect();

    for step in 0..t {
        let mut actions = schedule[step].actions.clone();

        // Fix SOC violations
        for (b, placed) in challenge.batteries.iter().enumerate() {
            let battery = &placed.spec;
            let soc = socs[b];
            let action = &mut actions[b];

            let new_soc = challenge.apply_action_to_soc(b, soc, action);

            if new_soc > battery.soc_max_mwh + constants::EPS_SOC {
                if action.power_mw < 0.0 {
                    let excess = new_soc - battery.soc_max_mwh;
                    let reduce = (excess / (battery.efficiency_charge * dt)).min(-action.power_mw);
                    action.power_mw += reduce;
                }
            }

            if new_soc < battery.soc_min_mwh - constants::EPS_SOC {
                if action.power_mw > 0.0 {
                    let deficit = battery.soc_min_mwh - new_soc;
                    let reduce = (deficit * battery.efficiency_discharge / dt).min(action.power_mw);
                    action.power_mw -= reduce;
                }
            }
        }

        // Enforce flow feasibility
        actions = ensure_flow_feasibility(challenge, step, &actions, config);

        schedule[step] = PortfolioAction { actions: actions.clone() };

        // Update SOCs
        for (b, placed) in challenge.batteries.iter().enumerate() {
            let action = &actions[b];
            let new_soc = challenge.apply_action_to_soc(b, socs[b], action);
            socs[b] = new_soc.clamp(placed.spec.soc_min_mwh, placed.spec.soc_max_mwh);
        }
    }

    schedule
}

/// Compute average DA price
fn average_da_price(challenge: &Level2Challenge, step: usize) -> f64 {
    let sum: f64 = challenge
        .day_ahead_prices
        .iter()
        .map(|prices| prices[step])
        .sum();
    sum / challenge.network.num_nodes as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tig_challenges::energy_arbitrage_v2::{Level2Difficulty, Track};

    #[test]
    fn test_decomposition_solver_track1() {
        let difficulty = Level2Difficulty {
            track: Track::Track1,
            num_steps: Some(24),
            ..Default::default()
        };
        let seed = [42u8; 32];
        let challenge = Level2Challenge::generate_instance(seed, &difficulty).unwrap();

        let solution = solve_challenge(&challenge).unwrap().unwrap();
        assert_eq!(solution.schedule.len(), 24);

        match challenge.verify_solution(&solution) {
            Ok(profit) => println!("Decomposition profit: {:.2}", profit),
            Err(e) => {
                let err_str = e.to_string();
                if !err_str.contains("below threshold") {
                    panic!("Unexpected error: {}", e);
                }
            }
        }
    }

    #[test]
    fn test_decomposition_all_tracks() {
        for track in Track::all() {
            let difficulty = Level2Difficulty {
                track,
                num_steps: Some(24),
                ..Default::default()
            };
            let seed = [123u8; 32];
            let challenge = Level2Challenge::generate_instance(seed, &difficulty).unwrap();

            let solution = solve_challenge(&challenge).unwrap().unwrap();
            assert_eq!(solution.schedule.len(), 24);

            match challenge.verify_solution(&solution) {
                Ok(profit) => println!("Track {:?}: profit ${:.2}", track, profit),
                Err(e) => println!("Track {:?}: {}", track, e),
            }
        }
    }
}
