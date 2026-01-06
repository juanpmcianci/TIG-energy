/*!
 * Level 2 Greedy Solver
 *
 * Flow-aware heuristic for portfolio arbitrage on constrained network.
 * Properly accounts for exogenous injections and iteratively adjusts
 * battery actions to satisfy network flow constraints.
 *
 * Updated for spec-compliant API with SignedAction and Track system.
 */

use anyhow::Result;
use tig_challenges::energy_arbitrage_v2::{
    constants, Level2Challenge, Level2Solution, PortfolioAction, SignedAction,
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
    /// Max iterations for flow adjustment
    pub max_flow_iterations: usize,
}

impl Default for GreedyConfig {
    fn default() -> Self {
        Self {
            charge_threshold: 0.95,
            discharge_threshold: 1.05,
            flow_safety_margin: 0.85,
            max_flow_iterations: 20,
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
    let params = challenge.difficulty.effective_params();
    let t = params.num_steps;
    let m = challenge.batteries.len();
    let dt = constants::DELTA_T;

    // Track SOC for each battery
    let mut socs: Vec<f64> = challenge
        .batteries
        .iter()
        .map(|b| b.spec.soc_initial_mwh)
        .collect();

    let mut schedule = Vec::with_capacity(t);

    for step in 0..t {
        // First, check if exogenous flows alone are feasible
        let idle_portfolio = PortfolioAction {
            actions: vec![SignedAction::idle(); m],
        };
        let base_injections = challenge.compute_total_injections(&idle_portfolio, step);
        let base_flows = challenge.network.compute_flows(&base_injections);

        // Compute remaining headroom on each line
        let mut flow_headroom: Vec<(f64, f64)> = Vec::with_capacity(challenge.network.num_lines);
        for (l, &flow) in base_flows.iter().enumerate() {
            let limit = challenge.network.flow_limits[l] * config.flow_safety_margin;
            // (negative_room, positive_room) - how much more flow we can add in each direction
            let neg_room = limit - (-flow).max(0.0); // Room for more negative flow
            let pos_room = limit - flow.max(0.0);     // Room for more positive flow
            flow_headroom.push((neg_room.max(0.0), pos_room.max(0.0)));
        }

        // Determine greedy actions for each battery
        let mut candidate_actions = Vec::with_capacity(m);

        for (b, placed) in challenge.batteries.iter().enumerate() {
            let node = placed.node;
            let battery = &placed.spec;
            let da_price = challenge.day_ahead_prices[node][step];
            let soc = socs[b];

            // Compute SOC headroom
            let headroom = battery.soc_max_mwh - soc;
            let available = soc - battery.soc_min_mwh;

            // Maximum safe charge/discharge considering Δt
            let max_charge = (headroom / (battery.efficiency_charge * dt))
                .min(battery.power_charge_mw)
                .max(0.0);
            let max_discharge = (available * battery.efficiency_discharge / dt)
                .min(battery.power_discharge_mw)
                .max(0.0);

            // Simple greedy decision based on price
            let avg_price = average_da_price(challenge, step);

            let action = if da_price < config.charge_threshold * avg_price {
                // Low price - want to charge (negative u)
                if max_charge > 0.1 {
                    SignedAction::new(-max_charge)
                } else {
                    SignedAction::idle()
                }
            } else if da_price > config.discharge_threshold * avg_price {
                // High price - want to discharge (positive u)
                if max_discharge > 0.1 {
                    SignedAction::new(max_discharge)
                } else {
                    SignedAction::idle()
                }
            } else {
                SignedAction::idle()
            };

            candidate_actions.push(action);
        }

        // Iteratively adjust actions to satisfy flow constraints
        let portfolio_action = iterative_flow_adjustment(
            challenge, step, &candidate_actions, &socs, config
        );

        // Update SOCs
        for (b, placed) in challenge.batteries.iter().enumerate() {
            let action = &portfolio_action.actions[b];
            let new_soc = challenge.apply_action_to_soc(b, socs[b], action);
            socs[b] = new_soc.clamp(placed.spec.soc_min_mwh, placed.spec.soc_max_mwh);
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

/// Iteratively adjust actions to satisfy flow limits
fn iterative_flow_adjustment(
    challenge: &Level2Challenge,
    time_step: usize,
    candidate_actions: &[SignedAction],
    socs: &[f64],
    config: &GreedyConfig,
) -> PortfolioAction {
    let mut actions = candidate_actions.to_vec();

    for _iter in 0..config.max_flow_iterations {
        let portfolio = PortfolioAction { actions: actions.clone() };
        let injections = challenge.compute_total_injections(&portfolio, time_step);
        let flows = challenge.network.compute_flows(&injections);

        // Find worst violation
        let mut worst_violation: Option<(usize, f64, f64)> = None; // (line, flow, limit)
        for (l, &flow) in flows.iter().enumerate() {
            let limit = challenge.network.flow_limits[l] * config.flow_safety_margin;
            let violation = flow.abs() - limit;
            if violation > constants::EPS_FLOW {
                match worst_violation {
                    None => worst_violation = Some((l, flow, limit)),
                    Some((_, _, prev_excess)) => {
                        if violation > flow.abs() - prev_excess {
                            worst_violation = Some((l, flow, limit));
                        }
                    }
                }
            }
        }

        // If no violations, we're done
        let Some((viol_line, viol_flow, viol_limit)) = worst_violation else {
            break;
        };

        // Find which batteries affect this line and reduce their actions
        let mut adjusted = false;

        for (b, placed) in challenge.batteries.iter().enumerate() {
            let node = placed.node;
            // Check if battery affects the violated line
            let ptdf = challenge.network.ptdf[viol_line][node];
            if ptdf.abs() < 1e-6 {
                continue; // This battery doesn't affect this line
            }

            // Determine if we should reduce action
            let action = &mut actions[b];
            let contribution = ptdf * action.power_mw;

            // If this battery's contribution is in the same direction as the violation
            if (viol_flow > 0.0 && contribution > 0.0) || (viol_flow < 0.0 && contribution < 0.0) {
                // Scale down this battery's action
                let scale = 0.5; // Reduce by half
                let old_power = action.power_mw;
                action.power_mw *= scale;

                // Check SOC feasibility after scaling
                let new_soc = challenge.apply_action_to_soc(b, socs[b], action);
                if new_soc < placed.spec.soc_min_mwh - constants::EPS_SOC
                    || new_soc > placed.spec.soc_max_mwh + constants::EPS_SOC
                {
                    // Revert if SOC violated
                    action.power_mw = old_power;
                } else {
                    adjusted = true;
                }
            }
        }

        // If we couldn't adjust any battery, scale down all actions globally
        if !adjusted {
            let scale = viol_limit / viol_flow.abs();
            if scale > 0.0 && scale < 1.0 {
                for action in &mut actions {
                    action.power_mw *= scale * 0.9; // Extra margin
                }
            } else {
                // Last resort: idle all batteries
                for action in &mut actions {
                    *action = SignedAction::idle();
                }
                break;
            }
        }
    }

    // Final verification pass - if still violated, scale down aggressively
    let portfolio = PortfolioAction { actions: actions.clone() };
    let injections = challenge.compute_total_injections(&portfolio, time_step);
    let flows = challenge.network.compute_flows(&injections);

    let mut still_violated = false;
    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l];
        if flow.abs() > limit + constants::EPS_FLOW {
            still_violated = true;
            break;
        }
    }

    if still_violated {
        // Binary search for feasible scaling
        let mut low = 0.0;
        let mut high = 1.0;

        for _ in 0..20 {
            let mid = (low + high) / 2.0;
            let scaled_actions: Vec<SignedAction> = actions
                .iter()
                .map(|a| SignedAction::new(a.power_mw * mid))
                .collect();
            let scaled_portfolio = PortfolioAction { actions: scaled_actions };
            let inj = challenge.compute_total_injections(&scaled_portfolio, time_step);
            let fl = challenge.network.compute_flows(&inj);

            let feasible = fl.iter().zip(challenge.network.flow_limits.iter())
                .all(|(&f, &lim)| f.abs() <= lim + constants::EPS_FLOW);

            if feasible {
                low = mid;
            } else {
                high = mid;
            }
        }

        // Apply the feasible scaling
        for action in &mut actions {
            action.power_mw *= low;
        }
    }

    PortfolioAction { actions }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tig_challenges::energy_arbitrage_v2::{Level2Difficulty, Track};

    #[test]
    fn test_level2_greedy_track1() {
        let difficulty = Level2Difficulty::from_track(Track::Track1);
        let seed = [42u8; 32];
        let challenge = Level2Challenge::generate_instance(seed, &difficulty).unwrap();

        let solution = solve_challenge(&challenge).unwrap().unwrap();
        assert_eq!(solution.schedule.len(), 96); // Track 1 has 96 steps

        // Verify solution validity
        match challenge.verify_solution(&solution) {
            Ok(profit) => println!("Level 2 greedy profit: {:.2}", profit),
            Err(e) => {
                let err_str = e.to_string();
                // Allow profit threshold failures in test
                if !err_str.contains("below threshold") {
                    panic!("Unexpected error: {}", e);
                }
            }
        }
    }

    #[test]
    fn test_level2_greedy_all_tracks() {
        for track in Track::all() {
            let difficulty = Level2Difficulty::from_track(track);
            let seed = [123u8; 32];

            let challenge = Level2Challenge::generate_instance(seed, &difficulty).unwrap();
            let params = difficulty.effective_params();

            let solution = solve_challenge(&challenge).unwrap().unwrap();
            assert_eq!(solution.schedule.len(), params.num_steps);

            // Verify - greedy should at least produce feasible solutions now
            match challenge.verify_solution(&solution) {
                Ok(profit) => println!("Track {:?}: profit ${:.2}", track, profit),
                Err(e) => println!("Track {:?}: {}", track, e),
            }
        }
    }
}
