/*!
 * Copyright 2025 TIG Energy Arbitrage Challenge
 * 
 * MPC-DP (Model Predictive Control with Dynamic Programming) solver
 * for the Energy Arbitrage challenge.
 * 
 * This algorithm uses rolling-horizon dynamic programming over
 * expected prices to compute optimal actions.
 * 
 * Licensed under the TIG Inbound Game License v2.0
 */

use anyhow::Result;
use tig_challenges::energy_arbitrage::{Challenge, Solution};

/// MPC parameters
const HORIZON: usize = 6;
const SOC_GRID_SIZE: usize = 51;
const ACTION_LEVELS: usize = 5;

/// Solve the energy arbitrage challenge using MPC-DP
pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let t = challenge.difficulty.num_steps;
    let n_scenarios = challenge.difficulty.num_scenarios;
    let battery = &challenge.battery;
    let frictions = &challenge.frictions;
    
    // Compute expected prices across scenarios
    let expected_prices: Vec<f64> = (0..t)
        .map(|i| {
            challenge.realtime_prices[i].iter().sum::<f64>() / n_scenarios as f64
        })
        .collect();
    
    // Build SOC grid
    let soc_grid: Vec<f64> = (0..SOC_GRID_SIZE)
        .map(|i| {
            battery.soc_min_mwh 
                + (battery.soc_max_mwh - battery.soc_min_mwh) 
                  * (i as f64 / (SOC_GRID_SIZE - 1) as f64)
        })
        .collect();
    
    // Build action levels
    let action_levels: Vec<f64> = (0..ACTION_LEVELS)
        .map(|i| {
            -battery.power_mw 
                + 2.0 * battery.power_mw * (i as f64 / (ACTION_LEVELS - 1) as f64)
        })
        .collect();
    
    let mut actions_mw = Vec::with_capacity(t);
    let mut current_soc = battery.soc_initial_mwh;
    
    // Rolling horizon MPC
    for step in 0..t {
        let horizon_end = (step + HORIZON).min(t);
        let horizon_prices = &expected_prices[step..horizon_end];
        let h = horizon_prices.len();
        
        // Value function: V[time][soc_index]
        let mut v = vec![vec![f64::NEG_INFINITY; SOC_GRID_SIZE]; h + 1];
        
        // Terminal value = 0 (no salvage)
        for si in 0..SOC_GRID_SIZE {
            v[h][si] = 0.0;
        }
        
        // Backward induction
        for tau in (0..h).rev() {
            let price = horizon_prices[tau];
            
            for si in 0..SOC_GRID_SIZE {
                let soc = soc_grid[si];
                let mut best_value = f64::NEG_INFINITY;
                
                for &action in &action_levels {
                    let (soc_next, feasible) = apply_action_check(
                        soc, action, battery
                    );
                    
                    if !feasible {
                        continue;
                    }
                    
                    let reward = compute_reward(action, price, frictions);
                    let next_si = soc_to_index(soc_next, battery, SOC_GRID_SIZE);
                    let future_value = v[tau + 1][next_si];
                    
                    let total = reward + future_value;
                    if total > best_value {
                        best_value = total;
                    }
                }
                
                v[tau][si] = best_value;
            }
        }
        
        // Extract optimal first action
        let current_si = soc_to_index(current_soc, battery, SOC_GRID_SIZE);
        let price = horizon_prices[0];
        
        let mut best_action = 0.0;
        let mut best_value = f64::NEG_INFINITY;
        
        for &action in &action_levels {
            let (soc_next, feasible) = apply_action_check(
                soc_grid[current_si], action, battery
            );
            
            if !feasible {
                continue;
            }
            
            let reward = compute_reward(action, price, frictions);
            let next_si = soc_to_index(soc_next, battery, SOC_GRID_SIZE);
            let future_value = if h > 1 { v[1][next_si] } else { 0.0 };
            
            let total = reward + future_value;
            if total > best_value {
                best_value = total;
                best_action = action;
            }
        }
        
        actions_mw.push(best_action);
        
        // Update state
        let (new_soc, _) = apply_action_check(current_soc, best_action, battery);
        current_soc = new_soc;
    }
    
    Ok(Some(Solution { actions_mw }))
}

/// Apply action and check feasibility
fn apply_action_check(
    soc: f64,
    action_mw: f64,
    battery: &tig_challenges::energy_arbitrage::BatterySpec,
) -> (f64, bool) {
    let p_charge = action_mw.max(0.0);
    let p_discharge = (-action_mw).max(0.0);
    
    let soc_next = soc 
        + battery.efficiency_charge * p_charge
        - p_discharge / battery.efficiency_discharge;
    
    let feasible = soc_next >= battery.soc_min_mwh - 1e-9 
                && soc_next <= battery.soc_max_mwh + 1e-9;
    
    (soc_next.clamp(battery.soc_min_mwh, battery.soc_max_mwh), feasible)
}

/// Compute single-step reward
fn compute_reward(
    action_mw: f64,
    price: f64,
    frictions: &tig_challenges::energy_arbitrage::Frictions,
) -> f64 {
    let p_charge = action_mw.max(0.0);
    let p_discharge = (-action_mw).max(0.0);
    
    let tc = frictions.transaction_cost_pct;
    let deg = frictions.degradation_cost_per_mwh;
    
    (1.0 - tc) * price * p_discharge 
        - (1.0 + tc) * price * p_charge 
        - deg * p_discharge
}

/// Convert SOC to grid index
fn soc_to_index(
    soc: f64,
    battery: &tig_challenges::energy_arbitrage::BatterySpec,
    grid_size: usize,
) -> usize {
    let frac = (soc - battery.soc_min_mwh) 
        / (battery.soc_max_mwh - battery.soc_min_mwh);
    let idx = (frac * (grid_size - 1) as f64).round() as usize;
    idx.min(grid_size - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tig_challenges::energy_arbitrage::Difficulty;

    #[test]
    fn test_mpc_dp_solver() {
        let difficulty = Difficulty {
            num_steps: 24,
            num_scenarios: 50,
            volatility_percent: 20,
            tail_risk_percent: 5,
            better_than_baseline: 0,
        };
        
        let seed = [42u8; 32];
        let challenge = tig_challenges::energy_arbitrage::Challenge::generate_instance(
            seed, &difficulty
        ).unwrap();
        
        let solution = solve_challenge(&challenge).unwrap().unwrap();
        assert_eq!(solution.actions_mw.len(), 24);
        
        // Verify solution is valid
        let result = challenge.verify_solution(&solution);
        assert!(result.is_ok(), "Solution should be valid: {:?}", result);
    }
}
