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
use tig_challenges::energy_arbitrage::{Challenge, Solution, BatterySpec, Frictions};

/// MPC configuration
pub struct MpcConfig {
    pub horizon: usize,
    pub soc_grid_size: usize,
    pub action_levels: usize,
    pub price_aggregation: PriceAggregation,
}

pub enum PriceAggregation {
    Mean,
    Percentile(f64),
    Robust { alpha: f64 }, // CVaR-like
}

impl Default for MpcConfig {
    fn default() -> Self {
        Self {
            horizon: 6,
            soc_grid_size: 51,
            action_levels: 5,
            price_aggregation: PriceAggregation::Mean,
        }
    }
}

/// Solve using MPC-DP
pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let config = MpcConfig::default();
    solve_with_config(challenge, &config)
}

/// Solve with custom configuration
pub fn solve_with_config(
    challenge: &Challenge,
    config: &MpcConfig,
) -> Result<Option<Solution>> {
    let t = challenge.difficulty.num_steps;
    let battery = &challenge.battery;
    let frictions = &challenge.frictions;
    
    // Aggregate prices based on configuration
    let expected_prices = aggregate_prices(challenge, &config.price_aggregation);
    
    // Build grids
    let soc_grid: Vec<f64> = (0..config.soc_grid_size)
        .map(|i| {
            battery.soc_min_mwh 
                + (battery.soc_max_mwh - battery.soc_min_mwh) 
                  * (i as f64 / (config.soc_grid_size - 1) as f64)
        })
        .collect();
    
    let action_levels: Vec<f64> = (0..config.action_levels)
        .map(|i| {
            -battery.power_mw 
                + 2.0 * battery.power_mw * (i as f64 / (config.action_levels - 1) as f64)
        })
        .collect();
    
    let mut actions_mw = Vec::with_capacity(t);
    let mut current_soc = battery.soc_initial_mwh;
    
    // Rolling horizon MPC
    for step in 0..t {
        let horizon_end = (step + config.horizon).min(t);
        let horizon_prices = &expected_prices[step..horizon_end];
        
        let best_action = dp_solve_first_action(
            current_soc,
            horizon_prices,
            &soc_grid,
            &action_levels,
            battery,
            frictions,
        );
        
        actions_mw.push(best_action);
        
        // Update state
        let p_charge = best_action.max(0.0);
        let p_discharge = (-best_action).max(0.0);
        current_soc = current_soc 
            + battery.efficiency_charge * p_charge
            - p_discharge / battery.efficiency_discharge;
        current_soc = current_soc.clamp(battery.soc_min_mwh, battery.soc_max_mwh);
    }
    
    Ok(Some(Solution { actions_mw }))
}

/// Aggregate prices across scenarios
fn aggregate_prices(challenge: &Challenge, method: &PriceAggregation) -> Vec<f64> {
    let t = challenge.difficulty.num_steps;
    let n = challenge.difficulty.num_scenarios;
    
    (0..t).map(|i| {
        let mut prices: Vec<f64> = challenge.realtime_prices[i].clone();
        
        match method {
            PriceAggregation::Mean => {
                prices.iter().sum::<f64>() / n as f64
            }
            PriceAggregation::Percentile(p) => {
                prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = ((p / 100.0) * (n - 1) as f64).round() as usize;
                prices[idx.min(n - 1)]
            }
            PriceAggregation::Robust { alpha } => {
                // CVaR-like: average of worst alpha% scenarios
                prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let cutoff = ((1.0 - alpha) * n as f64).ceil() as usize;
                prices[..cutoff.max(1)].iter().sum::<f64>() / cutoff.max(1) as f64
            }
        }
    }).collect()
}

/// Solve DP over horizon and return optimal first action
fn dp_solve_first_action(
    soc0: f64,
    prices: &[f64],
    soc_grid: &[f64],
    action_levels: &[f64],
    battery: &BatterySpec,
    frictions: &Frictions,
) -> f64 {
    let h = prices.len();
    let n_soc = soc_grid.len();
    
    // Value function: v[time][soc_idx]
    let mut v = vec![vec![f64::NEG_INFINITY; n_soc]; h + 1];
    
    // Terminal value = 0
    for si in 0..n_soc {
        v[h][si] = 0.0;
    }
    
    // Backward induction
    for tau in (0..h).rev() {
        let price = prices[tau];
        
        for si in 0..n_soc {
            let soc = soc_grid[si];
            let mut best = f64::NEG_INFINITY;
            
            for &action in action_levels {
                if let Some((soc_next, reward)) = transition(soc, action, price, battery, frictions) {
                    let next_si = soc_to_index(soc_next, soc_grid);
                    let value = reward + v[tau + 1][next_si];
                    if value > best {
                        best = value;
                    }
                }
            }
            
            v[tau][si] = best;
        }
    }
    
    // Find optimal first action
    let si0 = soc_to_index(soc0, soc_grid);
    let price0 = prices[0];
    
    let mut best_action = 0.0;
    let mut best_value = f64::NEG_INFINITY;
    
    for &action in action_levels {
        if let Some((soc_next, reward)) = transition(soc_grid[si0], action, price0, battery, frictions) {
            let next_si = soc_to_index(soc_next, soc_grid);
            let future = if h > 1 { v[1][next_si] } else { 0.0 };
            let value = reward + future;
            
            if value > best_value {
                best_value = value;
                best_action = action;
            }
        }
    }
    
    best_action
}

/// State transition: returns (new_soc, reward) or None if infeasible
fn transition(
    soc: f64,
    action: f64,
    price: f64,
    battery: &BatterySpec,
    frictions: &Frictions,
) -> Option<(f64, f64)> {
    let p_charge = action.max(0.0);
    let p_discharge = (-action).max(0.0);
    
    let soc_next = soc 
        + battery.efficiency_charge * p_charge
        - p_discharge / battery.efficiency_discharge;
    
    if soc_next < battery.soc_min_mwh - 1e-9 || soc_next > battery.soc_max_mwh + 1e-9 {
        return None;
    }
    
    let tc = frictions.transaction_cost_pct;
    let deg = frictions.degradation_cost_per_mwh;
    
    let reward = (1.0 - tc) * price * p_discharge 
               - (1.0 + tc) * price * p_charge 
               - deg * p_discharge;
    
    Some((soc_next.clamp(battery.soc_min_mwh, battery.soc_max_mwh), reward))
}

/// Map SOC value to grid index
fn soc_to_index(soc: f64, grid: &[f64]) -> usize {
    let n = grid.len();
    let soc_min = grid[0];
    let soc_max = grid[n - 1];
    let frac = (soc - soc_min) / (soc_max - soc_min);
    let idx = (frac * (n - 1) as f64).round() as usize;
    idx.min(n - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tig_challenges::energy_arbitrage::Difficulty;

    #[test]
    fn test_mpc_dp() {
        let difficulty = Difficulty {
            num_steps: 24,
            num_scenarios: 50,
            volatility_percent: 20,
            tail_risk_percent: 5,
            better_than_baseline: 0,
        };
        
        let seed = [42u8; 32];
        let challenge = Challenge::generate_instance(seed, &difficulty).unwrap();
        
        let solution = solve_challenge(&challenge).unwrap().unwrap();
        assert_eq!(solution.actions_mw.len(), 24);
        
        let result = challenge.verify_solution(&solution);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
    }

    #[test]
    fn test_robust_mpc() {
        let difficulty = Difficulty {
            num_steps: 24,
            num_scenarios: 50,
            volatility_percent: 30,
            tail_risk_percent: 10,
            better_than_baseline: 0,
        };
        
        let seed = [123u8; 32];
        let challenge = Challenge::generate_instance(seed, &difficulty).unwrap();
        
        let config = MpcConfig {
            horizon: 8,
            soc_grid_size: 81,
            action_levels: 9,
            price_aggregation: PriceAggregation::Robust { alpha: 0.1 },
        };
        
        let solution = solve_with_config(&challenge, &config).unwrap().unwrap();
        let result = challenge.verify_solution(&solution);
        assert!(result.is_ok());
    }
}
