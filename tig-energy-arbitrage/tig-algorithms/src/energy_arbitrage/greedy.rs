/*!
 * Copyright 2025 TIG Energy Arbitrage Challenge
 * 
 * Greedy threshold-based solver for Energy Arbitrage.
 * 
 * This algorithm uses simple price thresholds relative to day-ahead
 * prices to decide when to charge or discharge.
 * 
 * Licensed under the TIG Inbound Game License v2.0
 */

use anyhow::Result;
use tig_challenges::energy_arbitrage::{Challenge, Solution, BatterySpec, Frictions};

/// Greedy policy parameters
pub struct GreedyParams {
    pub buy_threshold: f64,   // Charge when RT < buy_threshold * DA
    pub sell_threshold: f64,  // Discharge when RT > sell_threshold * DA
    pub soc_buffer: f64,      // Buffer from SOC limits (fraction of capacity)
}

impl Default for GreedyParams {
    fn default() -> Self {
        Self {
            buy_threshold: 0.95,
            sell_threshold: 1.05,
            soc_buffer: 0.02,
        }
    }
}

/// Solve using greedy threshold policy
pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let params = GreedyParams::default();
    solve_with_params(challenge, &params)
}

/// Solve with custom parameters
pub fn solve_with_params(
    challenge: &Challenge,
    params: &GreedyParams,
) -> Result<Option<Solution>> {
    let t = challenge.difficulty.num_steps;
    let n_scenarios = challenge.difficulty.num_scenarios;
    let battery = &challenge.battery;
    let frictions = &challenge.frictions;
    
    // Use expected RT prices for decision making
    let expected_rt: Vec<f64> = (0..t)
        .map(|i| {
            challenge.realtime_prices[i].iter().sum::<f64>() / n_scenarios as f64
        })
        .collect();
    
    let mut actions_mw = Vec::with_capacity(t);
    let mut soc = battery.soc_initial_mwh;
    
    let soc_low = battery.soc_min_mwh + params.soc_buffer * battery.capacity_mwh;
    let soc_high = battery.soc_max_mwh - params.soc_buffer * battery.capacity_mwh;
    
    for i in 0..t {
        let rt = expected_rt[i];
        let da = challenge.day_ahead_prices[i];
        
        let action = if rt < params.buy_threshold * da && soc < soc_high {
            // Charge
            let max_charge = (soc_high - soc) / battery.efficiency_charge;
            battery.power_mw.min(max_charge)
        } else if rt > params.sell_threshold * da && soc > soc_low {
            // Discharge
            let max_discharge = (soc - soc_low) * battery.efficiency_discharge;
            -battery.power_mw.min(max_discharge)
        } else {
            0.0
        };
        
        actions_mw.push(action);
        
        // Update SOC
        let (new_soc, _) = apply_action(soc, action, rt, battery, frictions);
        soc = new_soc;
    }
    
    Ok(Some(Solution { actions_mw }))
}

/// Apply action and return new SOC and profit
fn apply_action(
    soc: f64,
    action_mw: f64,
    price: f64,
    battery: &BatterySpec,
    frictions: &Frictions,
) -> (f64, f64) {
    let p_charge = action_mw.max(0.0);
    let p_discharge = (-action_mw).max(0.0);
    
    let soc_next = soc 
        + battery.efficiency_charge * p_charge
        - p_discharge / battery.efficiency_discharge;
    let soc_clamped = soc_next.clamp(battery.soc_min_mwh, battery.soc_max_mwh);
    
    let tc = frictions.transaction_cost_pct;
    let deg = frictions.degradation_cost_per_mwh;
    
    let profit = (1.0 - tc) * price * p_discharge 
                - (1.0 + tc) * price * p_charge 
                - deg * p_discharge;
    
    (soc_clamped, profit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tig_challenges::energy_arbitrage::Difficulty;

    #[test]
    fn test_greedy_solver() {
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
        
        // Check that all actions are within bounds
        for &action in &solution.actions_mw {
            assert!(action.abs() <= challenge.battery.power_mw + 1e-6);
        }
    }
}
