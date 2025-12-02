/*!
 * Copyright 2025 TIG Energy Arbitrage Challenge
 * 
 * Template algorithm for the Energy Arbitrage challenge.
 * 
 * Licensed under the TIG Inbound Game License v2.0
 */

use anyhow::Result;
use tig_challenges::energy_arbitrage::{Challenge, Solution};

/// Solve the energy arbitrage challenge
/// 
/// Returns Ok(Some(solution)) if a valid solution is found,
/// Ok(None) if no solution exists, or Err on algorithm error.
pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let t = challenge.difficulty.num_steps;
    
    // TODO: Implement your algorithm here
    // 
    // The challenge provides:
    // - challenge.day_ahead_prices: Vec<f64> (length T)
    // - challenge.realtime_prices: Vec<Vec<f64>> (T x num_scenarios)
    // - challenge.battery: BatterySpec
    // - challenge.frictions: Frictions
    // - challenge.baseline_profit: f64
    //
    // Your solution should provide:
    // - actions_mw: Vec<f64> (length T)
    //   positive = charge, negative = discharge
    //
    // Constraints:
    // - |action| <= battery.power_mw
    // - SOC must stay within [soc_min_mwh, soc_max_mwh]
    // - Profit must beat baseline by difficulty.better_than_baseline percent
    
    // Placeholder: return zero actions (hold)
    let actions_mw = vec![0.0; t];
    
    Ok(Some(Solution { actions_mw }))
}
