/*!
 * Copyright 2025 TIG Energy Arbitrage Challenge
 * 
 * SDDP (Stochastic Dual Dynamic Programming) solver
 * for the Energy Arbitrage challenge.
 * 
 * This algorithm constructs piecewise-linear approximations of the
 * value function using Benders cuts from sampled scenarios.
 * 
 * Licensed under the TIG Inbound Game License v2.0
 */

use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand::rngs::SmallRng;
use tig_challenges::energy_arbitrage::{Challenge, Solution, BatterySpec, Frictions};

/// SDDP configuration
pub struct SddpConfig {
    pub iterations: usize,
    pub samples_per_iteration: usize,
    pub action_levels: usize,
}

impl Default for SddpConfig {
    fn default() -> Self {
        Self {
            iterations: 50,
            samples_per_iteration: 10,
            action_levels: 11,
        }
    }
}

/// A Benders cut: value >= alpha + beta * soc
#[derive(Clone)]
struct Cut {
    alpha: f64,  // intercept
    beta: f64,   // slope w.r.t. SOC
}

/// Solve using SDDP
pub fn solve_challenge(challenge: &Challenge) -> Result<Option<Solution>> {
    let config = SddpConfig::default();
    solve_with_config(challenge, &config, challenge.seed)
}

/// Solve with custom configuration
pub fn solve_with_config(
    challenge: &Challenge,
    config: &SddpConfig,
    seed: [u8; 32],
) -> Result<Option<Solution>> {
    let t = challenge.difficulty.num_steps;
    let n_scenarios = challenge.difficulty.num_scenarios;
    let battery = &challenge.battery;
    let frictions = &challenge.frictions;
    
    let mut rng = SmallRng::from_seed(seed);
    
    // Action levels
    let actions: Vec<f64> = (0..config.action_levels)
        .map(|i| {
            -battery.power_mw 
                + 2.0 * battery.power_mw * (i as f64 / (config.action_levels - 1) as f64)
        })
        .collect();
    
    // Cut sets for each stage (except terminal)
    let mut cuts: Vec<Vec<Cut>> = vec![Vec::new(); t];
    
    // SDDP iterations
    for _iter in 0..config.iterations {
        // Forward pass: sample scenarios and collect states
        let mut state_samples: Vec<Vec<f64>> = vec![Vec::new(); t];
        
        for _sample in 0..config.samples_per_iteration {
            let scenario_idx = rng.gen_range(0..n_scenarios);
            let mut soc = battery.soc_initial_mwh;
            
            for stage in 0..t {
                state_samples[stage].push(soc);
                
                let price = challenge.realtime_prices[stage][scenario_idx];
                let action = select_action_with_cuts(
                    soc, stage, price, &actions, &cuts, battery, frictions, t
                );
                
                // Transition
                let p_c = action.max(0.0);
                let p_d = (-action).max(0.0);
                soc = soc + battery.efficiency_charge * p_c - p_d / battery.efficiency_discharge;
                soc = soc.clamp(battery.soc_min_mwh, battery.soc_max_mwh);
            }
        }
        
        // Backward pass: generate cuts
        for stage in (0..t).rev() {
            for &soc in &state_samples[stage] {
                if stage == t - 1 {
                    // Terminal: no future value
                    continue;
                }
                
                // Compute cut at this state
                let cut = compute_cut(
                    soc, stage, &actions, &cuts, challenge, battery, frictions, n_scenarios
                );
                
                cuts[stage].push(cut);
            }
        }
        
        // Prune dominated cuts (keep only non-dominated ones)
        for stage in 0..t {
            prune_cuts(&mut cuts[stage], battery);
        }
    }
    
    // Final policy execution
    let mut actions_mw = Vec::with_capacity(t);
    let mut soc = battery.soc_initial_mwh;
    
    // Use mean prices for final execution
    let mean_prices: Vec<f64> = (0..t)
        .map(|i| {
            challenge.realtime_prices[i].iter().sum::<f64>() / n_scenarios as f64
        })
        .collect();
    
    for stage in 0..t {
        let action = select_action_with_cuts(
            soc, stage, mean_prices[stage], &actions, &cuts, battery, frictions, t
        );
        
        actions_mw.push(action);
        
        let p_c = action.max(0.0);
        let p_d = (-action).max(0.0);
        soc = soc + battery.efficiency_charge * p_c - p_d / battery.efficiency_discharge;
        soc = soc.clamp(battery.soc_min_mwh, battery.soc_max_mwh);
    }
    
    Ok(Some(Solution { actions_mw }))
}

/// Select best action using cut approximation of future value
fn select_action_with_cuts(
    soc: f64,
    stage: usize,
    price: f64,
    actions: &[f64],
    cuts: &[Vec<Cut>],
    battery: &BatterySpec,
    frictions: &Frictions,
    total_stages: usize,
) -> f64 {
    let mut best_action = 0.0;
    let mut best_value = f64::NEG_INFINITY;
    
    for &action in actions {
        let p_c = action.max(0.0);
        let p_d = (-action).max(0.0);
        
        let soc_next = soc 
            + battery.efficiency_charge * p_c 
            - p_d / battery.efficiency_discharge;
        
        if soc_next < battery.soc_min_mwh - 1e-9 || soc_next > battery.soc_max_mwh + 1e-9 {
            continue;
        }
        
        let tc = frictions.transaction_cost_pct;
        let deg = frictions.degradation_cost_per_mwh;
        let reward = (1.0 - tc) * price * p_d - (1.0 + tc) * price * p_c - deg * p_d;
        
        // Future value from cuts
        let future = if stage + 1 < total_stages {
            evaluate_cuts(&cuts[stage + 1], soc_next)
        } else {
            0.0
        };
        
        let value = reward + future;
        if value > best_value {
            best_value = value;
            best_action = action;
        }
    }
    
    best_action
}

/// Evaluate cut approximation at given SOC
fn evaluate_cuts(cuts: &[Cut], soc: f64) -> f64 {
    if cuts.is_empty() {
        return 0.0;
    }
    cuts.iter()
        .map(|c| c.alpha + c.beta * soc)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Compute a Benders cut at the given state
fn compute_cut(
    soc: f64,
    stage: usize,
    actions: &[f64],
    cuts: &[Vec<Cut>],
    challenge: &Challenge,
    battery: &BatterySpec,
    frictions: &Frictions,
    n_scenarios: usize,
) -> Cut {
    let t = challenge.difficulty.num_steps;
    
    // Compute expected value and derivative at this SOC
    let mut total_value = 0.0;
    let mut total_deriv = 0.0;
    
    for s in 0..n_scenarios {
        let price = challenge.realtime_prices[stage][s];
        
        // Find optimal action and its value
        let mut best_value = f64::NEG_INFINITY;
        let mut best_deriv = 0.0;
        
        for &action in actions {
            let p_c = action.max(0.0);
            let p_d = (-action).max(0.0);
            
            let soc_next = soc 
                + battery.efficiency_charge * p_c 
                - p_d / battery.efficiency_discharge;
            
            if soc_next < battery.soc_min_mwh - 1e-9 || soc_next > battery.soc_max_mwh + 1e-9 {
                continue;
            }
            
            let tc = frictions.transaction_cost_pct;
            let deg = frictions.degradation_cost_per_mwh;
            let reward = (1.0 - tc) * price * p_d - (1.0 + tc) * price * p_c - deg * p_d;
            
            let future = if stage + 1 < t - 1 {
                evaluate_cuts(&cuts[stage + 1], soc_next)
            } else {
                0.0
            };
            
            let value = reward + future;
            if value > best_value {
                best_value = value;
                // Derivative: how does value change with SOC?
                // For charging: d(reward)/d(soc) = 0, but future changes
                // Approximate: use finite difference
                best_deriv = if stage + 1 < t - 1 && !cuts[stage + 1].is_empty() {
                    // Propagate derivative through transition
                    let future_deriv = cuts[stage + 1].iter()
                        .map(|c| c.beta)
                        .fold(f64::NEG_INFINITY, f64::max);
                    // Chain rule: df/d(soc) = df/d(soc_next) * d(soc_next)/d(soc)
                    // d(soc_next)/d(soc) = 1 (from dynamics)
                    future_deriv
                } else {
                    0.0
                };
            }
        }
        
        total_value += best_value;
        total_deriv += best_deriv;
    }
    
    let avg_value = total_value / n_scenarios as f64;
    let avg_deriv = total_deriv / n_scenarios as f64;
    
    // Cut: V(soc') >= V(soc) + dV/d(soc) * (soc' - soc)
    //            = (V(soc) - dV/d(soc) * soc) + dV/d(soc) * soc'
    Cut {
        alpha: avg_value - avg_deriv * soc,
        beta: avg_deriv,
    }
}

/// Remove dominated cuts
fn prune_cuts(cuts: &mut Vec<Cut>, battery: &BatterySpec) {
    if cuts.len() <= 1 {
        return;
    }
    
    // Keep cuts that are maximal somewhere in the SOC range
    let soc_min = battery.soc_min_mwh;
    let soc_max = battery.soc_max_mwh;
    let n_test = 20;
    
    let mut keep = vec![false; cuts.len()];
    
    for i in 0..=n_test {
        let soc = soc_min + (soc_max - soc_min) * (i as f64 / n_test as f64);
        let mut best_idx = 0;
        let mut best_val = f64::NEG_INFINITY;
        
        for (j, cut) in cuts.iter().enumerate() {
            let val = cut.alpha + cut.beta * soc;
            if val > best_val {
                best_val = val;
                best_idx = j;
            }
        }
        
        keep[best_idx] = true;
    }
    
    let new_cuts: Vec<Cut> = cuts.iter()
        .enumerate()
        .filter(|(i, _)| keep[*i])
        .map(|(_, c)| c.clone())
        .collect();
    
    *cuts = new_cuts;
    
    // Limit total cuts per stage
    if cuts.len() > 100 {
        cuts.truncate(100);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tig_challenges::energy_arbitrage::Difficulty;

    #[test]
    fn test_sddp_solver() {
        let difficulty = Difficulty {
            num_steps: 24,
            num_scenarios: 30,
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
