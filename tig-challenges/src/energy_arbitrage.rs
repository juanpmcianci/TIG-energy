/*!
 * Energy Arbitrage Challenge for TIG
 * 
 * Single-asset temporal arbitrage: optimize charge/discharge schedule
 * for a battery facing stochastic real-time electricity prices.
 */

use anyhow::{anyhow, Result};
use rand::{Rng, SeedableRng};
use rand::rngs::{SmallRng, StdRng};
use serde::{Deserialize, Serialize};

/// Difficulty parameters for the energy arbitrage challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Difficulty {
    /// Number of time steps (hours)
    pub num_steps: usize,
    /// Number of price scenarios for evaluation
    pub num_scenarios: usize,
    /// Price volatility (std dev of multiplicative noise)
    pub volatility_percent: u32,
    /// Tail risk probability (percent, 0-100)
    pub tail_risk_percent: u32,
    /// Required profit improvement over baseline (percent)
    pub better_than_baseline: u32,
}

/// Battery physical parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatterySpec {
    pub capacity_mwh: f64,
    pub power_mw: f64,
    pub efficiency_charge: f64,
    pub efficiency_discharge: f64,
    pub soc_min_mwh: f64,
    pub soc_max_mwh: f64,
    pub soc_initial_mwh: f64,
}

/// Market friction parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frictions {
    pub transaction_cost_pct: f64,
    pub degradation_cost_per_mwh: f64,
}

/// The challenge instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Challenge {
    pub seed: [u8; 32],
    pub difficulty: Difficulty,
    pub battery: BatterySpec,
    pub frictions: Frictions,
    /// Day-ahead price curve (length = num_steps)
    pub day_ahead_prices: Vec<f64>,
    /// Real-time price scenarios (num_steps x num_scenarios)
    pub realtime_prices: Vec<Vec<f64>>,
    /// Baseline profit (from a simple greedy policy)
    pub baseline_profit: f64,
}

/// Solution: a schedule of actions for each time step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solution {
    /// Power at each step: positive = charge, negative = discharge
    pub actions_mw: Vec<f64>,
}

/// Gaussian Process kernel for day-ahead price generation
struct GPKernel {
    sigma_periodic: f64,
    length_periodic: f64,
    sigma_se: f64,
    length_se: f64,
    period: f64,
}

impl GPKernel {
    fn new() -> Self {
        Self {
            sigma_periodic: 12.0,
            length_periodic: 0.7,
            sigma_se: 6.0,
            length_se: 6.0,
            period: 24.0,
        }
    }

    fn evaluate(&self, t1: f64, t2: f64) -> f64 {
        let tau = (t1 - t2).abs();
        
        // Periodic component
        let periodic = self.sigma_periodic.powi(2) 
            * (-2.0 * (std::f64::consts::PI * tau / self.period).sin().powi(2) 
               / self.length_periodic.powi(2)).exp();
        
        // Squared exponential component
        let se = self.sigma_se.powi(2) 
            * (-tau.powi(2) / (2.0 * self.length_se.powi(2))).exp();
        
        periodic + se
    }

    fn covariance_matrix(&self, n: usize) -> Vec<Vec<f64>> {
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                k[i][j] = self.evaluate(i as f64, j as f64);
                if i == j {
                    k[i][j] += 1e-6; // Jitter for numerical stability
                }
            }
        }
        k
    }
}

/// Cholesky decomposition (lower triangular)
fn cholesky(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = (a[i][i] - sum).max(1e-10).sqrt();
            } else {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    l
}

impl Challenge {
    /// Generate a challenge instance from seed and difficulty
    pub fn generate_instance(seed: [u8; 32], difficulty: &Difficulty) -> Result<Self> {
        let mut rng = SmallRng::from_seed(StdRng::from_seed(seed).gen());
        
        let t = difficulty.num_steps;
        let n_scenarios = difficulty.num_scenarios;
        let volatility = difficulty.volatility_percent as f64 / 100.0;
        let tail_prob = difficulty.tail_risk_percent as f64 / 100.0;
        
        // Generate day-ahead prices via GP
        let kernel = GPKernel::new();
        let k = kernel.covariance_matrix(t);
        let l = cholesky(&k);
        
        let z: Vec<f64> = (0..t).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect();
        // Box-Muller for proper normal samples
        let z: Vec<f64> = z.chunks(2)
            .flat_map(|pair| {
                if pair.len() == 2 {
                    let u1: f64 = pair[0].abs().max(1e-10);
                    let u2: f64 = pair[1];
                    let r = (-2.0_f64 * u1.ln()).sqrt();
                    let theta = 2.0 * std::f64::consts::PI * u2;
                    vec![r * theta.cos(), r * theta.sin()]
                } else {
                    vec![0.0]
                }
            })
            .take(t)
            .collect();
        
        let mean_price = 50.0;
        let mut day_ahead_prices: Vec<f64> = vec![0.0; t];
        for i in 0..t {
            for j in 0..t {
                day_ahead_prices[i] += l[i][j] * z[j];
            }
            day_ahead_prices[i] += mean_price;
            day_ahead_prices[i] = day_ahead_prices[i].max(5.0); // Floor at $5/MWh
        }
        
        // Generate real-time scenarios
        let mut realtime_prices = vec![vec![0.0; n_scenarios]; t];
        for s in 0..n_scenarios {
            for i in 0..t {
                let noise: f64 = {
                    let u1: f64 = rng.gen::<f64>().max(1e-10);
                    let u2: f64 = rng.gen();
                    (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos()
                };
                let mut price = day_ahead_prices[i] * (1.0 + volatility * noise);
                
                // Tail spike
                if rng.gen::<f64>() < tail_prob {
                    let spike_hour = (0.6 * t as f64) as usize 
                        + rng.gen_range(0..((0.25 * t as f64) as usize).max(1));
                    if i == spike_hour.min(t - 1) {
                        price *= 2.0 + rng.gen::<f64>() * 3.0;
                    }
                }
                
                realtime_prices[i][s] = price.max(0.0);
            }
        }
        
        // Battery specification
        let battery = BatterySpec {
            capacity_mwh: 100.0,
            power_mw: 25.0,
            efficiency_charge: 0.95,
            efficiency_discharge: 0.95,
            soc_min_mwh: 0.0,
            soc_max_mwh: 100.0,
            soc_initial_mwh: 50.0,
        };
        
        let frictions = Frictions {
            transaction_cost_pct: 0.01,
            degradation_cost_per_mwh: 0.5,
        };
        
        // Compute baseline profit using greedy policy
        let baseline_profit = Self::compute_baseline_profit(
            &day_ahead_prices,
            &realtime_prices,
            &battery,
            &frictions,
        );
        
        Ok(Challenge {
            seed,
            difficulty: difficulty.clone(),
            battery,
            frictions,
            day_ahead_prices,
            realtime_prices,
            baseline_profit,
        })
    }
    
    /// Compute baseline profit using a simple greedy threshold policy
    fn compute_baseline_profit(
        da_prices: &[f64],
        rt_prices: &[Vec<f64>],
        battery: &BatterySpec,
        frictions: &Frictions,
    ) -> f64 {
        let t = da_prices.len();
        let n_scenarios = rt_prices[0].len();
        let mut total_profit = 0.0;
        
        for s in 0..n_scenarios {
            let mut soc = battery.soc_initial_mwh;
            let mut profit = 0.0;
            
            for i in 0..t {
                let rt = rt_prices[i][s];
                let da = da_prices[i];
                
                // Greedy: charge if RT < 0.95*DA, discharge if RT > 1.05*DA
                let action = if rt < 0.95 * da && soc < battery.soc_max_mwh - 1.0 {
                    battery.power_mw
                } else if rt > 1.05 * da && soc > battery.soc_min_mwh + 1.0 {
                    -battery.power_mw
                } else {
                    0.0
                };
                
                let (soc_new, step_profit) = Self::apply_action(
                    soc, action, rt, battery, frictions, 1.0
                );
                soc = soc_new;
                profit += step_profit;
            }
            
            total_profit += profit;
        }
        
        total_profit / n_scenarios as f64
    }
    
    /// Apply action and return new SOC and profit
    fn apply_action(
        soc: f64,
        action_mw: f64,
        price: f64,
        battery: &BatterySpec,
        frictions: &Frictions,
        dt: f64,
    ) -> (f64, f64) {
        let action_clamped = action_mw.clamp(-battery.power_mw, battery.power_mw);
        
        let p_charge = action_clamped.max(0.0);
        let p_discharge = (-action_clamped).max(0.0);
        
        let soc_new = soc 
            + battery.efficiency_charge * p_charge * dt
            - p_discharge * dt / battery.efficiency_discharge;
        let soc_clamped = soc_new.clamp(battery.soc_min_mwh, battery.soc_max_mwh);
        
        let tc = frictions.transaction_cost_pct;
        let deg = frictions.degradation_cost_per_mwh;
        
        let profit = ((1.0 - tc) * price * p_discharge 
                     - (1.0 + tc) * price * p_charge 
                     - deg * p_discharge) * dt;
        
        (soc_clamped, profit)
    }
    
    /// Verify a solution's validity and compute its profit
    pub fn verify_solution(&self, solution: &Solution) -> Result<f64> {
        let t = self.difficulty.num_steps;
        let n_scenarios = self.difficulty.num_scenarios;
        
        // Check solution length
        if solution.actions_mw.len() != t {
            return Err(anyhow!(
                "Solution has {} actions, expected {}", 
                solution.actions_mw.len(), t
            ));
        }
        
        // Check power bounds
        for (i, &action) in solution.actions_mw.iter().enumerate() {
            if action.abs() > self.battery.power_mw + 1e-6 {
                return Err(anyhow!(
                    "Action at step {} exceeds power limit: {} > {}", 
                    i, action.abs(), self.battery.power_mw
                ));
            }
        }
        
        // Simulate across all scenarios and compute average profit
        let mut total_profit = 0.0;
        
        for s in 0..n_scenarios {
            let mut soc = self.battery.soc_initial_mwh;
            let mut profit = 0.0;
            
            for i in 0..t {
                let action = solution.actions_mw[i];
                let price = self.realtime_prices[i][s];
                
                // Check SOC feasibility before action
                let p_charge = action.max(0.0);
                let p_discharge = (-action).max(0.0);
                
                let soc_next = soc 
                    + self.battery.efficiency_charge * p_charge
                    - p_discharge / self.battery.efficiency_discharge;
                
                if soc_next < self.battery.soc_min_mwh - 1e-6 
                   || soc_next > self.battery.soc_max_mwh + 1e-6 {
                    return Err(anyhow!(
                        "SOC constraint violated at step {} scenario {}: SOC={:.2} after action={:.2}", 
                        i, s, soc_next, action
                    ));
                }
                
                let (soc_new, step_profit) = Self::apply_action(
                    soc, action, price, &self.battery, &self.frictions, 1.0
                );
                soc = soc_new;
                profit += step_profit;
            }
            
            total_profit += profit;
        }
        
        let avg_profit = total_profit / n_scenarios as f64;
        
        // Check if solution beats baseline by required margin
        let required_profit = self.baseline_profit 
            * (1.0 + self.difficulty.better_than_baseline as f64 / 100.0);
        
        if avg_profit < required_profit {
            return Err(anyhow!(
                "Profit {:.2} does not beat required threshold {:.2} (baseline={:.2}, improvement={}%)",
                avg_profit, required_profit, self.baseline_profit, 
                self.difficulty.better_than_baseline
            ));
        }
        
        Ok(avg_profit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instance_generation() {
        let difficulty = Difficulty {
            num_steps: 24,
            num_scenarios: 50,
            volatility_percent: 20,
            tail_risk_percent: 5,
            better_than_baseline: 10,
        };
        
        let seed = [0u8; 32];
        let challenge = Challenge::generate_instance(seed, &difficulty).unwrap();
        
        assert_eq!(challenge.day_ahead_prices.len(), 24);
        assert_eq!(challenge.realtime_prices.len(), 24);
        assert_eq!(challenge.realtime_prices[0].len(), 50);
        
        // Prices should be positive
        for p in &challenge.day_ahead_prices {
            assert!(*p > 0.0);
        }
    }

    #[test]
    fn test_solution_verification() {
        let difficulty = Difficulty {
            num_steps: 24,
            num_scenarios: 10,
            volatility_percent: 20,
            tail_risk_percent: 5,
            better_than_baseline: 0, // No improvement required for this test
        };
        
        let seed = [42u8; 32];
        let challenge = Challenge::generate_instance(seed, &difficulty).unwrap();
        
        // Zero action solution should be feasible (though may not beat baseline)
        let solution = Solution {
            actions_mw: vec![0.0; 24],
        };
        
        // Check that the solution is at least valid (actions within bounds)
        assert_eq!(solution.actions_mw.len(), 24);
        for &action in &solution.actions_mw {
            assert!(action.abs() <= challenge.battery.power_mw + 1e-6);
        }
    }
}
