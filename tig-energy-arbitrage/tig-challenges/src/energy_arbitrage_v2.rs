/*!
 * Energy Arbitrage Challenge for TIG - Version 2
 *
 * Two-level challenge:
 * - Level 1: Single-asset temporal arbitrage with action-committed pricing
 * - Level 2: Portfolio arbitrage on a transmission-constrained network
 */

use anyhow::{anyhow, Result};
use rand::{Rng, SeedableRng};
use rand::rngs::{SmallRng, StdRng};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};

// ============================================================================
// Common Types
// ============================================================================

/// Battery physical parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatterySpec {
    pub id: usize,
    pub capacity_mwh: f64,
    pub power_charge_mw: f64,
    pub power_discharge_mw: f64,
    pub efficiency_charge: f64,
    pub efficiency_discharge: f64,
    pub soc_min_mwh: f64,
    pub soc_max_mwh: f64,
    pub soc_initial_mwh: f64,
}

impl BatterySpec {
    pub fn default_single() -> Self {
        Self {
            id: 0,
            capacity_mwh: 100.0,
            power_charge_mw: 25.0,
            power_discharge_mw: 25.0,
            efficiency_charge: 0.95,
            efficiency_discharge: 0.95,
            soc_min_mwh: 10.0,
            soc_max_mwh: 90.0,
            soc_initial_mwh: 50.0,
        }
    }
}

/// Market friction parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frictions {
    pub transaction_cost_per_mwh: f64,
    pub degradation_cost_per_mwh: f64,
    pub degradation_exponent: f64,
}

impl Default for Frictions {
    fn default() -> Self {
        Self {
            transaction_cost_per_mwh: 0.5,
            degradation_cost_per_mwh: 1.0,
            degradation_exponent: 2.0,
        }
    }
}

/// Market parameters for price generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketParams {
    pub volatility: f64,
    pub tail_probability: f64,
    pub tail_index: f64,
    pub mean_bias: f64,
}

impl Default for MarketParams {
    fn default() -> Self {
        Self {
            volatility: 0.2,
            tail_probability: 0.05,
            tail_index: 3.0,
            mean_bias: 0.0,
        }
    }
}

// ============================================================================
// Level 1: Single-Asset Temporal Arbitrage
// ============================================================================

/// Difficulty parameters for Level 1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level1Difficulty {
    /// Number of time steps (hours)
    pub num_steps: usize,
    /// Price volatility sigma in [0.1, 0.5]
    pub volatility: f64,
    /// Tail index alpha in (2, 5] - lower = heavier tails
    pub tail_index: f64,
    /// Transaction cost per MWh
    pub transaction_cost: f64,
    /// Degradation cost per MWh
    pub degradation_cost: f64,
    /// Required profit threshold (from SDDP upper bound)
    pub profit_threshold: f64,
}

impl Default for Level1Difficulty {
    fn default() -> Self {
        Self {
            num_steps: 24,
            volatility: 0.2,
            tail_index: 3.0,
            transaction_cost: 0.5,
            degradation_cost: 1.0,
            profit_threshold: 0.0, // Computed during generation
        }
    }
}

/// Action at a single time step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub charge_mw: f64,
    pub discharge_mw: f64,
}

impl Action {
    pub fn idle() -> Self {
        Self { charge_mw: 0.0, discharge_mw: 0.0 }
    }

    pub fn charge(power: f64) -> Self {
        Self { charge_mw: power.max(0.0), discharge_mw: 0.0 }
    }

    pub fn discharge(power: f64) -> Self {
        Self { charge_mw: 0.0, discharge_mw: power.max(0.0) }
    }
}

/// Transcript entry for verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptEntry {
    pub time_step: usize,
    pub action: Action,
    pub soc_mwh: f64,
    pub seed: [u8; 32],
    pub rt_price: f64,
    pub profit: f64,
}

/// Level 1 Challenge: Single battery with action-committed pricing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level1Challenge {
    pub seed: [u8; 32],
    pub difficulty: Level1Difficulty,
    pub battery: BatterySpec,
    pub frictions: Frictions,
    pub market: MarketParams,
    /// Day-ahead price curve (known at t=0)
    pub day_ahead_prices: Vec<f64>,
}

/// Level 1 Solution: A transcript of actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level1Solution {
    pub transcript: Vec<TranscriptEntry>,
}

impl Level1Challenge {
    /// Generate a Level 1 challenge instance
    pub fn generate_instance(seed: [u8; 32], difficulty: &Level1Difficulty) -> Result<Self> {
        let mut rng = SmallRng::from_seed(StdRng::from_seed(seed).gen());
        let t = difficulty.num_steps;

        // Generate day-ahead prices via GP
        let day_ahead_prices = generate_da_prices(&mut rng, t);

        let battery = BatterySpec::default_single();

        let frictions = Frictions {
            transaction_cost_per_mwh: difficulty.transaction_cost,
            degradation_cost_per_mwh: difficulty.degradation_cost,
            degradation_exponent: 2.0,
        };

        let market = MarketParams {
            volatility: difficulty.volatility,
            tail_probability: 0.05,
            tail_index: difficulty.tail_index,
            mean_bias: 0.0,
        };

        Ok(Level1Challenge {
            seed,
            difficulty: difficulty.clone(),
            battery,
            frictions,
            market,
            day_ahead_prices,
        })
    }

    /// Generate real-time price based on action-committed seed
    pub fn generate_rt_price(&self, da_price: f64, committed_seed: &[u8; 32]) -> f64 {
        let mut rng = SmallRng::from_seed(*committed_seed);

        // Normal noise component
        let u1: f64 = rng.gen::<f64>().max(1e-10);
        let u2: f64 = rng.gen();
        let noise = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();

        let base_price = da_price * (1.0 + self.market.mean_bias + self.market.volatility * noise);

        // Jump component (Pareto tail)
        let jump = if rng.gen::<f64>() < self.market.tail_probability {
            // Pareto distribution: X = (1 - U)^(-1/alpha) - 1
            let u: f64 = rng.gen::<f64>().max(1e-10);
            let pareto = (1.0 - u).powf(-1.0 / self.market.tail_index) - 1.0;
            da_price * pareto
        } else {
            0.0
        };

        (base_price + jump).max(0.0)
    }

    /// Commit to an action and compute next seed
    pub fn commit_action(
        &self,
        current_seed: &[u8; 32],
        action: &Action,
        time_step: usize,
        soc: f64,
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(current_seed);
        hasher.update(action.charge_mw.to_le_bytes());
        hasher.update(action.discharge_mw.to_le_bytes());
        hasher.update((time_step as u64).to_le_bytes());
        hasher.update(soc.to_le_bytes());

        let result = hasher.finalize();
        let mut new_seed = [0u8; 32];
        new_seed.copy_from_slice(&result);
        new_seed
    }

    /// Compute step profit
    pub fn compute_step_profit(
        &self,
        action: &Action,
        rt_price: f64,
        dt: f64,
    ) -> f64 {
        let c = action.charge_mw;
        let d = action.discharge_mw;

        // Revenue: (d - c) * price * dt
        let revenue = (d - c) * rt_price * dt;

        // Friction costs
        let tx_cost = self.frictions.transaction_cost_per_mwh * (c + d) * dt;
        let dod = (d * dt / self.battery.capacity_mwh).powf(self.frictions.degradation_exponent);
        let deg_cost = self.frictions.degradation_cost_per_mwh * dod;

        revenue - tx_cost - deg_cost
    }

    /// Apply action and return new SOC
    pub fn apply_action(&self, soc: f64, action: &Action, dt: f64) -> f64 {
        let new_soc = soc
            + self.battery.efficiency_charge * action.charge_mw * dt
            - action.discharge_mw * dt / self.battery.efficiency_discharge;

        new_soc.clamp(self.battery.soc_min_mwh, self.battery.soc_max_mwh)
    }

    /// Verify a solution transcript
    pub fn verify_solution(&self, solution: &Level1Solution) -> Result<f64> {
        let t = self.difficulty.num_steps;

        if solution.transcript.len() != t {
            return Err(anyhow!(
                "Transcript has {} entries, expected {}",
                solution.transcript.len(), t
            ));
        }

        // Verify initial conditions
        if solution.transcript[0].seed != self.seed {
            return Err(anyhow!("Initial seed mismatch"));
        }

        let mut current_seed = self.seed;
        let mut soc = self.battery.soc_initial_mwh;
        let mut total_profit = 0.0;

        for (i, entry) in solution.transcript.iter().enumerate() {
            // Verify time step
            if entry.time_step != i {
                return Err(anyhow!("Time step mismatch at entry {}", i));
            }

            // Verify seed
            if entry.seed != current_seed {
                return Err(anyhow!("Seed mismatch at step {}", i));
            }

            // Verify action constraints
            let action = &entry.action;
            if action.charge_mw < 0.0 || action.charge_mw > self.battery.power_charge_mw + 1e-6 {
                return Err(anyhow!("Charge power out of bounds at step {}", i));
            }
            if action.discharge_mw < 0.0 || action.discharge_mw > self.battery.power_discharge_mw + 1e-6 {
                return Err(anyhow!("Discharge power out of bounds at step {}", i));
            }
            if action.charge_mw > 1e-6 && action.discharge_mw > 1e-6 {
                return Err(anyhow!("Simultaneous charge/discharge at step {}", i));
            }

            // Regenerate RT price and verify
            let rt_price = self.generate_rt_price(self.day_ahead_prices[i], &current_seed);
            if (rt_price - entry.rt_price).abs() > 1e-6 {
                return Err(anyhow!(
                    "RT price mismatch at step {}: computed {}, transcript {}",
                    i, rt_price, entry.rt_price
                ));
            }

            // Verify SOC transition
            let new_soc = self.apply_action(soc, action, 1.0);
            if (new_soc - entry.soc_mwh).abs() > 1e-6 {
                return Err(anyhow!(
                    "SOC mismatch at step {}: computed {}, transcript {}",
                    i, new_soc, entry.soc_mwh
                ));
            }

            // Verify profit
            let profit = self.compute_step_profit(action, rt_price, 1.0);
            if (profit - entry.profit).abs() > 1e-6 {
                return Err(anyhow!(
                    "Profit mismatch at step {}: computed {}, transcript {}",
                    i, profit, entry.profit
                ));
            }

            total_profit += profit;
            soc = new_soc;

            // Commit action to get next seed
            current_seed = self.commit_action(&current_seed, action, i, soc);
        }

        // Check profit threshold
        if total_profit < self.difficulty.profit_threshold {
            return Err(anyhow!(
                "Total profit {:.2} below threshold {:.2}",
                total_profit, self.difficulty.profit_threshold
            ));
        }

        Ok(total_profit)
    }
}

// ============================================================================
// Level 2: Portfolio Arbitrage on Constrained Network
// ============================================================================

/// Network topology and parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    pub num_nodes: usize,
    pub num_lines: usize,
    /// Line definitions: (from_node, to_node)
    pub lines: Vec<(usize, usize)>,
    /// Line susceptances
    pub susceptances: Vec<f64>,
    /// Line flow limits (MW)
    pub flow_limits: Vec<f64>,
    /// Power Transfer Distribution Factor matrix (lines x nodes)
    pub ptdf: Vec<Vec<f64>>,
}

impl Network {
    /// Build a ring network with n nodes
    pub fn ring(n: usize, susceptance: f64, base_flow_limit: f64) -> Self {
        let mut lines = Vec::with_capacity(n);
        let mut susceptances = Vec::with_capacity(n);
        let mut flow_limits = Vec::with_capacity(n);

        for i in 0..n {
            let j = (i + 1) % n;
            lines.push((i, j));
            susceptances.push(susceptance);
            flow_limits.push(base_flow_limit);
        }

        let ptdf = Self::compute_ptdf(n, &lines, &susceptances);

        Network {
            num_nodes: n,
            num_lines: n,
            lines,
            susceptances,
            flow_limits,
            ptdf,
        }
    }

    /// Compute PTDF matrix using DC power flow
    fn compute_ptdf(n: usize, lines: &[(usize, usize)], susceptances: &[f64]) -> Vec<Vec<f64>> {
        // Build bus susceptance matrix B
        let mut b_matrix = vec![vec![0.0; n]; n];
        for (l, &(i, j)) in lines.iter().enumerate() {
            let b = susceptances[l];
            b_matrix[i][i] += b;
            b_matrix[j][j] += b;
            b_matrix[i][j] -= b;
            b_matrix[j][i] -= b;
        }

        // Remove slack bus (node 0) - create reduced matrix
        let n_red = n - 1;
        let mut b_red = vec![vec![0.0; n_red]; n_red];
        for i in 0..n_red {
            for j in 0..n_red {
                b_red[i][j] = b_matrix[i + 1][j + 1];
            }
        }

        // Invert reduced matrix (simple Gaussian elimination)
        let x_red = invert_matrix(&b_red);

        // Build full X matrix (with zeros for slack)
        let mut x = vec![vec![0.0; n]; n];
        for i in 0..n_red {
            for j in 0..n_red {
                x[i + 1][j + 1] = x_red[i][j];
            }
        }

        // Compute PTDF: PTDF[l,k] = b_l * (X[i,k] - X[j,k])
        let num_lines = lines.len();
        let mut ptdf = vec![vec![0.0; n]; num_lines];
        for (l, &(i, j)) in lines.iter().enumerate() {
            let b = susceptances[l];
            for k in 0..n {
                ptdf[l][k] = b * (x[i][k] - x[j][k]);
            }
        }

        ptdf
    }

    /// Compute line flows given nodal injections
    pub fn compute_flows(&self, injections: &[f64]) -> Vec<f64> {
        let mut flows = vec![0.0; self.num_lines];
        for l in 0..self.num_lines {
            for k in 0..self.num_nodes {
                flows[l] += self.ptdf[l][k] * injections[k];
            }
        }
        flows
    }

    /// Check if flows violate limits
    pub fn check_flow_limits(&self, flows: &[f64]) -> Option<(usize, f64)> {
        for (l, &flow) in flows.iter().enumerate() {
            if flow.abs() > self.flow_limits[l] + 1e-6 {
                return Some((l, flow));
            }
        }
        None
    }
}

/// Difficulty parameters for Level 2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level2Difficulty {
    /// Number of time steps
    pub num_steps: usize,
    /// Number of nodes in network
    pub num_nodes: usize,
    /// Number of batteries
    pub num_batteries: usize,
    /// Price volatility
    pub volatility: f64,
    /// Tail index
    pub tail_index: f64,
    /// Congestion factor: scales line limits (lower = more congestion)
    pub congestion_factor: f64,
    /// Portfolio heterogeneity: 0 = identical, 1 = diverse
    pub heterogeneity: f64,
    /// Congestion premium for LMP divergence
    pub congestion_premium: f64,
    /// Required profit threshold
    pub profit_threshold: f64,
}

impl Default for Level2Difficulty {
    fn default() -> Self {
        Self {
            num_steps: 24,
            num_nodes: 5,
            num_batteries: 3,
            volatility: 0.2,
            tail_index: 3.0,
            congestion_factor: 0.8,
            heterogeneity: 0.3,
            congestion_premium: 10.0,
            profit_threshold: 0.0,
        }
    }
}

/// Battery placement in network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacedBattery {
    pub spec: BatterySpec,
    pub node: usize,
}

/// Level 2 Challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level2Challenge {
    pub seed: [u8; 32],
    pub difficulty: Level2Difficulty,
    pub network: Network,
    pub batteries: Vec<PlacedBattery>,
    pub frictions: Frictions,
    pub market: MarketParams,
    /// Day-ahead prices per node (num_nodes x num_steps)
    pub day_ahead_prices: Vec<Vec<f64>>,
}

/// Portfolio action at a single time step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioAction {
    /// Actions for each battery: (charge_mw, discharge_mw)
    pub battery_actions: Vec<Action>,
}

/// Level 2 Solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level2Solution {
    /// Schedule for each time step
    pub schedule: Vec<PortfolioAction>,
}

impl Level2Challenge {
    /// Generate a Level 2 challenge instance
    pub fn generate_instance(seed: [u8; 32], difficulty: &Level2Difficulty) -> Result<Self> {
        let mut rng = SmallRng::from_seed(StdRng::from_seed(seed).gen());

        let n = difficulty.num_nodes;
        let t = difficulty.num_steps;
        let m = difficulty.num_batteries;

        // Build network
        let base_susceptance = 10.0;
        let base_flow_limit = 50.0 * difficulty.congestion_factor;
        let network = Network::ring(n, base_susceptance, base_flow_limit);

        // Generate batteries with heterogeneity
        let mut batteries = Vec::with_capacity(m);
        for b in 0..m {
            let node = b % n; // Distribute across nodes

            // Base parameters with heterogeneity
            let h = difficulty.heterogeneity;
            let capacity_factor = 1.0 + h * (rng.gen::<f64>() - 0.5) * 2.0;
            let power_factor = 1.0 + h * (rng.gen::<f64>() - 0.5) * 2.0;

            let spec = BatterySpec {
                id: b,
                capacity_mwh: 100.0 * capacity_factor,
                power_charge_mw: 25.0 * power_factor,
                power_discharge_mw: 25.0 * power_factor,
                efficiency_charge: 0.95 - 0.05 * h * rng.gen::<f64>(),
                efficiency_discharge: 0.95 - 0.05 * h * rng.gen::<f64>(),
                soc_min_mwh: 10.0 * capacity_factor,
                soc_max_mwh: 90.0 * capacity_factor,
                soc_initial_mwh: 50.0 * capacity_factor,
            };

            batteries.push(PlacedBattery { spec, node });
        }

        // Generate day-ahead prices for each node
        // Base price curve + node-specific variations
        let base_da = generate_da_prices(&mut rng, t);
        let mut day_ahead_prices = vec![vec![0.0; t]; n];

        for node in 0..n {
            for step in 0..t {
                // Add node-specific variation
                let variation = 1.0 + 0.1 * (rng.gen::<f64>() - 0.5);
                day_ahead_prices[node][step] = base_da[step] * variation;
            }
        }

        let frictions = Frictions::default();
        let market = MarketParams {
            volatility: difficulty.volatility,
            tail_probability: 0.05,
            tail_index: difficulty.tail_index,
            mean_bias: 0.0,
        };

        Ok(Level2Challenge {
            seed,
            difficulty: difficulty.clone(),
            network,
            batteries,
            frictions,
            market,
            day_ahead_prices,
        })
    }

    /// Generate real-time nodal prices
    pub fn generate_rt_prices(
        &self,
        time_step: usize,
        line_flows: &[f64],
        rng: &mut impl Rng,
    ) -> Vec<f64> {
        let n = self.network.num_nodes;
        let mut prices = Vec::with_capacity(n);

        // Check for congestion
        let mut congested_lines = Vec::new();
        for l in 0..self.network.num_lines {
            if line_flows[l].abs() > 0.9 * self.network.flow_limits[l] {
                congested_lines.push(l);
            }
        }

        for node in 0..n {
            let da = self.day_ahead_prices[node][time_step];

            // Base noise
            let u1: f64 = rng.gen::<f64>().max(1e-10);
            let u2: f64 = rng.gen();
            let noise = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();

            let mut price = da * (1.0 + self.market.volatility * noise);

            // Congestion premium for nodes adjacent to congested lines
            for &l in &congested_lines {
                let (from, to) = self.network.lines[l];
                if node == from || node == to {
                    let common_factor: f64 = rng.gen();
                    price += self.difficulty.congestion_premium * common_factor;
                }
            }

            // Tail spike
            if rng.gen::<f64>() < self.market.tail_probability {
                let u: f64 = rng.gen::<f64>().max(1e-10);
                let pareto = (1.0 - u).powf(-1.0 / self.market.tail_index) - 1.0;
                price += da * pareto;
            }

            prices.push(price.max(0.0));
        }

        prices
    }

    /// Compute portfolio profit for a single step
    pub fn compute_step_profit(
        &self,
        portfolio_action: &PortfolioAction,
        rt_prices: &[f64],
        dt: f64,
    ) -> f64 {
        let mut total_profit = 0.0;

        for (b, placed) in self.batteries.iter().enumerate() {
            let action = &portfolio_action.battery_actions[b];
            let price = rt_prices[placed.node];

            let c = action.charge_mw;
            let d = action.discharge_mw;

            let revenue = (d - c) * price * dt;
            let tx_cost = self.frictions.transaction_cost_per_mwh * (c + d) * dt;
            let dod = (d * dt / placed.spec.capacity_mwh).powf(self.frictions.degradation_exponent);
            let deg_cost = self.frictions.degradation_cost_per_mwh * dod;

            total_profit += revenue - tx_cost - deg_cost;
        }

        total_profit
    }

    /// Compute nodal injections from portfolio action
    pub fn compute_injections(&self, portfolio_action: &PortfolioAction) -> Vec<f64> {
        let mut injections = vec![0.0; self.network.num_nodes];

        for (b, placed) in self.batteries.iter().enumerate() {
            let action = &portfolio_action.battery_actions[b];
            // Positive injection = discharge (generation)
            injections[placed.node] += action.discharge_mw - action.charge_mw;
        }

        injections
    }

    /// Verify a solution
    pub fn verify_solution(&self, solution: &Level2Solution) -> Result<f64> {
        let t = self.difficulty.num_steps;
        let m = self.batteries.len();

        if solution.schedule.len() != t {
            return Err(anyhow!(
                "Schedule has {} steps, expected {}",
                solution.schedule.len(), t
            ));
        }

        let mut socs: Vec<f64> = self.batteries.iter()
            .map(|b| b.spec.soc_initial_mwh)
            .collect();

        let mut rng = SmallRng::from_seed(self.seed);
        let mut total_profit = 0.0;

        for step in 0..t {
            let portfolio_action = &solution.schedule[step];

            if portfolio_action.battery_actions.len() != m {
                return Err(anyhow!(
                    "Portfolio action at step {} has {} batteries, expected {}",
                    step, portfolio_action.battery_actions.len(), m
                ));
            }

            // Check action constraints for each battery
            for (b, placed) in self.batteries.iter().enumerate() {
                let action = &portfolio_action.battery_actions[b];

                if action.charge_mw < 0.0 || action.charge_mw > placed.spec.power_charge_mw + 1e-6 {
                    return Err(anyhow!(
                        "Battery {} charge power out of bounds at step {}", b, step
                    ));
                }
                if action.discharge_mw < 0.0 || action.discharge_mw > placed.spec.power_discharge_mw + 1e-6 {
                    return Err(anyhow!(
                        "Battery {} discharge power out of bounds at step {}", b, step
                    ));
                }
                if action.charge_mw > 1e-6 && action.discharge_mw > 1e-6 {
                    return Err(anyhow!(
                        "Battery {} simultaneous charge/discharge at step {}", b, step
                    ));
                }

                // Check SOC feasibility
                let new_soc = socs[b]
                    + placed.spec.efficiency_charge * action.charge_mw
                    - action.discharge_mw / placed.spec.efficiency_discharge;

                if new_soc < placed.spec.soc_min_mwh - 1e-6 || new_soc > placed.spec.soc_max_mwh + 1e-6 {
                    return Err(anyhow!(
                        "Battery {} SOC constraint violated at step {}: SOC={:.2}",
                        b, step, new_soc
                    ));
                }
            }

            // Check network flow constraints
            let injections = self.compute_injections(portfolio_action);
            let flows = self.network.compute_flows(&injections);

            if let Some((line, flow)) = self.network.check_flow_limits(&flows) {
                return Err(anyhow!(
                    "Line {} flow limit violated at step {}: |{:.2}| > {:.2}",
                    line, step, flow, self.network.flow_limits[line]
                ));
            }

            // Generate RT prices and compute profit
            let rt_prices = self.generate_rt_prices(step, &flows, &mut rng);
            let profit = self.compute_step_profit(portfolio_action, &rt_prices, 1.0);
            total_profit += profit;

            // Update SOCs
            for (b, placed) in self.batteries.iter().enumerate() {
                let action = &portfolio_action.battery_actions[b];
                socs[b] = socs[b]
                    + placed.spec.efficiency_charge * action.charge_mw
                    - action.discharge_mw / placed.spec.efficiency_discharge;
                socs[b] = socs[b].clamp(placed.spec.soc_min_mwh, placed.spec.soc_max_mwh);
            }
        }

        if total_profit < self.difficulty.profit_threshold {
            return Err(anyhow!(
                "Total profit {:.2} below threshold {:.2}",
                total_profit, self.difficulty.profit_threshold
            ));
        }

        Ok(total_profit)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate day-ahead prices using Gaussian Process with periodic kernel
fn generate_da_prices(rng: &mut impl Rng, n: usize) -> Vec<f64> {
    let kernel = GPKernel::new();
    let k = kernel.covariance_matrix(n);
    let l = cholesky(&k);

    // Generate standard normal samples via Box-Muller
    let z: Vec<f64> = (0..n)
        .map(|_| {
            let u1: f64 = rng.gen::<f64>().max(1e-10);
            let u2: f64 = rng.gen();
            (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos()
        })
        .collect();

    let mean_price = 50.0;
    let mut prices = vec![0.0; n];

    for i in 0..n {
        for j in 0..n {
            prices[i] += l[i][j] * z[j];
        }
        // Add diurnal pattern
        prices[i] += mean_price + 20.0 * (2.0 * std::f64::consts::PI * i as f64 / 24.0 - std::f64::consts::PI / 2.0).sin();
        prices[i] = prices[i].max(5.0);
    }

    prices
}

/// Gaussian Process kernel for price generation
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
            sigma_periodic: 15.0,
            length_periodic: 4.0,
            sigma_se: 5.0,
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
                    k[i][j] += 1e-6;
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

/// Simple matrix inversion via Gaussian elimination
fn invert_matrix(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return vec![];
    }

    // Augmented matrix [A | I]
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n + i] = 1.0;
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[k][i].abs() > aug[max_row][i].abs() {
                max_row = k;
            }
        }
        aug.swap(i, max_row);

        let pivot = aug[i][i];
        if pivot.abs() < 1e-12 {
            // Singular matrix - return identity as fallback
            return (0..n).map(|i| {
                (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()
            }).collect();
        }

        for j in 0..(2 * n) {
            aug[i][j] /= pivot;
        }

        for k in 0..n {
            if k != i {
                let factor = aug[k][i];
                for j in 0..(2 * n) {
                    aug[k][j] -= factor * aug[i][j];
                }
            }
        }
    }

    // Extract inverse
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = aug[i][n + j];
        }
    }
    inv
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level1_generation() {
        let difficulty = Level1Difficulty::default();
        let seed = [0u8; 32];
        let challenge = Level1Challenge::generate_instance(seed, &difficulty).unwrap();

        assert_eq!(challenge.day_ahead_prices.len(), 24);
        for &p in &challenge.day_ahead_prices {
            assert!(p >= 5.0);
        }
    }

    #[test]
    fn test_level1_action_commitment() {
        let difficulty = Level1Difficulty::default();
        let seed = [42u8; 32];
        let challenge = Level1Challenge::generate_instance(seed, &difficulty).unwrap();

        // Same action should produce same next seed
        let action = Action::charge(10.0);
        let seed1 = challenge.commit_action(&seed, &action, 0, 50.0);
        let seed2 = challenge.commit_action(&seed, &action, 0, 50.0);
        assert_eq!(seed1, seed2);

        // Different action should produce different seed
        let action2 = Action::discharge(10.0);
        let seed3 = challenge.commit_action(&seed, &action2, 0, 50.0);
        assert_ne!(seed1, seed3);
    }

    #[test]
    fn test_level2_generation() {
        let difficulty = Level2Difficulty::default();
        let seed = [0u8; 32];
        let challenge = Level2Challenge::generate_instance(seed, &difficulty).unwrap();

        assert_eq!(challenge.network.num_nodes, 5);
        assert_eq!(challenge.batteries.len(), 3);
        assert_eq!(challenge.day_ahead_prices.len(), 5);
        assert_eq!(challenge.day_ahead_prices[0].len(), 24);
    }

    #[test]
    fn test_network_ptdf() {
        let network = Network::ring(4, 10.0, 50.0);

        // Inject 1 MW at node 1, withdraw at node 0 (slack)
        let injections = vec![0.0, 1.0, 0.0, 0.0];
        let flows = network.compute_flows(&injections);

        // Flows should be distributed around the ring
        assert!(flows.iter().all(|&f| f.abs() <= 1.0));
    }

    #[test]
    fn test_level2_flow_limits() {
        let difficulty = Level2Difficulty {
            congestion_factor: 0.1, // Very tight limits
            ..Default::default()
        };
        let seed = [0u8; 32];
        let challenge = Level2Challenge::generate_instance(seed, &difficulty).unwrap();

        // Large action should violate flow limits
        let mut actions = Vec::new();
        for b in &challenge.batteries {
            actions.push(Action::discharge(b.spec.power_discharge_mw));
        }
        let portfolio_action = PortfolioAction { battery_actions: actions };

        let injections = challenge.compute_injections(&portfolio_action);
        let flows = challenge.network.compute_flows(&injections);

        // With tight limits, we expect a violation
        assert!(challenge.network.check_flow_limits(&flows).is_some());
    }
}
