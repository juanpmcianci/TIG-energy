/*!
 * Energy Arbitrage Challenge for TIG - Network Version
 *
 * Implementation of the "Optimal Arbitrage of Networked Energy Storage" challenge
 * as specified in tig_level_2_spec.tex.
 *
 * Portfolio arbitrage on a transmission-constrained network with 5 tracks.
 *
 * Key features:
 * - Action-committed pricing via SHA-256 hash chain
 * - PTDF-based DC power flow with slack bus balancing
 * - Spatially correlated price shocks with congestion premium
 * - Five tracks with progressively harder parameter regimes
 */

use crate::QUALITY_PRECISION;
use anyhow::{anyhow, Result};
use rand::{Rng, SeedableRng};
use rand::rngs::{SmallRng, StdRng};
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::f64::consts::PI;

mod baselines;

// ============================================================================
// Default Constants (from spec Appendix A)
// ============================================================================

/// Default constants as specified in the challenge specification.
pub mod constants {
    /// Time step duration in hours (15 minutes)
    pub const DELTA_T: f64 = 0.25;

    /// Slack bus index (1-indexed in spec, 0-indexed in code)
    pub const SLACK_BUS: usize = 0;

    /// Action quantization step (MW)
    pub const Q_U: f64 = 0.01;

    /// SOC quantization step (MWh)
    pub const Q_E: f64 = 0.01;

    /// Fractional SOC lower bound
    pub const E_MIN_FRAC: f64 = 0.10;

    /// Fractional SOC upper bound
    pub const E_MAX_FRAC: f64 = 0.90;

    /// Initial SOC fraction
    pub const E_INIT_FRAC: f64 = 0.50;

    /// Default charge efficiency
    pub const ETA_CHARGE: f64 = 0.95;

    /// Default discharge efficiency
    pub const ETA_DISCHARGE: f64 = 0.95;

    /// Transaction cost ($/MWh)
    pub const KAPPA_TX: f64 = 0.25;

    /// Degradation scale ($)
    pub const KAPPA_DEG: f64 = 1.00;

    /// Degradation exponent
    pub const BETA_DEG: f64 = 2.0;

    /// RT bias term
    pub const MU_BIAS: f64 = 0.0;

    /// Spatial correlation parameter
    pub const RHO_SPATIAL: f64 = 0.70;

    /// Congestion premium scale ($/MWh)
    pub const GAMMA_PRICE: f64 = 20.0;

    /// Congestion proximity threshold
    pub const TAU_CONG: f64 = 0.97;

    /// Jump probability
    pub const RHO_JUMP: f64 = 0.02;

    /// Pareto tail index
    pub const ALPHA_TAIL: f64 = 3.5;

    /// RT price floor ($/MWh)
    pub const LAMBDA_MIN: f64 = -200.0;

    /// RT price cap ($/MWh)
    pub const LAMBDA_MAX: f64 = 5000.0;

    /// DA price floor ($/MWh)
    pub const LAMBDA_DA_MIN: f64 = 0.0;

    /// Flow feasibility tolerance (per-unit)
    pub const EPS_FLOW: f64 = 1e-6;

    /// SOC feasibility tolerance (MWh)
    pub const EPS_SOC: f64 = 1e-9;

    /// Nominal battery capacity (MWh)
    pub const NOMINAL_CAPACITY: f64 = 100.0;

    /// Nominal battery power (MW)
    pub const NOMINAL_POWER: f64 = 25.0;

    /// Nominal line flow limit (MW)
    pub const NOMINAL_FLOW_LIMIT: f64 = 100.0;

    /// Base susceptance for network generation
    pub const BASE_SUSCEPTANCE: f64 = 10.0;

    /// Mean DA price ($/MWh)
    pub const MEAN_DA_PRICE: f64 = 50.0;

    /// DA price amplitude ($/MWh)
    pub const DA_AMPLITUDE: f64 = 20.0;
}

// ============================================================================
// Track Definitions (from spec Appendix B)
// ============================================================================

/// TIG Track identifier for the challenge.
/// Each track defines a specific parameter regime with increasing difficulty.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Track {
    /// Track 1: Correctness-and-baseline regime
    /// Small network, nominal limits, low volatility, rare thin-tailed spikes
    Track1,

    /// Track 2: Meaningful congestion with increased stochasticity
    /// Tighter limits, more volatility, more diverse fleet
    Track2,

    /// Track 3: Multi-day horizon with larger scale
    /// Longer horizon, larger network, more frequent spikes
    Track3,

    /// Track 4: High network density with frequent congestion
    /// Dense network, heavy tails, risk management critical
    Track4,

    /// Track 5: Capstone regime
    /// Largest scale, tightest limits, heaviest tails
    Track5,
}

impl Track {
    /// Get track parameters as specified in the challenge spec.
    pub fn parameters(&self) -> TrackParameters {
        match self {
            Track::Track1 => TrackParameters {
                num_nodes: 20,
                num_lines: 30,
                num_batteries: 10,
                num_steps: 96,
                gamma_cong: 1.00,
                sigma: 0.10,
                rho_jump: 0.01,
                alpha: 4.0,
                heterogeneity: 0.2,
            },
            Track::Track2 => TrackParameters {
                num_nodes: 40,
                num_lines: 60,
                num_batteries: 20,
                num_steps: 96,
                gamma_cong: 0.80,
                sigma: 0.15,
                rho_jump: 0.02,
                alpha: 3.5,
                heterogeneity: 0.4,
            },
            Track::Track3 => TrackParameters {
                num_nodes: 80,
                num_lines: 120,
                num_batteries: 40,
                num_steps: 192,
                gamma_cong: 0.60,
                sigma: 0.20,
                rho_jump: 0.03,
                alpha: 3.0,
                heterogeneity: 0.6,
            },
            Track::Track4 => TrackParameters {
                num_nodes: 100,
                num_lines: 200,
                num_batteries: 60,
                num_steps: 192,
                gamma_cong: 0.50,
                sigma: 0.25,
                rho_jump: 0.04,
                alpha: 2.7,
                heterogeneity: 0.8,
            },
            Track::Track5 => TrackParameters {
                num_nodes: 150,
                num_lines: 300,
                num_batteries: 100,
                num_steps: 192,
                gamma_cong: 0.40,
                sigma: 0.30,
                rho_jump: 0.05,
                alpha: 2.5,
                heterogeneity: 1.0,
            },
        }
    }

    /// Get all tracks in order of difficulty
    pub fn all() -> Vec<Track> {
        vec![Track::Track1, Track::Track2, Track::Track3, Track::Track4, Track::Track5]
    }
}

/// Track-specific parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackParameters {
    /// Number of nodes (n)
    pub num_nodes: usize,
    /// Number of lines (L)
    pub num_lines: usize,
    /// Number of batteries (m)
    pub num_batteries: usize,
    /// Number of time steps (H)
    pub num_steps: usize,
    /// Congestion scaling factor (γ_cong) - scales line limits
    pub gamma_cong: f64,
    /// Volatility (σ)
    pub sigma: f64,
    /// Jump probability (ρ_jump)
    pub rho_jump: f64,
    /// Pareto tail index (α)
    pub alpha: f64,
    /// Fleet heterogeneity (h) - 0 = identical, 1 = 3x spread
    pub heterogeneity: f64,
}

// ============================================================================
// Common Types
// ============================================================================

/// Battery physical parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatterySpec {
    /// Battery identifier
    pub id: usize,
    /// Energy capacity (MWh)
    pub capacity_mwh: f64,
    /// Maximum charge power (MW)
    pub power_charge_mw: f64,
    /// Maximum discharge power (MW)
    pub power_discharge_mw: f64,
    /// Charge efficiency (η^c)
    pub efficiency_charge: f64,
    /// Discharge efficiency (η^d)
    pub efficiency_discharge: f64,
    /// SOC lower bound (MWh)
    pub soc_min_mwh: f64,
    /// SOC upper bound (MWh)
    pub soc_max_mwh: f64,
    /// Initial SOC (MWh)
    pub soc_initial_mwh: f64,
}

impl BatterySpec {
    /// Create battery with given capacity using default parameters
    pub fn with_capacity(id: usize, capacity_mwh: f64, power_mw: f64) -> Self {
        Self {
            id,
            capacity_mwh,
            power_charge_mw: power_mw,
            power_discharge_mw: power_mw,
            efficiency_charge: constants::ETA_CHARGE,
            efficiency_discharge: constants::ETA_DISCHARGE,
            soc_min_mwh: constants::E_MIN_FRAC * capacity_mwh,
            soc_max_mwh: constants::E_MAX_FRAC * capacity_mwh,
            soc_initial_mwh: constants::E_INIT_FRAC * capacity_mwh,
        }
    }
}

/// Market friction parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Frictions {
    /// Transaction cost ($/MWh) - κ_tx
    pub transaction_cost_per_mwh: f64,
    /// Degradation scale ($) - κ_deg
    pub degradation_scale: f64,
    /// Degradation exponent - β
    pub degradation_exponent: f64,
}

impl Default for Frictions {
    fn default() -> Self {
        Self {
            transaction_cost_per_mwh: constants::KAPPA_TX,
            degradation_scale: constants::KAPPA_DEG,
            degradation_exponent: constants::BETA_DEG,
        }
    }
}

/// Market parameters for price generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketParams {
    /// Volatility (σ)
    pub volatility: f64,
    /// Jump probability (ρ_jump)
    pub jump_probability: f64,
    /// Pareto tail index (α)
    pub tail_index: f64,
    /// RT bias term (μ)
    pub mean_bias: f64,
    /// Spatial correlation (ρ_sp)
    pub spatial_correlation: f64,
    /// Congestion premium scale (γ_price)
    pub congestion_premium: f64,
    /// Congestion proximity threshold (τ_cong)
    pub congestion_threshold: f64,
}

impl Default for MarketParams {
    fn default() -> Self {
        Self {
            volatility: 0.20,
            jump_probability: constants::RHO_JUMP,
            tail_index: constants::ALPHA_TAIL,
            mean_bias: constants::MU_BIAS,
            spatial_correlation: constants::RHO_SPATIAL,
            congestion_premium: constants::GAMMA_PRICE,
            congestion_threshold: constants::TAU_CONG,
        }
    }
}

// ============================================================================
// Network and Challenge Types
// ============================================================================

/// Network topology and DC power flow parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    /// Number of nodes (n)
    pub num_nodes: usize,
    /// Number of lines (L)
    pub num_lines: usize,
    /// Line definitions: (from_node, to_node)
    pub lines: Vec<(usize, usize)>,
    /// Line susceptances (b_l)
    pub susceptances: Vec<f64>,
    /// Nominal line flow limits (MW)
    pub nominal_flow_limits: Vec<f64>,
    /// Effective line flow limits after congestion scaling (MW)
    pub flow_limits: Vec<f64>,
    /// Power Transfer Distribution Factor matrix (L x n)
    pub ptdf: Vec<Vec<f64>>,
    /// Slack bus index
    pub slack_bus: usize,
    /// Incidence: which lines are incident to each node
    pub node_incident_lines: Vec<Vec<usize>>,
}

impl Network {
    /// Build a network with given topology
    pub fn new(
        num_nodes: usize,
        lines: Vec<(usize, usize)>,
        susceptances: Vec<f64>,
        nominal_flow_limits: Vec<f64>,
        gamma_cong: f64,
    ) -> Self {
        let num_lines = lines.len();
        let slack_bus = constants::SLACK_BUS;

        // Compute PTDF matrix
        let ptdf = Self::compute_ptdf(num_nodes, &lines, &susceptances, slack_bus);

        // Apply congestion scaling to get effective limits
        let flow_limits: Vec<f64> = nominal_flow_limits
            .iter()
            .map(|&f| f * gamma_cong)
            .collect();

        // Build node-to-incident-lines mapping
        let mut node_incident_lines = vec![Vec::new(); num_nodes];
        for (l, &(from, to)) in lines.iter().enumerate() {
            node_incident_lines[from].push(l);
            node_incident_lines[to].push(l);
        }

        Network {
            num_nodes,
            num_lines,
            lines,
            susceptances,
            nominal_flow_limits,
            flow_limits,
            ptdf,
            slack_bus,
            node_incident_lines,
        }
    }

    /// Generate a connected network with given parameters
    pub fn generate(rng: &mut impl Rng, num_nodes: usize, num_lines: usize, gamma_cong: f64) -> Self {
        // Start with a spanning tree (n-1 lines), then add extra lines
        let mut lines = Vec::new();
        let mut susceptances = Vec::new();
        let mut nominal_limits = Vec::new();

        // Phase 1: Create spanning tree using random edges
        let mut connected = vec![false; num_nodes];
        connected[0] = true;
        let mut connected_count = 1;

        while connected_count < num_nodes {
            // Pick a random unconnected node
            let unconnected: Vec<usize> = (0..num_nodes)
                .filter(|&i| !connected[i])
                .collect();
            let new_node = unconnected[rng.gen_range(0..unconnected.len())];

            // Connect to a random connected node
            let connected_nodes: Vec<usize> = (0..num_nodes)
                .filter(|&i| connected[i])
                .collect();
            let existing = connected_nodes[rng.gen_range(0..connected_nodes.len())];

            let (from, to) = if new_node < existing {
                (new_node, existing)
            } else {
                (existing, new_node)
            };

            lines.push((from, to));
            susceptances.push(constants::BASE_SUSCEPTANCE * (0.8 + 0.4 * rng.gen::<f64>()));
            nominal_limits.push(constants::NOMINAL_FLOW_LIMIT * (0.8 + 0.4 * rng.gen::<f64>()));

            connected[new_node] = true;
            connected_count += 1;
        }

        // Phase 2: Add extra lines to reach target by sampling from non-existing edges
        let extra_needed = num_lines.saturating_sub(lines.len());
        if extra_needed > 0 {
            // Build set of existing edges for O(1) lookup
            let existing: std::collections::HashSet<(usize, usize)> =
                lines.iter().cloned().collect();

            // Build list of all non-existing edges
            let mut candidates: Vec<(usize, usize)> = Vec::new();
            for i in 0..num_nodes {
                for j in (i + 1)..num_nodes {
                    if !existing.contains(&(i, j)) {
                        candidates.push((i, j));
                    }
                }
            }

            // Randomly select using Fisher-Yates partial shuffle
            let to_add = extra_needed.min(candidates.len());
            for k in 0..to_add {
                let idx = rng.gen_range(k..candidates.len());
                candidates.swap(k, idx);

                let (from, to) = candidates[k];
                lines.push((from, to));
                susceptances.push(constants::BASE_SUSCEPTANCE * (0.8 + 0.4 * rng.gen::<f64>()));
                nominal_limits.push(constants::NOMINAL_FLOW_LIMIT * (0.8 + 0.4 * rng.gen::<f64>()));
            }
        }

        Self::new(num_nodes, lines, susceptances, nominal_limits, gamma_cong)
    }

    /// Compute PTDF matrix using DC power flow
    fn compute_ptdf(
        n: usize,
        lines: &[(usize, usize)],
        susceptances: &[f64],
        slack_bus: usize,
    ) -> Vec<Vec<f64>> {
        if n == 0 || lines.is_empty() {
            return vec![];
        }

        // Build bus susceptance matrix B (n x n)
        let mut b_matrix = vec![vec![0.0; n]; n];
        for (l, &(i, j)) in lines.iter().enumerate() {
            let b = susceptances[l];
            b_matrix[i][i] += b;
            b_matrix[j][j] += b;
            b_matrix[i][j] -= b;
            b_matrix[j][i] -= b;
        }

        // Remove slack bus - create reduced (n-1) x (n-1) matrix
        let n_red = n - 1;
        let mut b_red = vec![vec![0.0; n_red]; n_red];
        let mut row_map = Vec::with_capacity(n_red);
        for i in 0..n {
            if i != slack_bus {
                row_map.push(i);
            }
        }

        for (ri, &i) in row_map.iter().enumerate() {
            for (rj, &j) in row_map.iter().enumerate() {
                b_red[ri][rj] = b_matrix[i][j];
            }
        }

        // Invert reduced matrix
        let x_red = invert_matrix(&b_red);

        // Build full X matrix (with zeros for slack)
        let mut x = vec![vec![0.0; n]; n];
        for (ri, &i) in row_map.iter().enumerate() {
            for (rj, &j) in row_map.iter().enumerate() {
                x[i][j] = x_red[ri][rj];
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

    /// Compute line flows given nodal injections (including slack balancing)
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
    /// Returns Some((line_id, violation_amount)) if violated
    pub fn check_flow_limits(&self, flows: &[f64]) -> Option<(usize, f64)> {
        for (l, &flow) in flows.iter().enumerate() {
            let violation = flow.abs() - self.flow_limits[l];
            if violation > constants::EPS_FLOW * self.flow_limits[l] {
                return Some((l, flow));
            }
        }
        None
    }

    /// Compute congestion indicators for each node based on flows
    /// A node is congested if any incident line has |flow| >= τ_cong * limit
    pub fn compute_congestion_indicators(&self, flows: &[f64], tau: f64) -> Vec<bool> {
        let mut indicators = vec![false; self.num_nodes];

        for (l, &flow) in flows.iter().enumerate() {
            if flow.abs() >= tau * self.flow_limits[l] {
                let (from, to) = self.lines[l];
                indicators[from] = true;
                indicators[to] = true;
            }
        }

        indicators
    }
}

/// Difficulty parameters for the challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level2Difficulty {
    /// Track identifier (determines most parameters)
    pub track: Track,
    /// Override: number of time steps (H)
    pub num_steps: Option<usize>,
    /// Override: number of nodes (n)
    pub num_nodes: Option<usize>,
    /// Override: number of batteries (m)
    pub num_batteries: Option<usize>,
    /// Override: congestion factor (γ_cong)
    pub congestion_factor: Option<f64>,
    /// Override: volatility (σ)
    pub volatility: Option<f64>,
    /// Override: jump probability (ρ_jump)
    pub jump_probability: Option<f64>,
    /// Override: tail index (α)
    pub tail_index: Option<f64>,
    /// Override: heterogeneity (h)
    pub heterogeneity: Option<f64>,
    /// Required profit threshold
    pub profit_threshold: f64,
}

impl Level2Difficulty {
    /// Create difficulty from track (uses track defaults)
    pub fn from_track(track: Track) -> Self {
        Self {
            track,
            num_steps: None,
            num_nodes: None,
            num_batteries: None,
            congestion_factor: None,
            volatility: None,
            jump_probability: None,
            tail_index: None,
            heterogeneity: None,
            profit_threshold: 0.0,
        }
    }

    /// Get effective parameters (track defaults with overrides applied)
    pub fn effective_params(&self) -> TrackParameters {
        let mut params = self.track.parameters();

        if let Some(v) = self.num_steps { params.num_steps = v; }
        if let Some(v) = self.num_nodes { params.num_nodes = v; }
        if let Some(v) = self.num_batteries { params.num_batteries = v; }
        if let Some(v) = self.congestion_factor { params.gamma_cong = v; }
        if let Some(v) = self.volatility { params.sigma = v; }
        if let Some(v) = self.jump_probability { params.rho_jump = v; }
        if let Some(v) = self.tail_index { params.alpha = v; }
        if let Some(v) = self.heterogeneity { params.heterogeneity = v; }

        params
    }
}

impl Default for Level2Difficulty {
    fn default() -> Self {
        Self::from_track(Track::Track1)
    }
}

/// Battery placement in network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlacedBattery {
    pub spec: BatterySpec,
    pub node: usize,
}

/// The Challenge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level2Challenge {
    /// Initial commitment seed (s_0)
    pub seed: [u8; 32],
    /// Difficulty parameters
    pub difficulty: Level2Difficulty,
    /// Network topology and parameters
    pub network: Network,
    /// Battery portfolio with placements
    pub batteries: Vec<PlacedBattery>,
    /// Market frictions
    pub frictions: Frictions,
    /// Market price parameters
    pub market: MarketParams,
    /// Exogenous nodal injections (n x H)
    pub exogenous_injections: Vec<Vec<f64>>,
    /// Day-ahead nodal price forecast (n x H)
    pub day_ahead_prices: Vec<Vec<f64>>,
}

/// Signed power action for a battery at one time step
/// u > 0 means discharge (net injection), u < 0 means charge (net withdrawal)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedAction {
    /// Signed power: positive = discharge, negative = charge
    pub power_mw: f64,
}

impl SignedAction {
    pub fn new(power: f64) -> Self {
        Self { power_mw: power }
    }

    pub fn idle() -> Self {
        Self { power_mw: 0.0 }
    }

    /// Convert to charge/discharge decomposition
    pub fn decompose(&self) -> (f64, f64) {
        let c = (-self.power_mw).max(0.0); // charge if negative
        let d = self.power_mw.max(0.0);     // discharge if positive
        (c, d)
    }

    /// Quantize for hashing
    pub fn quantized(&self) -> i64 {
        (self.power_mw / constants::Q_U).round() as i64
    }
}

/// Portfolio action at a single time step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortfolioAction {
    /// Signed power actions for each battery
    pub actions: Vec<SignedAction>,
}

/// Simulation state visible to innovators.
/// Contains only what an algorithm needs to make its next decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// Current time step index (0-based)
    pub time_step: usize,
    /// Current state-of-charge for each battery (MWh)
    pub socs: Vec<f64>,
    /// Real-time nodal prices at the current time step ($/MWh)
    pub rt_prices: Vec<f64>,
}

/// The Solution: schedule of portfolio actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Level2Solution {
    /// Schedule for each time step
    pub schedule: Vec<PortfolioAction>,
}

impl Level2Solution {
    pub fn new() -> Self {
        Self { schedule: Vec::new() }
    }
}

// Monorepo naming aliases
pub type Challenge = Level2Challenge;
pub type Solution = Level2Solution;
pub type Difficulty = Level2Difficulty;

impl Level2Challenge {
    /// Generate a challenge instance
    pub fn generate_instance(seed: &[u8; 32], track: &Track) -> Result<Self> {
        let difficulty = Level2Difficulty::from_track(*track);
        Self::generate_instance_with_difficulty(seed, &difficulty)
    }

    /// Generate a challenge instance with explicit difficulty overrides
    pub fn generate_instance_with_difficulty(
        seed: &[u8; 32],
        difficulty: &Level2Difficulty,
    ) -> Result<Self> {
        let mut rng = SmallRng::from_seed(StdRng::from_seed(*seed).gen());
        let params = difficulty.effective_params();

        let n = params.num_nodes;
        let num_lines = params.num_lines;
        let m = params.num_batteries;
        let h = params.num_steps;

        // Generate network
        let network = Network::generate(&mut rng, n, num_lines, params.gamma_cong);

        // Generate batteries with heterogeneity
        let batteries = Self::generate_batteries(&mut rng, n, m, params.heterogeneity);

        // Generate exogenous injections
        let exogenous_injections = Self::generate_exogenous_injections(
            &mut rng, n, h, &network,
        );

        // Generate day-ahead prices
        let day_ahead_prices = Self::generate_da_prices(&mut rng, n, h);

        let frictions = Frictions::default();
        let market = MarketParams {
            volatility: params.sigma,
            jump_probability: params.rho_jump,
            tail_index: params.alpha,
            mean_bias: constants::MU_BIAS,
            spatial_correlation: constants::RHO_SPATIAL,
            congestion_premium: constants::GAMMA_PRICE,
            congestion_threshold: constants::TAU_CONG,
        };

        Ok(Level2Challenge {
            seed: *seed,
            difficulty: difficulty.clone(),
            network,
            batteries,
            frictions,
            market,
            exogenous_injections,
            day_ahead_prices,
        })
    }

    /// Build the initial simulation state at t=0.
    /// Uses the challenge seed with all-false congestion indicators (spec: 1^cong_{i,0} = 0).
    pub fn initial_state(&self) -> State {
        let socs: Vec<f64> = self.batteries
            .iter()
            .map(|b| b.spec.soc_initial_mwh)
            .collect();
        let congestion_indicators = vec![false; self.network.num_nodes];
        let rt_prices = self.generate_rt_prices(0, &self.seed, &congestion_indicators);
        State {
            time_step: 0,
            socs,
            rt_prices,
        }
    }

    /// Validate a portfolio action against all constraints and return new SOCs on success.
    ///
    /// Checks:
    /// 1. Action count matches battery count
    /// 2. Per-battery power bounds
    /// 3. Per-battery SOC feasibility after applying action
    /// 4. Network flow feasibility (PTDF line limits)
    ///
    /// Returns the vector of new SOCs (clamped to bounds) if all constraints pass.
    fn validate_action(
        &self,
        state: &State,
        action: &PortfolioAction,
    ) -> Result<Vec<f64>> {
        let m = self.batteries.len();

        if action.actions.len() != m {
            return Err(anyhow!(
                "Portfolio action has {} batteries, expected {}",
                action.actions.len(), m
            ));
        }

        let mut new_socs = Vec::with_capacity(m);
        for (b, placed) in self.batteries.iter().enumerate() {
            let u = action.actions[b].power_mw;
            let battery = &placed.spec;

            if u < -battery.power_charge_mw - 1e-6 {
                return Err(anyhow!(
                    "Battery {} charge exceeds limit at step {}: {} > {}",
                    b, state.time_step, -u, battery.power_charge_mw
                ));
            }
            if u > battery.power_discharge_mw + 1e-6 {
                return Err(anyhow!(
                    "Battery {} discharge exceeds limit at step {}: {} > {}",
                    b, state.time_step, u, battery.power_discharge_mw
                ));
            }

            let new_soc = self.apply_action_to_soc(b, state.socs[b], &action.actions[b]);
            if new_soc < battery.soc_min_mwh - constants::EPS_SOC {
                return Err(anyhow!(
                    "Battery {} SOC below minimum at step {}: {:.4} < {:.4}",
                    b, state.time_step, new_soc, battery.soc_min_mwh
                ));
            }
            if new_soc > battery.soc_max_mwh + constants::EPS_SOC {
                return Err(anyhow!(
                    "Battery {} SOC above maximum at step {}: {:.4} > {:.4}",
                    b, state.time_step, new_soc, battery.soc_max_mwh
                ));
            }

            new_socs.push(new_soc.clamp(battery.soc_min_mwh, battery.soc_max_mwh));
        }

        let injections = self.compute_total_injections(action, state.time_step);
        let flows = self.network.compute_flows(&injections);
        if let Some((line, flow)) = self.network.check_flow_limits(&flows) {
            return Err(anyhow!(
                "Line {} flow limit violated at step {}: |{:.2}| > {:.2}",
                line, state.time_step, flow, self.network.flow_limits[line]
            ));
        }

        Ok(new_socs)
    }

    /// Simulate one time step without commitment (for innovator use).
    ///
    /// Validates the action, applies it to the state, and returns the next state
    /// using the provided RT prices. No seed advancement or commitment chain update.
    ///
    /// `rt_prices_override` is required — innovators must provide their own
    /// price forecast/scenario for the next step.
    ///
    /// Returns `Err` if the action violates any constraint.
    pub fn take_step(
        &self,
        state: &State,
        action: &PortfolioAction,
        rt_prices_override: &[f64],
    ) -> Result<State> {
        let new_socs = self.validate_action(state, action)?;

        Ok(State {
            time_step: state.time_step + 1,
            socs: new_socs,
            rt_prices: rt_prices_override.to_vec(),
        })
    }

    /// Run the full rollout loop with commitment chain.
    ///
    /// Calls `algorithm` at each step to produce actions. If any action violates
    /// constraints, the error propagates immediately (terminates the rollout).
    ///
    /// The commitment chain (seed advancement, congestion indicators, RT price
    /// generation) is handled internally — innovators cannot access or bypass it.
    pub fn grid_optimize(
        &self,
        algorithm: &dyn Fn(&Level2Challenge, &State) -> Result<PortfolioAction>,
    ) -> Result<Level2Solution> {
        let params = self.difficulty.effective_params();
        let h = params.num_steps;

        let mut state = self.initial_state();
        let mut seed = self.seed;
        let mut congestion_indicators = vec![false; self.network.num_nodes];
        let mut schedule = Vec::with_capacity(h);

        for t in 0..h {
            let action = algorithm(self, &state)?;
            let new_socs = self.validate_action(&state, &action)?;

            let new_seed = self.commit_step(&seed, t, &action, &state.socs);

            let injections = self.compute_total_injections(&action, t);
            let flows = self.network.compute_flows(&injections);
            let new_congestion = self.network.compute_congestion_indicators(
                &flows, self.market.congestion_threshold,
            );

            let next_rt_prices = if t + 1 < h {
                self.generate_rt_prices(t + 1, &new_seed, &new_congestion)
            } else {
                vec![]
            };

            schedule.push(action);

            state = State {
                time_step: t + 1,
                socs: new_socs,
                rt_prices: next_rt_prices,
            };
            seed = new_seed;
            congestion_indicators = new_congestion;
        }

        let _ = congestion_indicators;
        Ok(Level2Solution { schedule })
    }

    /// Generate battery portfolio with heterogeneity mechanism
    fn generate_batteries(
        rng: &mut impl Rng,
        num_nodes: usize,
        num_batteries: usize,
        heterogeneity: f64,
    ) -> Vec<PlacedBattery> {
        let mut batteries = Vec::with_capacity(num_batteries);

        for b in 0..num_batteries {
            // Uniform random placement
            let node = rng.gen_range(0..num_nodes);

            // Heterogeneity mechanism: M_b = 3^{h(2r_b - 1)} where r_b ~ U(0,1)
            let r: f64 = rng.gen();
            let m_factor = 3.0_f64.powf(heterogeneity * (2.0 * r - 1.0));

            let capacity = constants::NOMINAL_CAPACITY * m_factor;
            let power = constants::NOMINAL_POWER * m_factor;

            let spec = BatterySpec::with_capacity(b, capacity, power);
            batteries.push(PlacedBattery { spec, node });
        }

        batteries
    }

    /// Generate exogenous nodal injections
    fn generate_exogenous_injections(
        rng: &mut impl Rng,
        num_nodes: usize,
        num_steps: usize,
        network: &Network,
    ) -> Vec<Vec<f64>> {
        // Generate as low-rank spatiotemporal process
        // Two time factors x two node loading patterns + noise
        let mut injections = vec![vec![0.0; num_steps]; num_nodes];

        // Time patterns (sinusoidal load curves)
        let time_pattern1: Vec<f64> = (0..num_steps)
            .map(|t| (2.0 * PI * t as f64 / 96.0).sin())
            .collect();
        let time_pattern2: Vec<f64> = (0..num_steps)
            .map(|t| (2.0 * PI * t as f64 / 48.0 + PI / 4.0).sin())
            .collect();

        // Node patterns (random loadings)
        let node_pattern1: Vec<f64> = (0..num_nodes)
            .map(|_| rng.gen::<f64>() - 0.5)
            .collect();
        let node_pattern2: Vec<f64> = (0..num_nodes)
            .map(|_| rng.gen::<f64>() - 0.5)
            .collect();

        // Combine with noise
        let base_load = 50.0; // MW
        let pattern_scale = 20.0;
        let noise_scale = 2.0;

        for i in 0..num_nodes {
            if i == network.slack_bus {
                continue; // Slack bus injection computed later
            }
            for t in 0..num_steps {
                let pattern = pattern_scale * (
                    node_pattern1[i] * time_pattern1[t] +
                    node_pattern2[i] * time_pattern2[t]
                );
                let noise = noise_scale * box_muller_normal(rng);
                injections[i][t] = base_load * (rng.gen::<f64>() - 0.5) + pattern + noise;
            }
        }

        // Balance at slack bus: p_s = -Σ_{i≠s} p_i
        for t in 0..num_steps {
            let mut sum = 0.0;
            for i in 0..num_nodes {
                if i != network.slack_bus {
                    sum += injections[i][t];
                }
            }
            injections[network.slack_bus][t] = -sum;
        }

        // Verify flows are within EFFECTIVE limits (after gamma_cong scaling)
        // Use a margin to leave room for battery actions
        let flow_margin = 0.7; // Use only 70% of limit for exogenous flows
        let mut scale: f64 = 1.0;
        for t in 0..num_steps {
            let inj: Vec<f64> = injections.iter().map(|v| v[t]).collect();
            let flows = network.compute_flows(&inj);
            for (l, &flow) in flows.iter().enumerate() {
                // Use effective flow_limits, not nominal, and leave margin
                let limit = network.flow_limits[l] * flow_margin;
                if flow.abs() > limit {
                    scale = scale.min(limit / flow.abs() * 0.95);
                }
            }
        }

        if scale < 1.0 {
            for i in 0..num_nodes {
                for t in 0..num_steps {
                    injections[i][t] *= scale;
                }
            }
            // Re-balance at slack
            for t in 0..num_steps {
                let mut sum = 0.0;
                for i in 0..num_nodes {
                    if i != network.slack_bus {
                        sum += injections[i][t];
                    }
                }
                injections[network.slack_bus][t] = -sum;
            }
        }

        injections
    }

    /// Generate day-ahead nodal prices
    fn generate_da_prices(
        rng: &mut impl Rng,
        num_nodes: usize,
        num_steps: usize,
    ) -> Vec<Vec<f64>> {
        // Base price curve via Gaussian Process (or simple sinusoidal for efficiency)
        let base_da = generate_da_prices(rng, num_steps);

        // Generate node offsets (correlated AR(1) residual)
        let mut prices = vec![vec![0.0; num_steps]; num_nodes];
        let ar_coef: f64 = 0.8;

        for node in 0..num_nodes {
            let offset: f64 = 5.0 * (rng.gen::<f64>() - 0.5);
            let mut residual: f64 = 0.0;

            for t in 0..num_steps {
                residual = ar_coef * residual + (1.0_f64 - ar_coef * ar_coef).sqrt() * 2.0 * box_muller_normal(rng);
                let price = base_da[t] + offset + residual;
                prices[node][t] = price.max(constants::LAMBDA_DA_MIN);
            }
        }

        prices
    }

    /// Compute storage injection at each node from portfolio action
    pub fn compute_storage_injections(&self, action: &PortfolioAction) -> Vec<f64> {
        let mut injections = vec![0.0; self.network.num_nodes];

        for (b, placed) in self.batteries.iter().enumerate() {
            // u > 0 = discharge = positive injection
            injections[placed.node] += action.actions[b].power_mw;
        }

        injections
    }

    /// Compute total nodal injections (exogenous + storage) with slack balancing
    pub fn compute_total_injections(&self, action: &PortfolioAction, time_step: usize) -> Vec<f64> {
        let mut injections = vec![0.0; self.network.num_nodes];

        // Add exogenous injections
        for i in 0..self.network.num_nodes {
            if i != self.network.slack_bus {
                injections[i] = self.exogenous_injections[i][time_step];
            }
        }

        // Add storage injections
        for (b, placed) in self.batteries.iter().enumerate() {
            injections[placed.node] += action.actions[b].power_mw;
        }

        // Slack bus balances the system
        let mut sum = 0.0;
        for i in 0..self.network.num_nodes {
            if i != self.network.slack_bus {
                sum += injections[i];
            }
        }
        injections[self.network.slack_bus] = -sum;

        injections
    }

    /// Generate real-time nodal prices using action-committed mechanism
    pub fn generate_rt_prices(
        &self,
        time_step: usize,
        seed: &[u8; 32],
        congestion_indicators: &[bool],
    ) -> Vec<f64> {
        let n = self.network.num_nodes;
        let mut prices = Vec::with_capacity(n);

        // Deterministic PRNG from seed using hash-in-counter-mode
        let mut counter = 0u64;

        // Draw common factor z_t
        let z_common = prng_normal(seed, counter);
        counter += 1;

        // Draw z'_t for congestion premium
        let z_prime = prng_normal(seed, counter);
        counter += 1;
        let zeta = z_prime.max(0.0);

        for i in 0..n {
            let da_price = self.day_ahead_prices[i][time_step];

            // Draw idiosyncratic shock ε_i,t
            let eps_i = prng_normal(seed, counter);
            counter += 1;

            // Spatially correlated shock
            let rho = self.market.spatial_correlation;
            let xi_i = rho.sqrt() * z_common + (1.0 - rho).sqrt() * eps_i;

            // Base price with shock
            let mu = self.market.mean_bias;
            let sigma = self.market.volatility;
            let mut price = da_price * (1.0 + mu + sigma * xi_i);

            // Congestion premium (uses lagged indicator)
            if congestion_indicators[i] {
                price += self.market.congestion_premium * zeta;
            }

            // Jump component
            let u_jump = prng_uniform(seed, counter);
            counter += 1;

            if u_jump < self.market.jump_probability {
                let u_pareto = prng_uniform(seed, counter).max(1e-10);
                counter += 1;
                // Pareto: X = (1-U)^(-1/α), support [1,∞)
                let pareto = (1.0 - u_pareto).powf(-1.0 / self.market.tail_index);
                let jump = da_price * pareto;
                price += jump;
            }

            prices.push(price.clamp(constants::LAMBDA_MIN, constants::LAMBDA_MAX));
        }

        prices
    }

    /// Compute seed commitment update per spec equation (3.4)
    /// s_{t+1} = H(s_t || t || (ũ_1,...,ũ_m) || (Ẽ_1,...,Ẽ_m))
    pub fn commit_step(
        &self,
        current_seed: &[u8; 32],
        time_step: usize,
        action: &PortfolioAction,
        socs: &[f64],
    ) -> [u8; 32] {
        let mut hasher = Sha256::new();

        // s_t (256 bits = 32 bytes)
        hasher.update(current_seed);

        // t (big-endian signed 64-bit)
        hasher.update((time_step as i64).to_be_bytes());

        // Quantized actions (ũ_1, ..., ũ_m)
        for sa in &action.actions {
            let quantized = sa.quantized();
            hasher.update(quantized.to_be_bytes());
        }

        // Quantized SOCs (Ẽ_1, ..., Ẽ_m)
        for &soc in socs {
            let quantized = (soc / constants::Q_E).round() as i64;
            hasher.update(quantized.to_be_bytes());
        }

        let result = hasher.finalize();
        let mut new_seed = [0u8; 32];
        new_seed.copy_from_slice(&result);
        new_seed
    }

    /// Apply action to SOC and return new SOC
    /// E_{t+1} = E_t + η^c * c * Δt - d * Δt / η^d
    pub fn apply_action_to_soc(&self, battery_idx: usize, soc: f64, action: &SignedAction) -> f64 {
        let battery = &self.batteries[battery_idx].spec;
        let (c, d) = action.decompose();
        let dt = constants::DELTA_T;

        let new_soc = soc
            + battery.efficiency_charge * c * dt
            - d * dt / battery.efficiency_discharge;

        new_soc
    }

    /// Compute per-step portfolio profit per spec equation (3.7)
    pub fn compute_step_profit(
        &self,
        action: &PortfolioAction,
        rt_prices: &[f64],
    ) -> f64 {
        let dt = constants::DELTA_T;
        let mut total_profit = 0.0;

        for (b, placed) in self.batteries.iter().enumerate() {
            let u = action.actions[b].power_mw;
            let price = rt_prices[placed.node];

            // Revenue: u * λ * Δt
            let revenue = u * price * dt;

            // Friction: φ_b(u) = κ_tx|u|Δt + κ_deg(|u|Δt/E̅_b)^β
            let abs_u = u.abs();
            let tx_cost = self.frictions.transaction_cost_per_mwh * abs_u * dt;
            let deg_base = (abs_u * dt) / placed.spec.capacity_mwh;
            let deg_cost = self.frictions.degradation_scale * deg_base.powf(self.frictions.degradation_exponent);

            total_profit += revenue - tx_cost - deg_cost;
        }

        total_profit
    }

    /// Verify a solution following spec Section 6 (constraint checks + profit)
    /// Verify a solution following spec Section 6 (constraint checks + profit).
    /// Replays the full commitment chain from s_0, validating every action
    /// and accumulating profit deterministically.
    fn compute_profit(&self, solution: &Solution) -> Result<f64> {
        let params = self.difficulty.effective_params();
        let h = params.num_steps;

        if solution.schedule.len() != h {
            return Err(anyhow!(
                "Schedule has {} steps, expected {}",
                solution.schedule.len(), h
            ));
        }

        let mut state = self.initial_state();
        let mut seed = self.seed;
        let mut congestion_indicators = vec![false; self.network.num_nodes];
        let mut total_profit = 0.0;

        for t in 0..h {
            let action = &solution.schedule[t];

            let new_socs = self.validate_action(&state, action)?;

            let profit = self.compute_step_profit(action, &state.rt_prices);
            total_profit += profit;

            let new_seed = self.commit_step(&seed, t, action, &state.socs);

            let injections = self.compute_total_injections(action, t);
            let flows = self.network.compute_flows(&injections);
            let new_congestion = self.network.compute_congestion_indicators(
                &flows, self.market.congestion_threshold,
            );

            let next_rt_prices = if t + 1 < h {
                self.generate_rt_prices(t + 1, &new_seed, &new_congestion)
            } else {
                vec![]
            };

            state = State {
                time_step: t + 1,
                socs: new_socs,
                rt_prices: next_rt_prices,
            };
            seed = new_seed;
            congestion_indicators = new_congestion;
        }

        let _ = congestion_indicators;
        Ok(total_profit)
    }

    fn compute_greedy_baseline_profit(&self) -> Result<f64> {
        let baseline_solution = baselines::level2_greedy::solve_challenge(self)?
            .ok_or_else(|| anyhow!("Baseline solver failed to produce a solution"))?;
        self.compute_profit(&baseline_solution)
    }

    /// Evaluate a solution and return better-than-baseline quality
    conditional_pub!(
        fn evaluate_solution(&self, solution: &Solution) -> Result<i32> {
            let total_profit = self.compute_profit(solution)?;
            let baseline_profit = self.compute_greedy_baseline_profit()?;

            // Better-than-baseline (same form as knapsack): total / baseline - 1.
            // If the baseline profit is effectively zero, fall back to absolute delta.
            let better = if baseline_profit.abs() < 1e-9 {
                total_profit - baseline_profit
            } else {
                (total_profit / baseline_profit) - 1.0
            };

            let quality = better.clamp(-10.0, 10.0) * QUALITY_PRECISION as f64;
            let quality = quality.round() as i32;
            Ok(quality)
        }
    );

    /// Compute nodal injections from portfolio action
    pub fn compute_injections(&self, portfolio_action: &PortfolioAction) -> Vec<f64> {
        self.compute_storage_injections(portfolio_action)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate day-ahead prices using periodic + smooth kernel
fn generate_da_prices(rng: &mut impl Rng, num_steps: usize) -> Vec<f64> {
    let kernel = GPKernel::new();
    let k = kernel.covariance_matrix(num_steps);
    let l = cholesky(&k);

    // Generate standard normal samples
    let z: Vec<f64> = (0..num_steps)
        .map(|_| box_muller_normal(rng))
        .collect();

    let mut prices = vec![0.0; num_steps];

    for i in 0..num_steps {
        for j in 0..num_steps {
            prices[i] += l[i][j] * z[j];
        }
        // Add diurnal pattern based on 15-min steps
        let hour = (i as f64) * constants::DELTA_T;
        prices[i] += constants::MEAN_DA_PRICE
            + constants::DA_AMPLITUDE * (2.0 * PI * hour / 24.0 - PI / 2.0).sin();
        prices[i] = prices[i].max(constants::LAMBDA_DA_MIN);
    }

    prices
}

/// Gaussian Process kernel for price generation
struct GPKernel {
    sigma_periodic: f64,
    length_periodic: f64,
    sigma_se: f64,
    length_se: f64,
    period_hours: f64,
}

impl GPKernel {
    fn new() -> Self {
        Self {
            sigma_periodic: 10.0,
            length_periodic: 2.0,
            sigma_se: 5.0,
            length_se: 4.0,
            period_hours: 24.0,
        }
    }

    fn evaluate(&self, t1: f64, t2: f64) -> f64 {
        // Convert step indices to hours
        let h1 = t1 * constants::DELTA_T;
        let h2 = t2 * constants::DELTA_T;
        let tau = (h1 - h2).abs();

        // Periodic component
        let periodic = self.sigma_periodic.powi(2)
            * (-2.0 * (PI * tau / self.period_hours).sin().powi(2)
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
                    k[i][j] += 1e-6; // Numerical stability
                }
            }
        }
        k
    }
}

/// Box-Muller transform for normal random variable
fn box_muller_normal(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-10);
    let u2: f64 = rng.gen();
    (-2.0_f64 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Deterministic PRNG: uniform from seed using hash-in-counter-mode
fn prng_uniform(seed: &[u8; 32], counter: u64) -> f64 {
    let mut hasher = Sha256::new();
    hasher.update(seed);
    hasher.update(counter.to_be_bytes());
    let hash = hasher.finalize();

    // Take first 8 bytes as u64
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&hash[0..8]);
    let val = u64::from_be_bytes(bytes);

    // Convert to [0,1)
    val as f64 / (u64::MAX as f64 + 1.0)
}

/// Deterministic PRNG: normal from seed using Box-Muller
fn prng_normal(seed: &[u8; 32], counter: u64) -> f64 {
    let u1 = prng_uniform(seed, counter * 2).max(1e-10);
    let u2 = prng_uniform(seed, counter * 2 + 1);
    (-2.0_f64 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
}

/// Cholesky decomposition (lower triangular)
fn cholesky(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    use nalgebra::DMatrix;

    let n = a.len();
    if n == 0 {
        return vec![];
    }

    // Build nalgebra matrix
    let matrix = DMatrix::from_fn(n, n, |i, j| a[i][j]);

    // Cholesky decomposition - returns lower triangular L where A = L * L^T
    let chol = matrix
        .cholesky()
        .expect("Covariance matrix should be positive definite");

    // Extract lower triangular factor
    let l_matrix = chol.l();

    // Convert back to Vec<Vec<f64>>
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            l[i][j] = l_matrix[(i, j)];
        }
    }
    l
}

/// Matrix inversion via Cholesky decomposition (for symmetric positive definite matrices)
fn invert_matrix(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    use nalgebra::DMatrix;

    let n = a.len();
    if n == 0 {
        return vec![];
    }

    // Build nalgebra matrix
    let matrix = DMatrix::from_fn(n, n, |i, j| a[i][j]);

    // Cholesky decomposition and inverse
    let inverse = matrix
        .cholesky()
        .expect("B_red should be positive definite for a connected network")
        .inverse();

    // Convert back to Vec<Vec<f64>>
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            result[i][j] = inverse[(i, j)];
        }
    }
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_parameters() {
        let params = Track::Track1.parameters();
        assert_eq!(params.num_nodes, 20);
        assert_eq!(params.num_lines, 30);
        assert_eq!(params.num_batteries, 10);
        assert_eq!(params.num_steps, 96);
        assert_eq!(params.gamma_cong, 1.0);

        let params = Track::Track5.parameters();
        assert_eq!(params.num_nodes, 150);
        assert_eq!(params.num_lines, 300);
        assert_eq!(params.num_batteries, 100);
        assert_eq!(params.num_steps, 192);
        assert_eq!(params.gamma_cong, 0.4);
    }

    #[test]
    fn test_challenge_generation_track1() {
        let difficulty = Level2Difficulty::from_track(Track::Track1);
        let seed = [42u8; 32];
        let challenge = Level2Challenge::generate_instance_with_difficulty(&seed, &difficulty).unwrap();

        assert_eq!(challenge.network.num_nodes, 20);
        assert_eq!(challenge.batteries.len(), 10);
        assert_eq!(challenge.day_ahead_prices.len(), 20); // n nodes
        assert_eq!(challenge.day_ahead_prices[0].len(), 96); // H steps
    }

    #[test]
    fn test_heterogeneity_mechanism() {
        let mut rng = SmallRng::seed_from_u64(42);
        let batteries = Level2Challenge::generate_batteries(&mut rng, 10, 20, 1.0);

        // With h=1, we should see ~3x spread
        let capacities: Vec<f64> = batteries.iter().map(|b| b.spec.capacity_mwh).collect();
        let min_cap = capacities.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_cap = capacities.iter().cloned().fold(0.0, f64::max);

        // Expected spread is [3^-1, 3^1] = [0.33, 3] relative to nominal
        assert!(max_cap / min_cap > 2.0, "Expected significant heterogeneity");
    }

    #[test]
    fn test_network_ptdf() {
        let network = Network::new(
            4,
            vec![(0, 1), (1, 2), (2, 3), (3, 0)], // Ring
            vec![10.0, 10.0, 10.0, 10.0],
            vec![50.0, 50.0, 50.0, 50.0],
            1.0,
        );

        // Inject 1 MW at node 1, slack at node 0
        let injections = vec![-1.0, 1.0, 0.0, 0.0]; // Balanced
        let flows = network.compute_flows(&injections);

        // Flows should be distributed around the ring
        let total_flow: f64 = flows.iter().map(|f| f.abs()).sum();
        assert!(total_flow > 0.0, "Expected non-zero flows");
        assert!(flows.iter().all(|&f| f.abs() <= 1.0 + 1e-6), "Flows should be bounded");
    }

    #[test]
    fn test_signed_action() {
        let discharge = SignedAction::new(10.0);
        let (c, d) = discharge.decompose();
        assert!((c - 0.0).abs() < 1e-6);
        assert!((d - 10.0).abs() < 1e-6);

        let charge = SignedAction::new(-10.0);
        let (c, d) = charge.decompose();
        assert!((c - 10.0).abs() < 1e-6);
        assert!((d - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_canonical_seed_update() {
        let difficulty = Level2Difficulty::from_track(Track::Track1);
        let seed = [42u8; 32];
        let challenge = Level2Challenge::generate_instance_with_difficulty(&seed, &difficulty).unwrap();

        let action = PortfolioAction {
            actions: vec![SignedAction::new(5.0); challenge.batteries.len()],
        };
        let socs: Vec<f64> = challenge.batteries
            .iter()
            .map(|b| b.spec.soc_initial_mwh)
            .collect();

        // Same inputs should produce same seed
        let seed1 = challenge.commit_step(&seed, 0, &action, &socs);
        let seed2 = challenge.commit_step(&seed, 0, &action, &socs);
        assert_eq!(seed1, seed2);

        // Different action should produce different seed
        let action2 = PortfolioAction {
            actions: vec![SignedAction::new(-5.0); challenge.batteries.len()],
        };
        let seed3 = challenge.commit_step(&seed, 0, &action2, &socs);
        assert_ne!(seed1, seed3);
    }

    #[test]
    fn test_congestion_indicators() {
        let network = Network::new(
            4,
            vec![(0, 1), (1, 2), (2, 3), (3, 0)],
            vec![10.0; 4],
            vec![50.0; 4],
            1.0,
        );

        // Create flows where line 0 is congested
        let flows = vec![49.0, 10.0, 10.0, 10.0]; // Line 0 at 98% capacity
        let indicators = network.compute_congestion_indicators(&flows, 0.97);

        // Nodes 0 and 1 should be marked congested (incident to line 0)
        assert!(indicators[0]);
        assert!(indicators[1]);
        assert!(!indicators[2]);
        assert!(!indicators[3]);
    }

    #[test]
    fn test_prng_determinism() {
        let seed = [123u8; 32];

        let u1 = prng_uniform(&seed, 0);
        let u2 = prng_uniform(&seed, 0);
        assert_eq!(u1, u2, "PRNG should be deterministic");

        let u3 = prng_uniform(&seed, 1);
        assert_ne!(u1, u3, "Different counters should give different values");

        let n1 = prng_normal(&seed, 0);
        let n2 = prng_normal(&seed, 0);
        assert_eq!(n1, n2, "Normal PRNG should be deterministic");
    }

    #[test]
    fn test_initial_state() {
        let difficulty = Level2Difficulty::from_track(Track::Track1);
        let seed = [42u8; 32];
        let challenge = Level2Challenge::generate_instance_with_difficulty(&seed, &difficulty).unwrap();
        let state = challenge.initial_state();

        assert_eq!(state.time_step, 0);
        assert_eq!(state.socs.len(), challenge.batteries.len());
        assert_eq!(state.rt_prices.len(), challenge.network.num_nodes);
        for (b, placed) in challenge.batteries.iter().enumerate() {
            assert!((state.socs[b] - placed.spec.soc_initial_mwh).abs() < 1e-9);
        }
    }

    #[test]
    fn test_take_step_idle() {
        let difficulty = Level2Difficulty::from_track(Track::Track1);
        let seed = [42u8; 32];
        let challenge = Level2Challenge::generate_instance_with_difficulty(&seed, &difficulty).unwrap();
        let state = challenge.initial_state();

        let idle_action = PortfolioAction {
            actions: vec![SignedAction::idle(); challenge.batteries.len()],
        };
        let fake_prices = vec![50.0; challenge.network.num_nodes];

        let next_state = challenge.take_step(&state, &idle_action, &fake_prices).unwrap();
        assert_eq!(next_state.time_step, 1);
        for b in 0..challenge.batteries.len() {
            assert!((next_state.socs[b] - state.socs[b]).abs() < 1e-9);
        }
        assert_eq!(next_state.rt_prices, fake_prices);
    }

    #[test]
    fn test_take_step_constraint_violation() {
        let difficulty = Level2Difficulty::from_track(Track::Track1);
        let seed = [42u8; 32];
        let challenge = Level2Challenge::generate_instance_with_difficulty(&seed, &difficulty).unwrap();
        let state = challenge.initial_state();

        let mut actions = vec![SignedAction::idle(); challenge.batteries.len()];
        actions[0] = SignedAction::new(99999.0);
        let bad_action = PortfolioAction { actions };
        let fake_prices = vec![50.0; challenge.network.num_nodes];

        assert!(challenge.take_step(&state, &bad_action, &fake_prices).is_err());
    }

    #[test]
    fn test_grid_optimize_idle() {
        let difficulty = Level2Difficulty::from_track(Track::Track1);
        let seed = [42u8; 32];
        let challenge = Level2Challenge::generate_instance_with_difficulty(&seed, &difficulty).unwrap();

        let solution = challenge.grid_optimize(&|challenge, _state| {
            Ok(PortfolioAction {
                actions: vec![SignedAction::idle(); challenge.batteries.len()],
            })
        }).unwrap();

        let params = difficulty.effective_params();
        assert_eq!(solution.schedule.len(), params.num_steps);
    }

    #[test]
    fn test_grid_optimize_matches_compute_profit() {
        let difficulty = Level2Difficulty::from_track(Track::Track1);
        let seed = [42u8; 32];
        let challenge = Level2Challenge::generate_instance_with_difficulty(&seed, &difficulty).unwrap();

        let solution = challenge.grid_optimize(&|challenge, _state| {
            Ok(PortfolioAction {
                actions: vec![SignedAction::idle(); challenge.batteries.len()],
            })
        }).unwrap();

        // compute_profit should accept this solution without error
        let profit = challenge.compute_profit(&solution).unwrap();
        assert!(profit.abs() < 1e-6, "Idle schedule profit should be ~0, got {}", profit);
    }
}
