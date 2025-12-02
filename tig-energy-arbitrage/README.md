# TIG Energy Arbitrage Challenge

A Rust implementation of the Energy Storage Arbitrage challenge for [The Innovation Game (TIG)](https://tig.foundation/).

## Overview

This challenge tasks solvers with optimizing battery storage operations in stochastic electricity markets. The goal is to maximize profit by buying power when prices are low and selling when prices spike, while respecting physical constraints and handling price uncertainty.

The challenge is structured in **two levels**:

| Level | Description | Key Feature |
|-------|-------------|-------------|
| **Level 1** | Single-asset temporal arbitrage | Action-committed pricing (prevents lookahead cheating) |
| **Level 2** | Portfolio arbitrage on constrained network | Spatial price differences + network flow constraints |

## Key Features

- **Stochastic pricing**: Real-time prices deviate from day-ahead forecasts with volatility and tail spikes
- **Action commitment**: Hash-based mechanism ensures solvers cannot exploit future price information
- **Physical constraints**: SOC limits, power limits, charging/discharging efficiency losses
- **Market frictions**: Transaction costs and battery degradation
- **Network constraints** (Level 2): PTDF-based power flow, line thermal limits, locational marginal prices
- **Efficient verification**: O(H) for Level 1, O(H * |L| * n) for Level 2

## Project Structure

```
tig-energy-arbitrage/
├── Cargo.toml                              # Workspace root
├── README.md
├── tig-challenges/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── energy_arbitrage.rs             # Original challenge (backward compat)
│       └── energy_arbitrage_v2.rs          # Two-level challenge
└── tig-algorithms/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs
    │   ├── energy_arbitrage/               # Original algorithms
    │   │   ├── mod.rs
    │   │   ├── template.rs
    │   │   ├── greedy.rs
    │   │   ├── mpc_dp.rs
    │   │   └── sddp.rs
    │   └── energy_arbitrage_v2/            # V2 algorithms
    │       ├── mod.rs
    │       ├── level1_greedy.rs
    │       ├── level1_dp.rs
    │       ├── level2_greedy.rs
    │       └── level2_decomposition.rs
    └── examples/
        ├── runner.rs                       # Original runner
        └── runner_v2.rs                    # V2 runner (both levels)
```

## Quick Start

```bash
# Build the project
cargo build --release

# Run tests
cargo test

# Run the V2 example runner (both levels)
cargo run --example runner_v2 --release
```

---

# Level 1: Single-Asset Temporal Arbitrage

## Problem Statement

A battery operator observes a day-ahead price forecast and must commit to charge/discharge decisions as real-time prices unfold. The key innovation is **action-committed pricing**: the real-time price at each step depends on the operator's previous action via a cryptographic hash.

This prevents lookahead cheating - solvers cannot pre-compute optimal responses to all possible future prices.

## Mathematical Formulation

### State Dynamics

The battery state of charge evolves as:

$$E_{t+1} = E_t + \eta^c \cdot c_t \cdot \Delta t - \frac{d_t \cdot \Delta t}{\eta^d}$$

Subject to:
- $E_{\min} \leq E_t \leq E_{\max}$ (SOC bounds)
- $c_t \leq \bar{P}^c$, $d_t \leq \bar{P}^d$ (power limits)
- $c_t \cdot d_t = 0$ (no simultaneous charge/discharge)

### Action-Committed Price Generation

The seed sequence evolves via cryptographic hash:

$$s_{t+1} = \mathcal{H}(s_t \| a_t \| t \| E_t)$$

Real-time price:

$$\lambda^{RT}_{t+1} = \lambda^{DA}_{t+1} \cdot (1 + \mu_{t+1} + \sigma \cdot \xi_{t+1}) + J_{t+1}$$

Where:
- $\xi_{t+1} \sim \mathcal{N}(0,1)$ drawn from PRNG seeded by $s_{t+1}$
- $J_{t+1}$ is a Pareto-distributed jump component

### Profit Function

Per-step profit:

$$r_t = (d_t - c_t) \cdot \lambda^{RT}_t \cdot \Delta t - \phi(c_t, d_t)$$

Where friction cost:

$$\phi(c, d) = \kappa_{tx}(c + d)\Delta t + \kappa_{deg} \cdot \left(\frac{d \cdot \Delta t}{E}\right)^\beta$$

### Objective

Maximize expected total profit:

$$\max_{\pi} \mathbb{E}\left[\sum_{t=0}^{H-1} r_t \mid E_0, \lambda^{DA}, s_0\right]$$

## Difficulty Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `volatility` | [0.1, 0.5] | Price variance around forecast |
| `tail_index` | (2, 5] | Lower = heavier tails, more spikes |
| `transaction_cost` | $/MWh | Trading friction |
| `degradation_cost` | $/MWh | Battery wear cost |

## API Usage

```rust
use tig_challenges::{Level1Challenge, Level1Difficulty, Level1Solution};
use tig_challenges::{Action, TranscriptEntry};

// Generate instance
let difficulty = Level1Difficulty {
    num_steps: 24,
    volatility: 0.2,
    tail_index: 3.0,
    transaction_cost: 0.5,
    degradation_cost: 1.0,
    profit_threshold: 0.0,
};

let seed = [0u8; 32];
let challenge = Level1Challenge::generate_instance(seed, &difficulty)?;

// Solve (build transcript with committed actions)
let mut transcript = Vec::new();
let mut current_seed = challenge.seed;
let mut soc = challenge.battery.soc_initial_mwh;

for step in 0..challenge.difficulty.num_steps {
    let da_price = challenge.day_ahead_prices[step];
    let rt_price = challenge.generate_rt_price(da_price, &current_seed);

    // Your policy decides action based on current info
    let action = decide_action(&challenge, soc, rt_price, da_price);

    let new_soc = challenge.apply_action(soc, &action, 1.0);
    let profit = challenge.compute_step_profit(&action, rt_price, 1.0);

    transcript.push(TranscriptEntry {
        time_step: step,
        action: action.clone(),
        soc_mwh: new_soc,
        seed: current_seed,
        rt_price,
        profit,
    });

    current_seed = challenge.commit_action(&current_seed, &action, step, new_soc);
    soc = new_soc;
}

let solution = Level1Solution { transcript };

// Verify
match challenge.verify_solution(&solution) {
    Ok(profit) => println!("Valid! Profit: ${:.2}", profit),
    Err(e) => println!("Invalid: {}", e),
}
```

---

# Level 2: Portfolio Arbitrage on Constrained Network

## Problem Statement

An operator manages multiple batteries distributed across nodes of a transmission network. Each node has a locational marginal price (LMP) that can differ due to congestion. The operator must coordinate all batteries to maximize portfolio profit while respecting network flow limits.

## Mathematical Formulation

### Network Model

Power flows under DC approximation:

$$f_\ell = \sum_{k \in \mathcal{N}} \text{PTDF}_{\ell k} \cdot p_k$$

Where PTDF (Power Transfer Distribution Factors) matrix is computed from network topology.

### Constraints

Battery dynamics apply independently to each battery $b$:

$$E_{b,t+1} = E_{b,t} + \eta^c_b \cdot c_{b,t} - \frac{d_{b,t}}{\eta^d_b}$$

Network flow constraints:

$$|f_{\ell,t}| \leq \bar{F}_\ell \quad \forall \ell \in \mathcal{L}$$

### Nodal Price Model

LMPs include congestion premiums:

$$\lambda^{RT}_{i,t} = \lambda^{DA}_{i,t} + \sigma_i \xi_{i,t} + \gamma \cdot \mathbf{1}\{\text{line incident to } i \text{ congested}\} \cdot \zeta_t$$

### Objective

Maximize portfolio profit:

$$\max \mathbb{E}\left[\sum_t \sum_{b \in \mathcal{B}} (d_{b,t} - c_{b,t}) \cdot \lambda^{RT}_{\nu(b),t} \cdot \Delta t - \phi_b(c_{b,t}, d_{b,t})\right]$$

Subject to flow constraints.

## Difficulty Parameters

| Parameter | Range | Effect |
|-----------|-------|--------|
| `num_nodes` | 3+ | Network size |
| `num_batteries` | 1+ | Portfolio size |
| `congestion_factor` | (0, 1] | Scales line limits (lower = more congestion) |
| `heterogeneity` | [0, 1] | Battery parameter diversity |
| `congestion_premium` | $/MWh | LMP divergence during congestion |

## API Usage

```rust
use tig_challenges::{Level2Challenge, Level2Difficulty, Level2Solution};
use tig_challenges::{Action, PortfolioAction};

// Generate instance
let difficulty = Level2Difficulty {
    num_steps: 24,
    num_nodes: 5,
    num_batteries: 3,
    volatility: 0.2,
    tail_index: 3.0,
    congestion_factor: 0.8,
    heterogeneity: 0.3,
    congestion_premium: 10.0,
    profit_threshold: 0.0,
};

let seed = [0u8; 32];
let challenge = Level2Challenge::generate_instance(seed, &difficulty)?;

// Build schedule
let mut schedule = Vec::new();
for step in 0..challenge.difficulty.num_steps {
    let mut battery_actions = Vec::new();

    for (b, placed) in challenge.batteries.iter().enumerate() {
        // Your policy for battery b at node placed.node
        let action = decide_battery_action(&challenge, b, step);
        battery_actions.push(action);
    }

    // Check flow constraints
    let portfolio = PortfolioAction { battery_actions };
    let injections = challenge.compute_injections(&portfolio);
    let flows = challenge.network.compute_flows(&injections);

    if challenge.network.check_flow_limits(&flows).is_some() {
        // Adjust actions to respect limits
    }

    schedule.push(portfolio);
}

let solution = Level2Solution { schedule };

// Verify
match challenge.verify_solution(&solution) {
    Ok(profit) => println!("Valid! Portfolio profit: ${:.2}", profit),
    Err(e) => println!("Invalid: {}", e),
}
```

---

# Solvers

## Level 1 Solvers

### Greedy Threshold (`level1_greedy.rs`)

Simple rule-based policy:
- Charge when RT price < 0.95 x DA price
- Discharge when RT price > 1.05 x DA price
- Otherwise hold

Fast baseline implementation.

### DP with MPC (`level1_dp.rs`)

Model Predictive Control with lookahead:
- Rolling horizon optimization
- Monte Carlo sampling for expected value
- Discretized SOC and action grids

Better quality with moderate computational cost.

## Level 2 Solvers

### Greedy with Flow Adjustment (`level2_greedy.rs`)

Per-battery greedy decisions with portfolio-level flow feasibility adjustment:
- Each battery uses price-based heuristics
- Actions scaled down proportionally if flow limits violated

### Benders Decomposition (`level2_decomposition.rs`)

Iterative improvement approach:
- Master: battery-level optimization
- Subproblem: flow feasibility check
- Cuts added when constraints violated

Better coordination across batteries.

---

# Verification Protocol

## Level 1

Verification is O(H):

1. Check initial seed matches instance
2. For each step t:
   - Recompute $s_{t+1}$ via hash commitment
   - Regenerate $\lambda^{RT}_{t+1}$
   - Verify constraints (1)-(3)
   - Recompute profit
3. Sum total profit and check threshold

Hash commitment ensures any modification to actions invalidates subsequent prices.

## Level 2

Additional O(|L| * n) per step:

1. All Level 1 checks per battery
2. Compute nodal injections from actions
3. Check flow constraints via PTDF multiplication
4. Verify no line violations

---

# Day-Ahead Price Generation

Prices sampled from Gaussian Process with periodic kernel:

$$k(t, t') = \sigma_p^2 \exp\left(-\frac{2\sin^2(\pi|t-t'|/24)}{\ell_p^2}\right) + \sigma_{SE}^2 \exp\left(-\frac{(t-t')^2}{2\ell_{SE}^2}\right)$$

- First term: 24-hour periodicity (diurnal pattern)
- Second term: smooth deviations

Mean function captures typical load shape:

$$\mu(t) = \bar{\lambda}(1 + 0.3\sin(2\pi t/24))$$

---

# Performance Expectations

| Level | Solver | Typical Improvement | Time (24 steps) |
|-------|--------|---------------------|-----------------|
| L1 | Greedy | Baseline | <1ms |
| L1 | DP-MPC | +20-40% | 5-20ms |
| L2 | Greedy | Baseline | <5ms |
| L2 | Decomposition | +15-30% | 50-200ms |

---

# License

Licensed under the TIG Inbound Game License v2.0. See [LICENSE](LICENSE) for details.

# Contributing

To submit an algorithm to TIG:

1. Fork this repository
2. Create your algorithm in `tig-algorithms/src/energy_arbitrage_v2/`
3. Ensure it passes all tests
4. Submit via the TIG protocol

See [TIG Documentation](https://docs.tig.foundation/) for submission guidelines.

# References

- [Challenge Specification PDF](docs/tig_challenge_improved.pdf)
- DC Power Flow: Wood & Wollenberg, "Power Generation, Operation and Control"
- SDDP: Pereira & Pinto, "Multi-stage stochastic optimization applied to energy planning"
