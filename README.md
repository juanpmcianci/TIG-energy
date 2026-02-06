# TIG Energy Arbitrage Network Challenge

A Rust implementation of the Energy Storage Arbitrage challenge for [The Innovation Game (TIG)](https://tig.foundation/).

## Overview

This challenge tasks solvers with optimizing battery storage operations across a transmission-constrained network. An operator manages multiple batteries distributed across nodes, each with locational marginal prices (LMPs) that differ due to congestion. The goal is to maximize portfolio profit while respecting network flow limits and handling price uncertainty.

## Key Features

- **Network constraints**: PTDF-based DC power flow, line thermal limits, locational marginal prices
- **Stochastic pricing**: Real-time prices deviate from day-ahead forecasts with volatility and tail spikes
- **Action commitment**: SHA-256 hash-based mechanism ensures solvers cannot exploit future price information
- **Physical constraints**: SOC limits, power limits, charging/discharging efficiency losses
- **Market frictions**: Transaction costs and battery degradation
- **Track system**: 5 predefined difficulty regimes from correctness baseline to capstone
- **Efficient verification**: O(H × L × n) complexity

## Track System

The challenge features **5 predefined tracks** with progressively harder parameter regimes:

| Track | Nodes | Lines | Batteries | Steps | γ_cong | σ | α | h |
|-------|-------|-------|-----------|-------|--------|-----|-----|-----|
| **Track 1** | 20 | 30 | 10 | 96 | 1.00 | 0.10 | 4.0 | 0.2 |
| **Track 2** | 40 | 60 | 20 | 96 | 0.80 | 0.15 | 3.5 | 0.4 |
| **Track 3** | 80 | 120 | 40 | 192 | 0.60 | 0.20 | 3.0 | 0.6 |
| **Track 4** | 100 | 200 | 60 | 192 | 0.50 | 0.25 | 2.7 | 0.8 |
| **Track 5** | 150 | 300 | 100 | 192 | 0.40 | 0.30 | 2.5 | 1.0 |

**Parameter definitions:**
- **γ_cong**: Line limit scaling factor (lower = tighter limits, more congestion)
- **σ**: RT price volatility
- **α**: Pareto tail index (lower = heavier tails, more extreme spikes)
- **h**: Fleet heterogeneity (battery capacity/power spread factor)

### Track Descriptions

1. **Track 1 - Correctness Baseline**: Small network with nominal limits. Tests basic feasibility.
2. **Track 2 - Meaningful Congestion**: Tighter limits create real spatial arbitrage opportunities.
3. **Track 3 - Multi-Day Horizon**: 48-hour horizon with larger network and more frequent spikes.
4. **Track 4 - Dense Network**: High line density, frequent congestion, heavy tails require risk management.
5. **Track 5 - Capstone**: Largest scale, tightest limits, heaviest tails. Full difficulty.

## Project Structure

```
TIG-energy/
├── Cargo.toml                              # Workspace root
├── README.md
├── tig_level_2_spec.tex                    # Full specification document
├── tig-challenges/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── energy_arbitrage_v2.rs          # Challenge implementation
└── tig-algorithms/
    ├── Cargo.toml
    ├── src/
    │   ├── lib.rs
    │   └── energy_arbitrage_v2/
    │       ├── mod.rs
    │       ├── level2_greedy.rs            # Flow-aware greedy solver
    │       └── level2_decomposition.rs     # Benders decomposition solver
    └── examples/
        └── runner_v2.rs                    # Example runner (all tracks)
```

## Quick Start

```bash
# Build the project
cargo build --release

# Run tests
cargo test

# Run the example runner (all tracks)
cargo run --example runner_v2 --release
```

---

## Mathematical Formulation

### Signed Action Model

Battery actions use signed power convention:
- $u_b > 0$: discharge (net injection to grid)
- $u_b < 0$: charge (net withdrawal from grid)

Bounded by: $-\bar{P}^c_b \leq u_{b,t} \leq \bar{P}^d_b$

### SOC Dynamics

$$E_{b,t+1} = E_{b,t} + \eta^c_b \cdot c_{b,t} \cdot \Delta t - \frac{d_{b,t} \cdot \Delta t}{\eta^d_b}$$

Where $(c, d)$ decompose from signed action $u$.

### Network Model (DC Power Flow)

Nodal injection at node $i$:
$$p_{i,t} = p^{exog}_{i,t} + \sum_{b: \nu(b)=i} u_{b,t}$$

Line flows via PTDF:
$$f_{\ell,t} = \sum_{k \in \mathcal{N}} \text{PTDF}_{\ell,k} \cdot p_{k,t}$$

Slack bus balances system: $p_{s,t} = -\sum_{i \neq s} p_{i,t}$

Flow constraints:
$$|f_{\ell,t}| \leq \bar{F}_\ell \cdot \gamma_{cong} \quad \forall \ell \in \mathcal{L}$$

### Congestion-Aware Pricing

Real-time prices include congestion premiums:

$$\lambda^{RT}_{i,t} = \lambda^{DA}_{i,t} \cdot (1 + \mu + \sigma \cdot \xi_{i,t}) + \gamma_{price} \cdot \mathbf{1}^{cong}_{i,t-1} \cdot \zeta_t + J_{i,t}$$

Where:
- $\xi_{i,t}$: Spatially correlated shock ($\rho_{sp} = 0.70$)
- $\mathbf{1}^{cong}_{i,t-1}$: Lagged congestion indicator (1 if incident line at 97%+ capacity)
- $\zeta_t$: Common congestion premium factor
- $J_{i,t}$: Pareto jump component

### Battery Heterogeneity

Capacity and power scaled by multiplicative factor:
$$M_b = 3^{h(2r_b - 1)}, \quad r_b \sim U(0,1)$$

At $h = 1.0$ (Track 5), fleet spans ~9x range in capacity.

### Seed Commitment

Canonical encoding with big-endian signed 64-bit integers:

$$s_{t+1} = \text{SHA256}(s_t \| t \| \tilde{u}_1 \| \cdots \| \tilde{u}_m \| \tilde{E}_1 \| \cdots \| \tilde{E}_m)$$

Where $\tilde{u} = \lfloor u / q_u \rceil$ with $q_u = 0.01$ MW.

### Objective

Maximize portfolio profit:

$$\max \sum_{t=0}^{H-1} \sum_{b \in \mathcal{B}} \left[ u_{b,t} \cdot \lambda^{RT}_{\nu(b),t} \cdot \Delta t - \phi_b(u_{b,t}) \right]$$

## Default Constants

| Constant | Value | Description |
|----------|-------|-------------|
| $\Delta t$ | 0.25 h | Time step (15 minutes) |
| $\eta^c, \eta^d$ | 0.95 | Charge/discharge efficiency |
| $\kappa_{tx}$ | $0.25/MWh | Transaction cost |
| $\kappa_{deg}$ | $1.00 | Degradation scale |
| $\beta$ | 2.0 | Degradation exponent |
| $\rho_{sp}$ | 0.70 | Spatial correlation |
| $\gamma_{price}$ | $20/MWh | Congestion premium |
| $\tau_{cong}$ | 0.97 | Congestion threshold |
| $\bar{E}$ | 100 MWh | Nominal battery capacity |
| $\bar{P}$ | 25 MW | Nominal battery power |

## API Usage

```rust
use tig_challenges::{
    Level2Challenge, Level2Difficulty, Level2Solution,
    Track, PortfolioAction, SignedAction, constants,
};

// Generate instance from track
let difficulty = Level2Difficulty::from_track(Track::Track1);
let seed = [42u8; 32];
let challenge = Level2Challenge::generate_instance(seed, &difficulty)?;

// Get effective parameters
let params = difficulty.effective_params();
println!("Network: {} nodes, {} lines", params.num_nodes, params.num_lines);
println!("Batteries: {}, Steps: {}", params.num_batteries, params.num_steps);

// Build schedule
let mut schedule = Vec::new();
let mut socs: Vec<f64> = challenge.batteries.iter()
    .map(|b| b.spec.soc_initial_mwh)
    .collect();

for step in 0..params.num_steps {
    let mut actions = Vec::new();

    for (b, placed) in challenge.batteries.iter().enumerate() {
        let node = placed.node;
        let da_price = challenge.day_ahead_prices[node][step];

        // Your policy: positive = discharge, negative = charge
        let power = decide_power(&challenge, b, socs[b], da_price);
        actions.push(SignedAction::new(power));
    }

    let portfolio = PortfolioAction { actions };

    // Check flow constraints
    let injections = challenge.compute_total_injections(&portfolio, step);
    let flows = challenge.network.compute_flows(&injections);

    if let Some((line, flow)) = challenge.network.check_flow_limits(&flows) {
        // Scale down or adjust actions to satisfy limits
    }

    // Update SOCs
    for (b, placed) in challenge.batteries.iter().enumerate() {
        let new_soc = challenge.apply_action_to_soc(b, socs[b], &portfolio.actions[b]);
        socs[b] = new_soc.clamp(placed.spec.soc_min_mwh, placed.spec.soc_max_mwh);
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

### Custom Difficulty (Override Track Defaults)

```rust
let difficulty = Level2Difficulty {
    track: Track::Track2,
    num_steps: Some(48),           // Override: shorter horizon
    congestion_factor: Some(0.5),  // Override: tighter limits
    profit_threshold: 1000.0,      // Require minimum profit
    ..Default::default()
};
```

---

## Solvers

### Flow-Aware Greedy (`level2_greedy.rs`)

Per-battery greedy decisions with iterative flow adjustment:
- Each battery uses price-based heuristics
- PTDF-aware action reduction when flows violated
- Binary search fallback for guaranteed feasibility

### Benders Decomposition (`level2_decomposition.rs`)

Iterative improvement approach:
- Conservative initial schedule
- Per-step local optimization with flow checking
- Binary search for feasible action scaling
- Final constraint enforcement pass

Better coordination across batteries.

---

## Verification Protocol

Verification is O(H × L × n):

1. Check initial seed matches instance
2. For each step t:
   - Check action bounds for all batteries
   - Update SOCs and verify feasibility
   - Compute nodal injections (exogenous + storage)
   - Balance at slack bus
   - Compute flows via PTDF multiplication
   - Verify no line limit violations
   - Compute period profit
   - Update congestion indicators for next step
   - Generate RT prices using committed seed
3. Sum total profit and check threshold

Hash commitment ensures any modification to actions invalidates subsequent prices.

---

## Performance Expectations

| Track | Solver | Typical Profit | Time (demo) |
|-------|--------|----------------|-------------|
| Track 1 | Greedy | $5,000-15,000 | <1ms |
| Track 1 | Decomposition | $10,000-15,000 | 10-20ms |
| Track 3 | Greedy | $1,000-5,000 | 2-5ms |
| Track 3 | Decomposition | $20,000-30,000 | 400-600ms |
| Track 5 | Greedy | $5,000-15,000 | 20-50ms |
| Track 5 | Decomposition | $15,000-25,000 | 1.5-2.5s |

*Results vary with random seed. Greedy may produce negative profit on difficult tracks.*

---

## Day-Ahead Price Generation

Prices sampled from Gaussian Process with periodic + smooth kernel:

$$k(t, t') = \sigma_p^2 \exp\left(-\frac{2\sin^2(\pi|t-t'|/T)}{\ell_p^2}\right) + \sigma_{SE}^2 \exp\left(-\frac{(t-t')^2}{2\ell_{SE}^2}\right)$$

- First term: 24-hour periodicity (diurnal pattern)
- Second term: smooth local deviations

Mean function captures typical load shape:

$$\mu(t) = \bar{\lambda} + A \cdot \sin(2\pi t / 24 - \pi/2)$$

Per-node variations added via AR(1) residuals.

---

## License

Licensed under the TIG Inbound Game License v2.0. See [LICENSE](LICENSE) for details.

## Contributing

To submit an algorithm to TIG:

1. Fork this repository
2. Create your algorithm in `tig-algorithms/src/energy_arbitrage_v2/`
3. Ensure it passes all tests: `cargo test`
4. Submit via the TIG protocol

See [TIG Documentation](https://docs.tig.foundation/) for submission guidelines.

## References

- [Specification](tig_level_2_spec.tex) - Full mathematical specification
- DC Power Flow: Wood & Wollenberg, "Power Generation, Operation and Control"
- SDDP: Pereira & Pinto, "Multi-stage stochastic optimization applied to energy planning"
