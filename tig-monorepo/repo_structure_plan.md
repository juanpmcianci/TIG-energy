`tig-challenges` and `tig-algorithms` are modules in my codebase. For the grid energy challenge, I’m considering the following structure.

## `tig-challenges`

### 1) `generate_instance()`
Generates a single challenge instance, e.g. the grid/graph topology, parameters, and the day-ahead (DA) price process.

### 2) `grid_optimize(instance, innovators_algorithm, ...)`
Runs the simulation/optimization loop for \(t = 1, \dots, T\). It takes:
- the **challenge instance** (from `generate_instance`)
- the **innovator’s policy/decision function** `innovators_algorithm` (defined in `tig-algorithms` by the innovator)

Workflow:
1. Initialize the state \(S_0\).
2. For each time step \(t\):
   - call `innovators_algorithm(S_t)` to obtain an action \(A_t\)
   - update the state via `S_{t+1} = take_step(S_t, A_t)`
   - append \(A_t\) to the running solution (so the solution is a list of actions)

So `grid_optimize` provides the standard “rollout” framework: state → action → transition → repeat.

### 3) `take_step(state, action, rt_prices_override=None, ...)`
Implements the state transition. Given the current state and action, it returns the next state. Part of this transition is producing the **real-time (RT) node prices** for the next time step.

By default, RT prices are generated inside `take_step` in an unpredictable way, and can depend on (adapt to) the action \(A_t\).

Important extra feature:
- `take_step` can also be called directly from `tig-algorithms`.
- It optionally accepts an override `rt_prices_override` (RT prices at nodes). If provided, this bypasses the internal random/adaptive RT price generation.
- This gives innovators a way to simulate "what-if" charging/discharging scenarios under hypothetical RT price paths.

### 4) `verify_solution(instance, solution) -> Result<profit>`
Takes a challenge instance and a complete solution (a list of portfolio actions for each time step) and **deterministically replays** the entire schedule to verify correctness and compute the score. This is distinct from `grid_optimize` — it does not call the innovator's algorithm. It receives a pre-computed solution and checks:
- Action bounds (power limits) at each step
- SOC feasibility at each step
- Flow feasibility (PTDF line limits) at each step
- Replays the commitment chain to regenerate RT prices at each step
- Accumulates total profit and checks it against the threshold

`verify_solution` is the authoritative scoring function. It re-derives the full commitment chain from the initial seed, so the RT prices it uses are guaranteed to match those that `grid_optimize` would have produced during the rollout.

---

## Commitment Chain

The commitment chain ties RT prices to the innovator's actions via a SHA-256 hash, making future prices **unpredictable** at decision time.

**Initialization:** The instance provides an initial seed \(s_0\).

**At each time step \(t\):**
1. RT prices \(\lambda^{RT}_{i,t}\) are generated from \(s_t\) (plus DA prices and congestion indicators).
2. The innovator observes \(\lambda^{RT}_{i,t}\) and chooses action \(A_t\).
3. The seed advances: \(s_{t+1} = \text{SHA-256}(s_t \| t \| \tilde{u}_1 \dots \tilde{u}_m \| \tilde{E}_1 \dots \tilde{E}_m)\), where \(\tilde{u}_b\) and \(\tilde{E}_b\) are the quantized action and SOC for each battery.
4. \(s_{t+1}\) determines \(\lambda^{RT}_{i,t+1}\) at the next step.

**Why this matters:** Because \(s_{t+1}\) depends on the hash of the action, the innovator cannot predict \(\lambda^{RT}_{t+1}\) when choosing \(A_t\). Any change to the action produces a completely different seed and therefore different future prices. This prevents "oracle" strategies that exploit foreknowledge of prices.

**Verification:** `verify_solution` replays this chain from \(s_0\). Given the same initial seed and the same sequence of actions, it deterministically reconstructs every \(s_t\) and \(\lambda^{RT}_{i,t}\), so the profit is reproducible and tamper-proof.

---

## State, Action, and Transition

### Instance (fixed context)
Generated once by `generate_instance()` and available to the innovator for the entire horizon:
- **DA prices** \(\lambda^{DA}_{i,t}\) for all nodes \(i\) and all times \(t\)
- **Exogenous injections** \(p^{exo}_{i,t}\) for all nodes \(i\) and all times \(t\)
- **PTDF matrix** (encodes network topology and DC power flow)
- **Battery specs** (capacity, power limits, efficiencies, SOC bounds) and **placements** (which node each battery sits at)
- **Market parameters** and **frictions** (transaction costs, degradation)

### State \(S_t\)
The current physical state at time \(t\):
- Time step \(t\)
- \(SOC_{b,t}\) for each battery \(b\)
- \(\lambda^{RT}_{i,t}\) — the real-time price at each node \(i\) (already revealed; the innovator trades against these)

### Action \(A_t\)
The innovator chooses signed power \(p^{stor}_{b,t}\) for each battery \(b\) (positive = discharge, negative = charge). This must satisfy:
- Power bounds: \(-P^c_b \le p^{stor}_{b,t} \le P^d_b\)
- SOC feasibility: the resulting \(SOC_{b,t+1}\) stays within \([E^{min}_b, E^{max}_b]\)
- Flow feasibility: the total nodal injection \(p^{exo}_{i,t} + \sum_{b \in \text{node } i} p^{stor}_{b,t}\), passed through the PTDF matrix, must not violate any line flow limit

### Transition \(S_t \xrightarrow{A_t} S_{t+1}\)

**Deterministic part** (given state + action):
1. SOC update: \(SOC_{b,t+1} = SOC_{b,t} + \eta^c_b \cdot c_b \cdot \Delta t - d_b \cdot \Delta t / \eta^d_b\)
2. Period profit: \(R_t = \sum_b \bigl(p^{stor}_{b,t} \cdot \lambda^{RT}_{node(b),t} \cdot \Delta t - \phi_b(p^{stor}_{b,t})\bigr)\)

**Stochastic part** (unpredictable to the innovator):
3. Commitment: \(s_{t+1} = \text{SHA-256}(s_t \| t \| \tilde{u}_1 \dots \tilde{u}_m \| \tilde{E}_1 \dots \tilde{E}_m)\), where \(\tilde{u}, \tilde{E}\) are quantized actions and SOCs
4. RT price generation: \(\lambda^{RT}_{i,t+1}\) is produced from \(s_{t+1}\), \(\lambda^{DA}_{i,t+1}\), and congestion indicators (which lines are near their limits at time \(t\))

The commitment chain is what makes RT prices **action-dependent yet unpredictable**: the innovator cannot predict \(\lambda^{RT}_{t+1}\) because it depends on a hash of their own action.

---

## `tig-algorithms`

The innovator's code lives here. The file `template.rs` provides the starting point.

### `solve_challenge(instance) -> solution`
Intentionally thin. Its only job is to call `grid_optimize` from `tig-challenges`, passing in the instance and the innovator's `innovators_algorithm`, and return the resulting solution. Innovators should not need to modify this function.

### `innovators_algorithm(state) -> action`
This is the **only function innovators are meant to write**. It maps the current state \(S_t\) to an action \(A_t\). The surrounding rollout logic (looping over time, applying transitions, recording actions) is handled by `grid_optimize`.

In `template.rs`, this function is left as an empty stub (e.g. returning idle actions) for innovators to fill in with their own strategy.
