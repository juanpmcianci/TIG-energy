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
- This gives innovators a way to simulate “what-if” charging/discharging scenarios under hypothetical RT price paths.

---

## `tig-algorithms`

### `solve_challenge(instance)`
This is the entry point on the innovator side. It takes a challenge instance and solves it by calling the shared optimizer:

- imports `grid_optimize` from `tig-challenges`
- passes in the instance and the innovator’s own `innovators_algorithm`
- returns the resulting list of actions (the solution)

### `innovators_algorithm(state) -> action`
This is the function innovators are meant to write. It maps the current state \(S_t\) to an action \(A_t\). The surrounding rollout logic (looping over time, applying transitions, recording actions) is handled by `grid_optimize`.
