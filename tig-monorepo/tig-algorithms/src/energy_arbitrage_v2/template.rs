use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage_v2::*;

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    _hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    // Innovators: implement your algorithm in `innovators_algorithm` below,
    // then uncomment the following lines:
    //
    //   let solution = challenge.grid_optimize(&innovators_algorithm)?;
    //   save_solution(&solution)?;
    //   Ok(())
    //
    // Your algorithm receives the challenge instance and current state at each
    // time step, and must return a PortfolioAction.
    //
    // You can use challenge.take_step(state, action, rt_prices) for what-if
    // simulation (you must supply your own RT price forecast).

    Err(anyhow!("Algorithm not implemented"))
}

fn innovators_algorithm(
    challenge: &Challenge,
    state: &State,
) -> Result<PortfolioAction> {
    // Return idle actions as placeholder.
    // Replace this with your strategy.
    Ok(PortfolioAction {
        actions: vec![SignedAction::idle(); challenge.batteries.len()],
    })
}

pub fn help() {
    println!("Energy Arbitrage v2 template algorithm");
    println!("This is a placeholder. Implement innovators_algorithm to submit a solution.");
    println!();
    println!("Pattern:");
    println!("  1. Implement innovators_algorithm(challenge, state) -> PortfolioAction");
    println!("  2. solve_challenge calls challenge.grid_optimize(&innovators_algorithm)");
    println!("  3. Use challenge.take_step() for simulation/lookahead");
}
