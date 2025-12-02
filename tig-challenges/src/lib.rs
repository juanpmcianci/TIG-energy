//! TIG Challenges Crate
//!
//! Contains challenge definitions for The Innovation Game.
//!
//! ## Challenge: Energy Arbitrage
//!
//! Two-level challenge for optimizing battery storage arbitrage:
//!
//! - **Level 1**: Single-asset temporal arbitrage with action-committed pricing
//! - **Level 2**: Portfolio arbitrage on a transmission-constrained network
//!
//! See [`energy_arbitrage_v2`] for the full two-level implementation.

pub mod energy_arbitrage;
pub mod energy_arbitrage_v2;

// Re-export common types from original module (backward compatibility)
pub use energy_arbitrage::{Challenge, Solution, Difficulty};

// Re-export v2 types
pub use energy_arbitrage_v2::{
    // Common types
    BatterySpec, Frictions, MarketParams, Action,
    // Level 1
    Level1Challenge, Level1Solution, Level1Difficulty, TranscriptEntry,
    // Level 2
    Level2Challenge, Level2Solution, Level2Difficulty,
    Network, PlacedBattery, PortfolioAction,
};
