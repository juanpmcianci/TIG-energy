//! TIG Challenges Crate
//!
//! Contains challenge definitions for The Innovation Game.
//!
//! ## Challenge: Energy Arbitrage (Network)
//!
//! Portfolio arbitrage on a transmission-constrained network with 5 tracks.
//!
//! See [`energy_arbitrage_v2`] for the full implementation.
//!
//! ## Track System
//!
//! The challenge has five predefined tracks with increasing difficulty:
//! - Track 1: Small network (20 nodes), nominal limits, low volatility
//! - Track 2: Medium network (40 nodes), tighter limits, more volatility
//! - Track 3: Large network (80 nodes), 2-day horizon, frequent spikes
//! - Track 4: Dense network (100 nodes), heavy tails, congestion-critical
//! - Track 5: Capstone (150 nodes), tightest limits, heaviest tails

pub mod energy_arbitrage_v2;

// Re-export types
pub use energy_arbitrage_v2::{
    // Constants module
    constants,
    // Track system
    Track, TrackParameters,
    // Common types
    BatterySpec, Frictions, MarketParams,
    // Challenge types
    Level2Challenge, Level2Solution, Level2Difficulty,
    Network, PlacedBattery, PortfolioAction, SignedAction,
};
