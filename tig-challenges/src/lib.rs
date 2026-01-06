//! TIG Challenges Crate
//!
//! Contains challenge definitions for The Innovation Game.
//!
//! ## Challenge: Energy Arbitrage
//!
//! Two-level challenge for optimizing battery storage arbitrage:
//!
//! - **Level 1**: Single-asset temporal arbitrage with action-committed pricing
//! - **Level 2**: Portfolio arbitrage on a transmission-constrained network with 5 tracks
//!
//! See [`energy_arbitrage_v2`] for the full two-level implementation.
//!
//! ## Track System (Level 2)
//!
//! Level 2 has five predefined tracks with increasing difficulty:
//! - Track 1: Small network (20 nodes), nominal limits, low volatility
//! - Track 2: Medium network (40 nodes), tighter limits, more volatility
//! - Track 3: Large network (80 nodes), 2-day horizon, frequent spikes
//! - Track 4: Dense network (100 nodes), heavy tails, congestion-critical
//! - Track 5: Capstone (150 nodes), tightest limits, heaviest tails

pub mod energy_arbitrage;
pub mod energy_arbitrage_v2;

// Re-export common types from original module (backward compatibility)
pub use energy_arbitrage::{Challenge, Solution, Difficulty};

// Re-export v2 types
pub use energy_arbitrage_v2::{
    // Constants module
    constants,
    // Track system
    Track, TrackParameters,
    // Common types
    BatterySpec, Frictions, MarketParams, Action,
    // Level 1
    Level1Challenge, Level1Solution, Level1Difficulty, TranscriptEntry,
    // Level 2
    Level2Challenge, Level2Solution, Level2Difficulty,
    Network, PlacedBattery, PortfolioAction, SignedAction,
    // Legacy compatibility
    LegacyPortfolioAction, LegacyLevel2Solution,
};
