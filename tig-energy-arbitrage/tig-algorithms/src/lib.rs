//! TIG Algorithms Crate
//!
//! Contains algorithm implementations for TIG challenges.
//!
//! ## Energy Arbitrage
//!
//! Original single-asset implementation in [`energy_arbitrage`].
//!
//! ## Energy Arbitrage V2
//!
//! Two-level challenge implementation in [`energy_arbitrage_v2`]:
//! - Level 1: Single-asset with action-committed pricing
//! - Level 2: Portfolio arbitrage on constrained network

pub mod energy_arbitrage;
pub mod energy_arbitrage_v2;
