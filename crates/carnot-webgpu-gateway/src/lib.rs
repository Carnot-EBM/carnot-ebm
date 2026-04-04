//! # carnot-webgpu-gateway: Distributed GPU compute via WebGPU
//!
//! A WebSocket gateway that distributes energy computation work to browser
//! clients running WebGPU. Browsers visit a webpage, connect via WebSocket,
//! and contribute their GPU for Carnot energy computations.
//!
//! ## Architecture
//!
//! ```text
//! Browser (WebGPU) ──┐
//! Browser (WebGPU) ──┼──> Gateway (Rust, axum+tokio) ──> Carnot
//! Browser (WebGPU) ──┘
//! ```
//!
//! ## How it works
//!
//! 1. Gateway serves a static HTML+JS page with WebGPU compute code
//! 2. Browsers visit the page and connect via WebSocket
//! 3. Carnot submits work units (batch energy evaluation)
//! 4. Gateway distributes work across connected browsers
//! 5. Each browser runs the WGSL compute shader on its GPU
//! 6. Results are sent back via WebSocket and aggregated
//!
//! ## Security
//!
//! - Browsers only see energy function parameters, not training data
//! - Results are verified server-side (untrusted browser output)
//! - WebGPU runs in browser sandbox (no filesystem/network access)
//! - WGSL shaders are compiled by the browser, not arbitrary code

pub mod gateway;
pub mod protocol;
pub mod worker_page;
