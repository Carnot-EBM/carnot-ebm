//! WebSocket protocol messages between gateway and browser workers.

use serde::{Deserialize, Serialize};

/// Message from gateway to browser worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    /// Welcome message with worker ID.
    #[serde(rename = "welcome")]
    Welcome { worker_id: String },

    /// Work unit: compute energy for a batch of inputs.
    #[serde(rename = "compute")]
    Compute {
        /// Unique work ID for tracking.
        work_id: String,
        /// WGSL shader source code.
        shader: String,
        /// Flattened coupling matrix (dim*dim f32).
        coupling: Vec<f32>,
        /// Bias vector (dim f32).
        bias: Vec<f32>,
        /// Flattened input batch (batch_size * dim f32).
        inputs: Vec<f32>,
        /// Input dimensionality.
        dim: u32,
        /// Number of samples in the batch.
        batch_size: u32,
    },

    /// No work available, wait.
    #[serde(rename = "idle")]
    Idle,
}

/// Message from browser worker to gateway.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WorkerMessage {
    /// Worker is ready for work.
    #[serde(rename = "ready")]
    Ready {
        /// GPU name reported by WebGPU adapter.
        gpu_name: String,
    },

    /// Work result: computed energies.
    #[serde(rename = "result")]
    Result {
        work_id: String,
        energies: Vec<f32>,
        /// Compute time in milliseconds (as reported by the browser).
        compute_ms: f64,
    },

    /// Worker encountered an error.
    #[serde(rename = "error")]
    Error { work_id: String, message: String },

    /// Heartbeat — worker is still alive.
    #[serde(rename = "heartbeat")]
    Heartbeat,
}
