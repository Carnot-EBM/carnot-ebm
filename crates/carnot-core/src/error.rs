//! Error types for the Carnot framework.

/// Top-level error type for Carnot operations.
#[derive(Debug, thiserror::Error)]
pub enum CarnotError {
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Numeric error: {0}")]
    Numeric(String),
}
