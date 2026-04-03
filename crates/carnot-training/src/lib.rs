//! carnot-training: Training algorithms for Energy Based Models.
//!
//! Spec: REQ-TRAIN-001, REQ-TRAIN-002, REQ-TRAIN-003, REQ-TRAIN-004

pub mod cd;
pub mod optimizer;
pub mod score_matching;

use carnot_core::Float;

/// Result of a single training step.
pub struct TrainStepResult {
    pub loss: Float,
    pub grad_norm: Float,
}
