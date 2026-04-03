//! Parameter initialization strategies.
//!
//! Spec: REQ-TIER-006

use ndarray::Array2;
use ndarray_rand::RandomExt;
use rand_distr::{Normal, Uniform};

use crate::Float;

/// Parameter initialization strategy.
///
/// Spec: REQ-TIER-006
pub enum Initializer {
    /// Xavier/Glorot uniform initialization.
    XavierUniform,
    /// He/Kaiming normal initialization.
    HeNormal,
    /// Zero initialization (for biases).
    Zeros,
}

impl Initializer {
    /// Create a weight matrix with the given shape using this initialization strategy.
    pub fn init_matrix(&self, rows: usize, cols: usize) -> Array2<Float> {
        match self {
            Initializer::XavierUniform => {
                let limit = (6.0 / (rows + cols) as Float).sqrt();
                Array2::random((rows, cols), Uniform::new(-limit, limit))
            }
            Initializer::HeNormal => {
                let std = (2.0 / rows as Float).sqrt();
                Array2::random((rows, cols), Normal::new(0.0 as Float, std).unwrap())
            }
            Initializer::Zeros => Array2::zeros((rows, cols)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xavier_uniform_shape() {
        // REQ-TIER-006
        let w = Initializer::XavierUniform.init_matrix(10, 5);
        assert_eq!(w.shape(), &[10, 5]);
    }

    #[test]
    fn test_xavier_uniform_bounds() {
        // REQ-TIER-006
        let w = Initializer::XavierUniform.init_matrix(100, 100);
        let limit = (6.0 / 200.0 as Float).sqrt();
        assert!(w.iter().all(|&v| v >= -limit && v <= limit));
    }

    #[test]
    fn test_he_normal_shape() {
        // REQ-TIER-006
        let w = Initializer::HeNormal.init_matrix(10, 5);
        assert_eq!(w.shape(), &[10, 5]);
    }

    #[test]
    fn test_zeros() {
        // REQ-TIER-006
        let w = Initializer::Zeros.init_matrix(3, 4);
        assert!(w.iter().all(|&v| v == 0.0));
    }
}
