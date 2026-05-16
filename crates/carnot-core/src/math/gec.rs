//! Gradient-Guided Epsilon Constraint (GEC) Method
//!
//! Spec: REQ-FR11-1683

use ndarray::{Array1, ArrayView1};
use crate::Float;

/// Projects a gradient `grad` onto the feasible region defined by the reference
/// gradient `ref_grad` and epsilon, to prevent catastrophic forgetting.
///
/// If `dot(grad, ref_grad) < -epsilon`, it projects `grad` to ensure
/// `dot(grad_proj, ref_grad) >= -epsilon`.
///
/// Formula: g_proj = g - ((dot(g, g_ref) + epsilon) / ||g_ref||^2) * g_ref
pub fn project_gradient(
    grad: ArrayView1<Float>,
    ref_grad: ArrayView1<Float>,
    epsilon: Float,
) -> Array1<Float> {
    let dot_product = grad.dot(&ref_grad);
    if dot_product >= -epsilon {
        return grad.to_owned();
    }
    
    let ref_norm_sq = ref_grad.dot(&ref_grad);
    if ref_norm_sq == 0.0 {
        return grad.to_owned();
    }
    
    let factor = (dot_product + epsilon) / ref_norm_sq;
    let mut proj = grad.to_owned();
    proj.scaled_add(-factor, &ref_grad);
    proj
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn assert_array_eq(a: &Array1<Float>, b: &Array1<Float>) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert!((a[i] - b[i]).abs() < 1e-5, "Mismatch at {}: {} != {}", i, a[i], b[i]);
        }
    }

    #[test]
    fn test_gec_projection_no_conflict() {
        // Spec traces: REQ-FR11-1683
        let grad = array![1.0, 1.0];
        let ref_grad = array![1.0, 1.0];
        let proj = project_gradient(grad.view(), ref_grad.view(), 0.0);
        assert_array_eq(&proj, &array![1.0, 1.0]);
    }

    #[test]
    fn test_gec_projection_conflict_strict() {
        // Spec traces: REQ-FR11-1683
        let grad = array![-1.0, 0.0];
        let ref_grad = array![1.0, 0.0];
        let proj = project_gradient(grad.view(), ref_grad.view(), 0.0);
        assert_array_eq(&proj, &array![0.0, 0.0]);
    }
    
    #[test]
    fn test_gec_projection_conflict_epsilon() {
        // Spec traces: REQ-FR11-1683
        let grad = array![-1.0, 0.0];
        let ref_grad = array![1.0, 0.0];
        let proj = project_gradient(grad.view(), ref_grad.view(), 0.5);
        assert_array_eq(&proj, &array![-0.5, 0.0]);
    }
}
