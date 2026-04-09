//! Built-in constraint types that implement `ConstraintTerm`.
//!
//! ## For Researchers
//!
//! Provides three reusable constraint primitives:
//! - `BoundConstraint`: box constraint E(x) = max(0, lo - x_i)^2 + max(0, x_i - hi)^2
//! - `EqualityConstraint`: pinning constraint E(x) = (x_i - target)^2
//! - `IsingConstraint`: wraps an Ising model as E(x) = max(0, ising_energy(x))
//!
//! All return energy >= 0 with energy = 0 iff satisfied, and provide analytical gradients.
//!
//! ## For Engineers
//!
//! These are the "Lego bricks" of constraint-based verification. Instead of writing
//! custom constraint logic every time, you compose these primitives:
//!
//! ```text
//! // "Temperature must be between 20 and 30 degrees"
//! BoundConstraint::new("temp", 0, 20.0, 30.0)
//!
//! // "Pressure must equal 101.3 kPa (within 0.1)"
//! EqualityConstraint::new("pressure", 1, 101.3, 0.1)
//!
//! // "Configuration must satisfy pairwise logic encoded in Ising couplings"
//! IsingConstraint::new("sat_clause", ising_model, 1e-4)
//! ```
//!
//! Spec: REQ-VERIFY-001

use carnot_core::verify::ConstraintTerm;
use carnot_core::Float;
use carnot_ising::IsingModel;
use ndarray::{Array1, ArrayView1};

/// Constraint: value at a specific index must be within `[lo, hi]`.
///
/// ## For Researchers
///
/// Box constraint on a single dimension. Energy is the squared distance to the
/// feasible interval: E(x) = max(0, lo - x_i)^2 + max(0, x_i - hi)^2.
/// Gradient is piecewise linear: -2(lo - x_i) if below, +2(x_i - hi) if above, 0 inside.
///
/// ## For Engineers
///
/// This is the most common constraint: "this value must be between A and B."
///
/// Examples:
/// - "Pixel values must be in [0, 255]"
/// - "Temperature must be between -40 and 50 degrees"
/// - "Probability must be in [0, 1]"
///
/// The energy is zero when the value is inside the bounds, and grows quadratically
/// as the value moves outside. The quadratic growth means the gradient gets stronger
/// the further you are from the valid range, which helps repair converge quickly.
///
/// ```text
/// Energy landscape for BoundConstraint(lo=2, hi=5):
///
///   E |
///     |*                              *
///     | *                            *
///     |  *                          *
///     |   *                        *
///     |    *______________________*
///     |    2                      5
///     +---------------------------------> x
/// ```
///
/// Spec: REQ-VERIFY-001
pub struct BoundConstraint {
    /// Human-readable name (e.g., "pixel_42_range", "temperature_bounds").
    constraint_name: String,
    /// Which dimension of the input vector this constraint applies to.
    index: usize,
    /// Lower bound (inclusive). The value must be >= this.
    lo: Float,
    /// Upper bound (inclusive). The value must be <= this.
    hi: Float,
}

impl BoundConstraint {
    /// Create a new bound constraint: x[index] must be in [lo, hi].
    ///
    /// # Arguments
    /// * `name` - Human-readable name for verification reports
    /// * `index` - Which element of the input vector to constrain
    /// * `lo` - Lower bound (inclusive)
    /// * `hi` - Upper bound (inclusive). Must be >= lo.
    ///
    /// # Panics
    /// Panics if `lo > hi` (empty interval makes no sense).
    pub fn new(name: &str, index: usize, lo: Float, hi: Float) -> Self {
        assert!(
            lo <= hi,
            "BoundConstraint: lo ({lo}) must be <= hi ({hi})"
        );
        Self {
            constraint_name: name.to_string(),
            index,
            lo,
            hi,
        }
    }
}

impl ConstraintTerm for BoundConstraint {
    fn name(&self) -> &str {
        &self.constraint_name
    }

    /// Energy = max(0, lo - x_i)^2 + max(0, x_i - hi)^2.
    ///
    /// Zero inside [lo, hi], quadratic penalty outside.
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        let val = x[self.index];
        let below = (self.lo - val).max(0.0);
        let above = (val - self.hi).max(0.0);
        below * below + above * above
    }

    /// Gradient: nonzero only at the constrained index, and only when violated.
    ///
    /// - Below lo: grad[index] = -2(lo - x_i) (negative, pushing x_i up)
    /// - Above hi: grad[index] = 2(x_i - hi) (positive, pushing x_i down via descent)
    /// - Inside [lo, hi]: grad = 0
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        let mut grad = Array1::zeros(x.len());
        let val = x[self.index];
        if val < self.lo {
            // Below lower bound: energy = (lo - val)^2, dE/dval = -2(lo - val)
            grad[self.index] = -2.0 * (self.lo - val);
        } else if val > self.hi {
            // Above upper bound: energy = (val - hi)^2, dE/dval = 2(val - hi)
            grad[self.index] = 2.0 * (val - self.hi);
        }
        grad
    }
}

/// Constraint: value at a specific index must equal a target (within tolerance).
///
/// ## For Researchers
///
/// Pinning constraint: E(x) = (x_i - target)^2. Satisfied when |x_i - target| <= tol.
/// Gradient: dE/dx_i = 2(x_i - target).
///
/// ## For Engineers
///
/// This constraint says "this value must be exactly (or very close to) some target."
///
/// Examples:
/// - "Cell (2,5) in the Sudoku must be 7" (a given clue)
/// - "The total probability must sum to 1.0"
/// - "The output voltage must be 5.0V"
///
/// The tolerance parameter controls how close is "close enough." A tolerance of
/// 0.01 means the constraint is satisfied if the value is within 0.01 of the target.
/// The satisfaction threshold is set to tolerance^2 (since energy = (x - target)^2).
///
/// Spec: REQ-VERIFY-001
pub struct EqualityConstraint {
    /// Human-readable name for verification reports.
    constraint_name: String,
    /// Which dimension of the input vector this constraint applies to.
    index: usize,
    /// The target value that x[index] must equal.
    target: Float,
    /// How close is "close enough." Satisfaction threshold = tol^2.
    tol: Float,
}

impl EqualityConstraint {
    /// Create a new equality constraint: x[index] must equal target (within tol).
    ///
    /// # Arguments
    /// * `name` - Human-readable name for verification reports
    /// * `index` - Which element of the input vector to constrain
    /// * `target` - The value x[index] should equal
    /// * `tol` - Tolerance: satisfied when |x[index] - target| <= tol
    pub fn new(name: &str, index: usize, target: Float, tol: Float) -> Self {
        Self {
            constraint_name: name.to_string(),
            index,
            target,
            tol,
        }
    }
}

impl ConstraintTerm for EqualityConstraint {
    fn name(&self) -> &str {
        &self.constraint_name
    }

    /// Energy = (x_i - target)^2. Zero when x_i == target.
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        let diff = x[self.index] - self.target;
        diff * diff
    }

    /// Gradient: dE/dx_i = 2(x_i - target). Zero at all other indices.
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        let mut grad = Array1::zeros(x.len());
        grad[self.index] = 2.0 * (x[self.index] - self.target);
        grad
    }

    /// Satisfied when |x_i - target| <= tol, i.e., energy <= tol^2.
    fn satisfaction_threshold(&self) -> Float {
        self.tol * self.tol
    }
}

/// Constraint: configuration must have low energy under an Ising model.
///
/// ## For Researchers
///
/// Wraps `IsingModel` as a constraint term. The Ising Hamiltonian
/// H(x) = -0.5 x^T J x - b^T x can encode SAT clauses, graph coloring,
/// and other combinatorial problems via appropriate coupling matrices.
/// The constraint energy is max(0, H(x) - threshold), so configurations
/// with Ising energy <= threshold are considered satisfied.
///
/// ## For Engineers
///
/// An Ising model assigns an energy to every configuration based on pairwise
/// interactions between variables. By wrapping it as a constraint, you can say
/// "this configuration must be a low-energy state of this interaction network."
///
/// This is useful for encoding logical/combinatorial constraints:
/// - **SAT problems**: Each clause becomes a pairwise coupling
/// - **Graph coloring**: Adjacent nodes with same color get high energy
/// - **Spin glasses**: Physical systems with competing interactions
///
/// The `energy_threshold` parameter sets what counts as "low enough." An Ising
/// model can have negative energies (which are good — they mean strong agreement
/// between coupled variables), so the threshold might be negative too.
///
/// Spec: REQ-VERIFY-001
pub struct IsingConstraint {
    /// Human-readable name for verification reports.
    constraint_name: String,
    /// The Ising model whose energy landscape defines the constraint.
    model: IsingModel,
    /// Ising energies at or below this value are considered "satisfied."
    /// For a well-trained Ising model encoding SAT clauses, this would be
    /// the energy of a satisfying assignment (typically negative).
    energy_threshold: Float,
}

impl IsingConstraint {
    /// Create a new Ising constraint.
    ///
    /// # Arguments
    /// * `name` - Human-readable name for verification reports
    /// * `model` - The Ising model defining the energy landscape
    /// * `energy_threshold` - Ising energy at or below this is "satisfied"
    pub fn new(name: &str, model: IsingModel, energy_threshold: Float) -> Self {
        Self {
            constraint_name: name.to_string(),
            model,
            energy_threshold,
        }
    }
}

impl ConstraintTerm for IsingConstraint {
    fn name(&self) -> &str {
        &self.constraint_name
    }

    /// Energy = max(0, ising_energy(x) - threshold).
    ///
    /// If the Ising energy is below the threshold, the constraint is satisfied
    /// (energy = 0). If it's above, the constraint energy equals the excess,
    /// which gradient descent can reduce.
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        use carnot_core::EnergyFunction;
        let ising_e = self.model.energy(x);
        (ising_e - self.energy_threshold).max(0.0)
    }

    /// Gradient: if Ising energy > threshold, gradient = Ising gradient. Else zero.
    ///
    /// The gradient of max(0, f(x)) is f'(x) when f(x) > 0, and 0 otherwise.
    /// This means the constraint only "pushes" the configuration when it's
    /// actually violated — consistent with the selective repair philosophy.
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        use carnot_core::EnergyFunction;
        let ising_e = self.model.energy(x);
        if ising_e > self.energy_threshold {
            self.model.grad_energy(x)
        } else {
            Array1::zeros(x.len())
        }
    }

    /// Satisfied when Ising energy <= threshold, i.e., constraint energy <= 0.
    /// We use a small epsilon to handle floating-point edge cases.
    fn satisfaction_threshold(&self) -> Float {
        1e-6
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use carnot_ising::IsingConfig;
    use ndarray::array;

    // --- BoundConstraint tests ---

    #[test]
    fn test_bound_satisfied_inside() {
        // REQ-VERIFY-001: value inside bounds -> energy = 0
        let c = BoundConstraint::new("test_bound", 0, 1.0, 5.0);
        let x = array![3.0, 0.0];
        assert!(c.energy(&x.view()) < 1e-10);
        assert!(c.is_satisfied(&x.view()));
    }

    #[test]
    fn test_bound_satisfied_at_boundary() {
        // REQ-VERIFY-001: value exactly at boundary -> energy = 0
        let c = BoundConstraint::new("test_bound", 0, 1.0, 5.0);
        let x_lo = array![1.0, 0.0];
        let x_hi = array![5.0, 0.0];
        assert!(c.energy(&x_lo.view()) < 1e-10);
        assert!(c.energy(&x_hi.view()) < 1e-10);
    }

    #[test]
    fn test_bound_violated_below() {
        // REQ-VERIFY-001: value below lo -> positive energy
        let c = BoundConstraint::new("test_bound", 0, 2.0, 5.0);
        let x = array![0.0, 0.0]; // 2.0 below lo
        let e = c.energy(&x.view());
        // (2.0 - 0.0)^2 = 4.0
        assert!((e - 4.0).abs() < 1e-6);
        assert!(!c.is_satisfied(&x.view()));
    }

    #[test]
    fn test_bound_violated_above() {
        // REQ-VERIFY-001: value above hi -> positive energy
        let c = BoundConstraint::new("test_bound", 0, 1.0, 3.0);
        let x = array![5.0, 0.0]; // 2.0 above hi
        let e = c.energy(&x.view());
        // (5.0 - 3.0)^2 = 4.0
        assert!((e - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_bound_gradient_below() {
        // REQ-VERIFY-001: gradient pushes value up when below lo
        let c = BoundConstraint::new("test_bound", 0, 2.0, 5.0);
        let x = array![0.0, 0.0];
        let grad = c.grad_energy(&x.view());
        // dE/dx[0] = -2*(2.0 - 0.0) = -4.0 (negative = descent increases x[0])
        assert!((grad[0] - (-4.0)).abs() < 1e-6);
        assert!(grad[1].abs() < 1e-10);
    }

    #[test]
    fn test_bound_gradient_above() {
        // REQ-VERIFY-001: gradient pushes value down when above hi
        let c = BoundConstraint::new("test_bound", 0, 1.0, 3.0);
        let x = array![5.0, 0.0];
        let grad = c.grad_energy(&x.view());
        // dE/dx[0] = 2*(5.0 - 3.0) = 4.0 (positive = descent decreases x[0])
        assert!((grad[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_bound_gradient_inside() {
        // REQ-VERIFY-001: zero gradient when inside bounds
        let c = BoundConstraint::new("test_bound", 0, 1.0, 5.0);
        let x = array![3.0, 7.0];
        let grad = c.grad_energy(&x.view());
        assert!(grad.iter().all(|&g| g.abs() < 1e-10));
    }

    #[test]
    fn test_bound_gradient_finite_difference() {
        // Verify analytical gradient matches numerical gradient
        let c = BoundConstraint::new("test_bound", 0, 2.0, 5.0);
        let x = array![0.5, 1.0]; // below lo
        let grad = c.grad_energy(&x.view());
        let eps: Float = 1e-4;
        for i in 0..x.len() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let fd = (c.energy(&xp.view()) - c.energy(&xm.view())) / (2.0 * eps);
            assert!(
                (grad[i] - fd).abs() < 1e-2,
                "Bound grad mismatch at {i}: analytic={}, fd={fd}",
                grad[i]
            );
        }
    }

    #[test]
    #[should_panic(expected = "lo")]
    fn test_bound_invalid_range() {
        BoundConstraint::new("bad", 0, 5.0, 1.0);
    }

    // --- EqualityConstraint tests ---

    #[test]
    fn test_equality_satisfied() {
        // REQ-VERIFY-001: value at target -> energy = 0
        let c = EqualityConstraint::new("test_eq", 0, 3.0, 0.01);
        let x = array![3.0, 0.0];
        assert!(c.energy(&x.view()) < 1e-10);
        assert!(c.is_satisfied(&x.view()));
    }

    #[test]
    fn test_equality_satisfied_within_tolerance() {
        // REQ-VERIFY-001: value within tolerance -> satisfied
        let c = EqualityConstraint::new("test_eq", 0, 3.0, 0.1);
        let x = array![3.05, 0.0]; // diff = 0.05, energy = 0.0025, threshold = 0.01
        assert!(c.is_satisfied(&x.view()));
    }

    #[test]
    fn test_equality_violated() {
        // REQ-VERIFY-001: value far from target -> positive energy
        let c = EqualityConstraint::new("test_eq", 0, 3.0, 0.01);
        let x = array![5.0, 0.0]; // diff = 2.0, energy = 4.0
        let e = c.energy(&x.view());
        assert!((e - 4.0).abs() < 1e-6);
        assert!(!c.is_satisfied(&x.view()));
    }

    #[test]
    fn test_equality_gradient() {
        // REQ-VERIFY-001: gradient points away from target
        let c = EqualityConstraint::new("test_eq", 1, 3.0, 0.01);
        let x = array![0.0, 5.0]; // x[1] = 5.0, target = 3.0
        let grad = c.grad_energy(&x.view());
        // dE/dx[1] = 2*(5.0 - 3.0) = 4.0
        assert!((grad[1] - 4.0).abs() < 1e-6);
        assert!(grad[0].abs() < 1e-10);
    }

    #[test]
    fn test_equality_gradient_finite_difference() {
        let c = EqualityConstraint::new("test_eq", 0, 3.0, 0.01);
        let x = array![1.0, 2.0];
        let grad = c.grad_energy(&x.view());
        let eps: Float = 1e-4;
        for i in 0..x.len() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let fd = (c.energy(&xp.view()) - c.energy(&xm.view())) / (2.0 * eps);
            assert!(
                (grad[i] - fd).abs() < 1e-2,
                "Equality grad mismatch at {i}: analytic={}, fd={fd}",
                grad[i]
            );
        }
    }

    // --- IsingConstraint tests ---

    #[test]
    fn test_ising_constraint_satisfied() {
        // REQ-VERIFY-001: Ising energy below threshold -> constraint satisfied
        let model = IsingModel::new(IsingConfig {
            input_dim: 3,
            hidden_dim: None,
            coupling_init: "zeros".to_string(),
        })
        .unwrap();
        // With zero couplings and zero bias, energy is always 0.
        let c = IsingConstraint::new("test_ising", model, 0.1);
        let x = array![1.0, -1.0, 0.5];
        assert!(c.energy(&x.view()) < 1e-10);
        assert!(c.is_satisfied(&x.view()));
    }

    #[test]
    fn test_ising_constraint_violated() {
        // REQ-VERIFY-001: Ising energy above threshold -> constraint violated
        let mut model = IsingModel::new(IsingConfig {
            input_dim: 2,
            hidden_dim: None,
            coupling_init: "zeros".to_string(),
        })
        .unwrap();
        // Set coupling so that same-sign variables have high energy.
        // J = [[ 0, -1], [-1,  0]] -> E(x) = -0.5 * x^T J x = 0.5 * x0*x1 + 0.5 * x1*x0 = x0*x1
        // Wait, E = -0.5 * x^T J x. With J[0,1] = J[1,0] = -1:
        // x^T J x = x0*(-1)*x1 + x1*(-1)*x0 = -2*x0*x1
        // E = -0.5 * (-2*x0*x1) = x0*x1
        // For x=[1,1]: E = 1.0. For x=[1,-1]: E = -1.0.
        model.coupling[[0, 1]] = -1.0;
        model.coupling[[1, 0]] = -1.0;

        let c = IsingConstraint::new("test_ising", model, 0.0);
        // x=[1,1]: Ising energy = 1.0, threshold = 0.0, constraint energy = 1.0
        let x = array![1.0, 1.0];
        let e = c.energy(&x.view());
        assert!((e - 1.0).abs() < 1e-6);
        assert!(!c.is_satisfied(&x.view()));
    }

    #[test]
    fn test_ising_constraint_gradient_when_violated() {
        let mut model = IsingModel::new(IsingConfig {
            input_dim: 2,
            hidden_dim: None,
            coupling_init: "zeros".to_string(),
        })
        .unwrap();
        model.coupling[[0, 1]] = -1.0;
        model.coupling[[1, 0]] = -1.0;

        let c = IsingConstraint::new("test_ising", model, 0.0);
        let x = array![1.0, 1.0]; // violated
        let grad = c.grad_energy(&x.view());
        // Ising grad = -J*x - b = -[[-1*1 + 0*1], [0*1 + -1*1]] = [1, 1]
        // Wait: J = [[0, -1], [-1, 0]], so J*x = [0*1 + (-1)*1, (-1)*1 + 0*1] = [-1, -1]
        // grad_ising = -J*x - b = -[-1, -1] - [0, 0] = [1, 1]
        assert!((grad[0] - 1.0).abs() < 1e-6);
        assert!((grad[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ising_constraint_zero_grad_when_satisfied() {
        let model = IsingModel::new(IsingConfig {
            input_dim: 3,
            hidden_dim: None,
            coupling_init: "zeros".to_string(),
        })
        .unwrap();
        let c = IsingConstraint::new("test_ising", model, 0.1);
        let x = array![1.0, -1.0, 0.5];
        let grad = c.grad_energy(&x.view());
        assert!(grad.iter().all(|&g| g.abs() < 1e-10));
    }
}
