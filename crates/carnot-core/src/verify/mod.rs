//! Verifiable reasoning: constraints as energy terms.
//!
//! The core insight: a constraint is an energy function that returns 0 when satisfied
//! and > 0 when violated. The same function trains AND verifies.
//!
//! Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, REQ-VERIFY-004

pub mod sudoku;

use ndarray::{Array1, ArrayView1};

use crate::{EnergyFunction, Float};

/// A single verifiable constraint, expressed as an energy term.
///
/// Satisfied constraint → energy = 0.
/// Violated constraint → energy > 0, proportional to violation severity.
///
/// Spec: REQ-VERIFY-001
pub trait ConstraintTerm: Send + Sync {
    /// Human-readable name for this constraint.
    fn name(&self) -> &str;

    /// Compute constraint energy. Returns 0.0 if satisfied.
    fn energy(&self, x: &ArrayView1<Float>) -> Float;

    /// Gradient of constraint energy w.r.t. x.
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float>;

    /// Threshold below which the constraint is considered satisfied.
    fn satisfaction_threshold(&self) -> Float {
        1e-6
    }

    /// Is this constraint satisfied for configuration x?
    fn is_satisfied(&self, x: &ArrayView1<Float>) -> bool {
        self.energy(x) <= self.satisfaction_threshold()
    }
}

/// Report for a single constraint's evaluation.
///
/// Spec: REQ-VERIFY-002, REQ-VERIFY-003
#[derive(Debug, Clone)]
pub struct ConstraintReport {
    pub name: String,
    pub energy: Float,
    pub weighted_energy: Float,
    pub satisfied: bool,
}

/// Verification verdict.
///
/// Spec: REQ-VERIFY-003
#[derive(Debug, Clone)]
pub enum Verdict {
    /// All constraints satisfied.
    Verified,
    /// One or more constraints violated, with failing constraint names.
    Violated { failing: Vec<String> },
}

/// Complete verification result for a configuration.
///
/// Spec: REQ-VERIFY-003
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub total_energy: Float,
    pub constraints: Vec<ConstraintReport>,
    pub verdict: Verdict,
}

impl VerificationResult {
    /// Returns true if all constraints are satisfied.
    pub fn is_verified(&self) -> bool {
        matches!(self.verdict, Verdict::Verified)
    }

    /// Returns the names of failing constraints, if any.
    pub fn failing_constraints(&self) -> Vec<&str> {
        match &self.verdict {
            Verdict::Verified => vec![],
            Verdict::Violated { failing } => failing.iter().map(|s| s.as_str()).collect(),
        }
    }
}

/// An energy function composed of weighted constraint terms.
///
/// This is the primary structure for verifiable reasoning: each constraint
/// contributes independently to the total energy, and the decomposition
/// reveals exactly which constraints are satisfied or violated.
///
/// Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, REQ-VERIFY-004
pub struct ComposedEnergy {
    terms: Vec<(Box<dyn ConstraintTerm>, Float)>,
    input_dim: usize,
}

impl ComposedEnergy {
    /// Create a new composed energy function.
    pub fn new(input_dim: usize) -> Self {
        Self {
            terms: Vec::new(),
            input_dim,
        }
    }

    /// Add a constraint term with a weight.
    ///
    /// Spec: REQ-VERIFY-004
    pub fn add_constraint(&mut self, term: Box<dyn ConstraintTerm>, weight: Float) {
        self.terms.push((term, weight));
    }

    /// Number of constraint terms.
    pub fn num_constraints(&self) -> usize {
        self.terms.len()
    }

    /// Per-constraint energy decomposition.
    ///
    /// Spec: REQ-VERIFY-002
    pub fn decompose(&self, x: &ArrayView1<Float>) -> Vec<ConstraintReport> {
        self.terms
            .iter()
            .map(|(term, weight)| {
                let raw_energy = term.energy(x);
                ConstraintReport {
                    name: term.name().to_string(),
                    energy: raw_energy,
                    weighted_energy: weight * raw_energy,
                    satisfied: term.is_satisfied(x),
                }
            })
            .collect()
    }

    /// Produce a full verification result.
    ///
    /// Spec: REQ-VERIFY-003
    pub fn verify(&self, x: &ArrayView1<Float>) -> VerificationResult {
        let reports = self.decompose(x);
        let total_energy: Float = reports.iter().map(|r| r.weighted_energy).sum();
        let failing: Vec<String> = reports
            .iter()
            .filter(|r| !r.satisfied)
            .map(|r| r.name.clone())
            .collect();

        let verdict = if failing.is_empty() {
            Verdict::Verified
        } else {
            Verdict::Violated { failing }
        };

        VerificationResult {
            total_energy,
            constraints: reports,
            verdict,
        }
    }

    /// Compute gradient from only the violated constraints.
    ///
    /// This is the key primitive for gradient-based repair: descend only
    /// on the violated constraint energy, leaving satisfied regions undisturbed.
    ///
    /// Spec: REQ-VERIFY-005
    pub fn grad_violated_only(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        let mut grad = Array1::zeros(x.len());
        for (term, weight) in &self.terms {
            if !term.is_satisfied(x) {
                grad = grad + *weight * term.grad_energy(x);
            }
        }
        grad
    }
}

impl EnergyFunction for ComposedEnergy {
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        self.terms
            .iter()
            .map(|(term, weight)| weight * term.energy(x))
            .sum()
    }

    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        let mut grad = Array1::zeros(x.len());
        for (term, weight) in &self.terms {
            grad = grad + *weight * term.grad_energy(x);
        }
        grad
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }
}

/// Gradient-based repair: descend on violated constraints to fix a configuration.
///
/// Returns the repaired configuration and the verification history at each step.
///
/// Key property: only violated constraints contribute gradient, so satisfied
/// constraints are not disturbed.
///
/// Spec: REQ-VERIFY-005
pub fn repair(
    composed: &ComposedEnergy,
    x: &Array1<Float>,
    step_size: Float,
    max_steps: usize,
) -> (Array1<Float>, Vec<VerificationResult>) {
    let mut x = x.clone();
    let mut history = Vec::new();

    for _ in 0..max_steps {
        let result = composed.verify(&x.view());
        let done = result.is_verified();
        history.push(result);
        if done {
            break;
        }

        let grad = composed.grad_violated_only(&x.view());
        x = &x - step_size * &grad;
    }

    (x, history)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Constraint: x[0] should be >= 1.0
    /// Energy = max(0, 1.0 - x[0])^2
    struct MinValueConstraint {
        index: usize,
        min_val: Float,
    }

    impl ConstraintTerm for MinValueConstraint {
        fn name(&self) -> &str {
            "min_value"
        }

        fn energy(&self, x: &ArrayView1<Float>) -> Float {
            let violation = (self.min_val - x[self.index]).max(0.0);
            violation * violation
        }

        fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
            let mut grad = Array1::zeros(x.len());
            let violation = self.min_val - x[self.index];
            if violation > 0.0 {
                grad[self.index] = -2.0 * violation;
            }
            grad
        }
    }

    /// Constraint: sum of x should equal target
    /// Energy = (sum(x) - target)^2
    struct SumConstraint {
        target: Float,
    }

    impl ConstraintTerm for SumConstraint {
        fn name(&self) -> &str {
            "sum_equals_target"
        }

        fn energy(&self, x: &ArrayView1<Float>) -> Float {
            let diff = x.sum() - self.target;
            diff * diff
        }

        fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
            let diff = x.sum() - self.target;
            Array1::from_elem(x.len(), 2.0 * diff)
        }
    }

    #[test]
    fn test_constraint_term_satisfied() {
        // REQ-VERIFY-001: satisfied constraint returns ~0 energy
        let c = MinValueConstraint {
            index: 0,
            min_val: 1.0,
        };
        let x = array![2.0, 0.0];
        assert!(c.energy(&x.view()) < 1e-6);
        assert!(c.is_satisfied(&x.view()));
    }

    #[test]
    fn test_constraint_term_violated() {
        // REQ-VERIFY-001: violated constraint returns > 0 energy
        let c = MinValueConstraint {
            index: 0,
            min_val: 1.0,
        };
        let x = array![0.5, 0.0];
        let e = c.energy(&x.view());
        assert!(e > 0.0);
        assert!(!c.is_satisfied(&x.view()));
        // Energy proportional to violation: (1.0 - 0.5)^2 = 0.25
        assert!((e - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_composed_energy_sum() {
        // REQ-VERIFY-004: composed energy is sum of weighted terms
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 1.0,
            }),
            1.0,
        );
        composed.add_constraint(
            Box::new(SumConstraint { target: 3.0 }),
            2.0,
        );

        let x = array![0.5, 1.0];
        let total = composed.energy(&x.view());

        // min_value: (1.0 - 0.5)^2 = 0.25, weight 1.0 -> 0.25
        // sum: (1.5 - 3.0)^2 = 2.25, weight 2.0 -> 4.5
        // total: 4.75
        assert!((total - 4.75).abs() < 1e-4);
    }

    #[test]
    fn test_decomposition_sums_to_total() {
        // SCENARIO-VERIFY-003: sum of decomposed energies = total
        let mut composed = ComposedEnergy::new(3);
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 1.0,
            }),
            1.0,
        );
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 1,
                min_val: 2.0,
            }),
            1.5,
        );
        composed.add_constraint(
            Box::new(SumConstraint { target: 6.0 }),
            1.0,
        );

        let x = array![0.5, 1.0, 2.0];
        let reports = composed.decompose(&x.view());
        let total = composed.energy(&x.view());

        let decomposed_sum: Float = reports.iter().map(|r| r.weighted_energy).sum();
        assert!(
            (total - decomposed_sum).abs() < 1e-6,
            "Decomposition sum {decomposed_sum} != total {total}"
        );
        // Each constraint is named
        assert!(reports.iter().all(|r| !r.name.is_empty()));
    }

    #[test]
    fn test_verification_all_satisfied() {
        // REQ-VERIFY-003: VERIFIED when all constraints satisfied
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 1.0,
            }),
            1.0,
        );
        composed.add_constraint(
            Box::new(SumConstraint { target: 5.0 }),
            1.0,
        );

        let x = array![2.0, 3.0]; // min satisfied (2>=1), sum satisfied (5==5)
        let result = composed.verify(&x.view());
        assert!(result.is_verified());
        assert!(result.failing_constraints().is_empty());
        assert!(result.total_energy < 1e-6);
    }

    #[test]
    fn test_verification_with_violations() {
        // REQ-VERIFY-003: VIOLATED with specific failing constraint names
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 5.0,
            }),
            1.0,
        );
        composed.add_constraint(
            Box::new(SumConstraint { target: 10.0 }),
            1.0,
        );

        let x = array![1.0, 2.0]; // min violated (1<5), sum violated (3!=10)
        let result = composed.verify(&x.view());
        assert!(!result.is_verified());
        let failing = result.failing_constraints();
        assert_eq!(failing.len(), 2);
        assert!(failing.contains(&"min_value"));
        assert!(failing.contains(&"sum_equals_target"));
    }

    #[test]
    fn test_composition_add_constraints() {
        // REQ-VERIFY-004: constraints can be composed independently
        let mut composed = ComposedEnergy::new(2);
        assert_eq!(composed.num_constraints(), 0);

        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 0.0,
            }),
            1.0,
        );
        assert_eq!(composed.num_constraints(), 1);

        composed.add_constraint(
            Box::new(SumConstraint { target: 0.0 }),
            2.0,
        );
        assert_eq!(composed.num_constraints(), 2);
    }

    #[test]
    fn test_composed_as_energy_function() {
        // REQ-VERIFY-004: ComposedEnergy implements EnergyFunction
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(
            Box::new(SumConstraint { target: 0.0 }),
            1.0,
        );

        // Use through the EnergyFunction trait
        let ef: &dyn EnergyFunction = &composed;
        let x = array![1.0, -1.0]; // sum = 0, satisfied
        let e = ef.energy(&x.view());
        assert!(e < 1e-6);
        assert_eq!(ef.input_dim(), 2);
    }

    #[test]
    fn test_grad_violated_only() {
        // REQ-VERIFY-005: gradient from only violated constraints
        let mut composed = ComposedEnergy::new(2);
        // Constraint 1: x[0] >= 5.0 (will be violated)
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 5.0,
            }),
            1.0,
        );
        // Constraint 2: sum = 10.0 (will also be violated at x=[1,2], sum=3)
        // But we use x=[1,9] so sum=10 is satisfied while min_value is violated
        composed.add_constraint(
            Box::new(SumConstraint { target: 10.0 }),
            1.0,
        );

        // x=[1,9]: min_value violated (1<5), sum satisfied (1+9=10)
        let x = array![1.0, 9.0];

        // Sum constraint is satisfied, so its gradient contribution is non-zero
        // in the total but excluded from violated-only
        let grad_violated = composed.grad_violated_only(&x.view());
        let grad_all = composed.grad_energy(&x.view());

        // Total gradient includes sum constraint gradient (which is 0 here since
        // energy=0 means grad=0 too for sum constraint). Actually for a satisfied
        // quadratic constraint at exactly the target, grad IS 0. So let's verify
        // the violated-only gradient is correct on its own.

        // Violated-only gradient should only come from min_value constraint
        // min_value grad at x[0]=1, min=5: -2*(5-1) = -8 at index 0, 0 at index 1
        assert!(
            (grad_violated[0] - (-8.0)).abs() < 1e-4,
            "Expected grad[0]=-8.0, got {}", grad_violated[0]
        );
        assert!(
            grad_violated[1].abs() < 1e-6,
            "Expected grad[1]=0, got {}", grad_violated[1]
        );

        // Now test a case where both are violated so the gradient includes both
        let x2 = array![1.0, 2.0]; // min violated (1<5), sum violated (3!=10)
        let grad_both = composed.grad_violated_only(&x2.view());
        // min_value grad: [-8, 0], sum grad: [2*(3-10), 2*(3-10)] = [-14, -14]
        // total violated: [-22, -14]
        assert!(
            (grad_both[0] - (-22.0)).abs() < 1e-3,
            "Expected grad[0]=-22, got {}", grad_both[0]
        );
        assert!(
            (grad_both[1] - (-14.0)).abs() < 1e-3,
            "Expected grad[1]=-14, got {}", grad_both[1]
        );
    }

    #[test]
    fn test_repair_fixes_violations() {
        // REQ-VERIFY-005, SCENARIO-VERIFY-002: repair reduces violations
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 3.0,
            }),
            1.0,
        );
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 1,
                min_val: 3.0,
            }),
            1.0,
        );

        let x = array![0.0, 0.0]; // both violated
        let (repaired, history) = repair(&composed, &x, 0.1, 200);

        // Energy should decrease
        let initial_energy = history.first().unwrap().total_energy;
        let final_energy = history.last().unwrap().total_energy;
        assert!(
            final_energy < initial_energy,
            "Repair should reduce energy: {initial_energy} -> {final_energy}"
        );

        // Repaired values should be closer to the minimum
        assert!(
            repaired[0] > 0.0 && repaired[1] > 0.0,
            "Repair should push values toward min: {:?}",
            repaired
        );
    }

    #[test]
    fn test_repair_preserves_satisfied() {
        // SCENARIO-VERIFY-002: previously satisfied constraints remain satisfied
        let mut composed = ComposedEnergy::new(2);
        // x[0] >= 1.0 (will be satisfied from the start)
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 0,
                min_val: 1.0,
            }),
            1.0,
        );
        // x[1] >= 5.0 (will be violated)
        composed.add_constraint(
            Box::new(MinValueConstraint {
                index: 1,
                min_val: 5.0,
            }),
            1.0,
        );

        let x = array![10.0, 0.0]; // x[0] satisfied, x[1] violated
        let (repaired, _) = repair(&composed, &x, 0.1, 100);

        // x[0] should still be >= 1.0 (was satisfied, repair shouldn't touch it)
        // Because grad_violated_only only uses the violated x[1] constraint,
        // the gradient for x[0] is 0, so x[0] stays at 10.0
        assert!(
            repaired[0] >= 1.0,
            "Satisfied constraint should be preserved: x[0]={}", repaired[0]
        );
        // x[1] should have moved toward 5.0
        assert!(
            repaired[1] > 0.0,
            "Violated constraint should improve: x[1]={}", repaired[1]
        );
    }

    #[test]
    fn test_deterministic_reproducibility() {
        // SCENARIO-VERIFY-006: identical inputs produce identical results
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(
            Box::new(SumConstraint { target: 5.0 }),
            1.0,
        );

        let x = array![1.0, 2.0];
        let e1 = composed.energy(&x.view());
        let e2 = composed.energy(&x.view());
        let r1 = composed.verify(&x.view());
        let r2 = composed.verify(&x.view());

        assert_eq!(e1, e2, "Energy must be deterministic");
        assert_eq!(r1.total_energy, r2.total_energy, "Verification must be deterministic");
    }
}
