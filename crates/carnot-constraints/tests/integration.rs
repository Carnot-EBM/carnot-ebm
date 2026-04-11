//! Integration tests for carnot-constraints.
//!
//! These tests verify that the built-in constraint types work correctly
//! with ComposedEnergy for verification, decomposition, and repair.
//!
//! Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, REQ-VERIFY-004, REQ-VERIFY-005

use carnot_constraints::{
    BoundConstraint, ComposedEnergy, EqualityConstraint, IsingConstraint, VerificationCertificate,
};
use carnot_core::verify::repair;
use carnot_core::EnergyFunction;
use carnot_ising::{IsingConfig, IsingModel};
use ndarray::array;

/// SCENARIO-VERIFY-001: Compose multiple built-in constraints and verify
/// a fully satisfied configuration.
#[test]
fn test_composed_all_satisfied() {
    let mut composed = ComposedEnergy::new(3);
    composed.add_constraint(
        Box::new(BoundConstraint::new("x0_bound", 0, 0.0, 10.0)),
        1.0,
    );
    composed.add_constraint(
        Box::new(BoundConstraint::new("x1_bound", 1, -5.0, 5.0)),
        1.0,
    );
    composed.add_constraint(
        Box::new(EqualityConstraint::new("x2_target", 2, 3.0, 0.1)),
        2.0,
    );

    let x = array![5.0, 0.0, 3.0];
    let result = composed.verify(&x.view());
    assert!(result.is_verified());
    assert!(result.total_energy < 1e-6);
    assert!(result.failing_constraints().is_empty());
}

/// SCENARIO-VERIFY-001: Compose multiple built-in constraints and verify
/// a configuration with violations.
#[test]
fn test_composed_with_violations() {
    let mut composed = ComposedEnergy::new(3);
    composed.add_constraint(
        Box::new(BoundConstraint::new("x0_bound", 0, 0.0, 10.0)),
        1.0,
    );
    composed.add_constraint(
        Box::new(BoundConstraint::new("x1_bound", 1, -5.0, 5.0)),
        1.0,
    );
    composed.add_constraint(
        Box::new(EqualityConstraint::new("x2_target", 2, 3.0, 0.01)),
        2.0,
    );

    // x0 = 15 (violated: above 10), x1 = 0 (ok), x2 = 7 (violated: far from 3)
    let x = array![15.0, 0.0, 7.0];
    let result = composed.verify(&x.view());
    assert!(!result.is_verified());

    let failing = result.failing_constraints();
    assert!(failing.contains(&"x0_bound"));
    assert!(failing.contains(&"x2_target"));
    assert!(!failing.contains(&"x1_bound"));
}

/// SCENARIO-VERIFY-003: Decomposition sums to total energy.
#[test]
fn test_decomposition_consistency() {
    let mut composed = ComposedEnergy::new(2);
    composed.add_constraint(Box::new(BoundConstraint::new("x0_bound", 0, 2.0, 8.0)), 1.5);
    composed.add_constraint(
        Box::new(EqualityConstraint::new("x1_eq", 1, 4.0, 0.01)),
        2.0,
    );

    let x = array![0.0, 6.0]; // x0 violated (below 2), x1 violated (6 != 4)
    let reports = composed.decompose(&x.view());
    let total = composed.energy(&x.view());
    let decomposed_sum: carnot_core::Float = reports.iter().map(|r| r.weighted_energy).sum();

    assert!(
        (total - decomposed_sum).abs() < 1e-6,
        "Decomposition sum {decomposed_sum} != total {total}"
    );
}

/// SCENARIO-VERIFY-002: Repair reduces violations using built-in constraints.
#[test]
fn test_repair_with_bound_constraints() {
    let mut composed = ComposedEnergy::new(2);
    composed.add_constraint(Box::new(BoundConstraint::new("x0_bound", 0, 3.0, 7.0)), 1.0);
    composed.add_constraint(Box::new(BoundConstraint::new("x1_bound", 1, 3.0, 7.0)), 1.0);

    let x = array![0.0, 10.0]; // x0 below, x1 above
    let (repaired, history) = repair(&composed, &x, 0.1, 200);

    let initial_energy = history.first().unwrap().total_energy;
    let final_energy = history.last().unwrap().total_energy;
    assert!(
        final_energy < initial_energy,
        "Repair should reduce energy: {initial_energy} -> {final_energy}"
    );
    // Values should move toward the valid range
    assert!(repaired[0] > 0.0, "x0 should increase toward [3,7]");
    assert!(repaired[1] < 10.0, "x1 should decrease toward [3,7]");
}

/// SCENARIO-VERIFY-002: Repair with equality constraints.
#[test]
fn test_repair_with_equality_constraints() {
    let mut composed = ComposedEnergy::new(2);
    composed.add_constraint(Box::new(EqualityConstraint::new("x0_eq", 0, 5.0, 0.1)), 1.0);
    composed.add_constraint(
        Box::new(EqualityConstraint::new("x1_eq", 1, -2.0, 0.1)),
        1.0,
    );

    let x = array![0.0, 0.0]; // both far from target
    let (repaired, history) = repair(&composed, &x, 0.1, 500);

    let final_energy = history.last().unwrap().total_energy;
    assert!(
        final_energy < 1.0,
        "Repair should converge: final_energy={final_energy}"
    );
    // Values should approach targets
    assert!(
        (repaired[0] - 5.0).abs() < 1.0,
        "x0 should approach 5.0, got {}",
        repaired[0]
    );
    assert!(
        (repaired[1] - (-2.0)).abs() < 1.0,
        "x1 should approach -2.0, got {}",
        repaired[1]
    );
}

/// SCENARIO-VERIFY-004: IsingConstraint integrates with ComposedEnergy.
#[test]
fn test_ising_in_composed_energy() {
    let model = IsingModel::new(IsingConfig {
        input_dim: 3,
        hidden_dim: None,
        coupling_init: "zeros".to_string(),
    })
    .unwrap();

    let mut composed = ComposedEnergy::new(3);
    composed.add_constraint(
        Box::new(IsingConstraint::new("ising_logic", model, 0.1)),
        1.0,
    );
    composed.add_constraint(
        Box::new(BoundConstraint::new("x0_bound", 0, -1.0, 1.0)),
        1.0,
    );

    let x = array![0.5, -0.5, 0.3];
    let result = composed.verify(&x.view());
    // With zero couplings, Ising energy is 0 which is <= 0.1 threshold -> satisfied
    // x0 = 0.5 is in [-1, 1] -> satisfied
    assert!(result.is_verified());
}

/// REQ-VERIFY-003: VerificationCertificate generation and JSON serialization.
#[test]
fn test_certificate_full_workflow() {
    let mut composed = ComposedEnergy::new(3);
    composed.add_constraint(
        Box::new(BoundConstraint::new("temp_range", 0, 20.0, 30.0)),
        1.0,
    );
    composed.add_constraint(
        Box::new(EqualityConstraint::new("pressure_target", 1, 101.3, 0.5)),
        2.0,
    );
    composed.add_constraint(
        Box::new(BoundConstraint::new("humidity_range", 2, 30.0, 70.0)),
        1.0,
    );

    // All satisfied
    let x = array![25.0, 101.3, 50.0];
    let cert = VerificationCertificate::generate(&composed, &x.view(), "2026-04-09T19:00:00Z");
    assert!(cert.is_verified());
    assert_eq!(cert.num_constraints, 3);
    assert_eq!(cert.num_satisfied, 3);

    // Serialize and deserialize
    let json = cert.to_json().unwrap();
    assert!(json.contains("Verified"));
    assert!(json.contains("temp_range"));
    let restored = VerificationCertificate::from_json(&json).unwrap();
    assert!(restored.is_verified());

    // Now with a violation
    let x_bad = array![25.0, 200.0, 50.0]; // pressure way off
    let cert_bad =
        VerificationCertificate::generate(&composed, &x_bad.view(), "2026-04-09T19:01:00Z");
    assert!(!cert_bad.is_verified());
    assert_eq!(cert_bad.num_violated, 1);
    assert!(cert_bad
        .failing_constraints
        .contains(&"pressure_target".to_string()));
}

/// SCENARIO-VERIFY-006: Deterministic reproducibility.
#[test]
fn test_deterministic_results() {
    let mut composed = ComposedEnergy::new(2);
    composed.add_constraint(Box::new(BoundConstraint::new("x0_bound", 0, 0.0, 5.0)), 1.0);
    composed.add_constraint(
        Box::new(EqualityConstraint::new("x1_eq", 1, 3.0, 0.01)),
        1.0,
    );

    let x = array![7.0, 1.0];
    let r1 = composed.verify(&x.view());
    let r2 = composed.verify(&x.view());

    assert_eq!(r1.total_energy, r2.total_energy);
    assert_eq!(r1.constraints.len(), r2.constraints.len());
    for (a, b) in r1.constraints.iter().zip(r2.constraints.iter()) {
        assert_eq!(a.energy, b.energy);
        assert_eq!(a.satisfied, b.satisfied);
    }
}
