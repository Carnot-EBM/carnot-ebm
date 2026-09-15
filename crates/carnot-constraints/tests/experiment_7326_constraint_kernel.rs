//! Integration checks for REQ-VERIFY-7326 and SCENARIO-VERIFY-7326-*.

use carnot_constraints::{evaluate_schedule, ScheduleRequest};

fn parse(value: serde_json::Value) -> ScheduleRequest {
    serde_json::from_value(value).expect("the test request has the public schema")
}

// REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-ENERGY
#[test]
fn boundary_equality_and_overlapping_windows_have_exact_energy() {
    let request = parse(serde_json::json!({
        "schema": "carnot.constraint_kernel.request.v1",
        "executor_version": "executor-v1",
        "slot_min": 0,
        "slot_max": 3,
        "schedule": [
            {"activity": "a", "slot": 0},
            {"activity": "b", "slot": 2},
            {"activity": "c", "slot": 1},
            {"activity": "d", "slot": 2}
        ],
        "constraints": [
            {
                "kind": "pairwise_separation",
                "constraint_id": "sep-a-b",
                "version": "executor-v1",
                "left": "a",
                "right": "b",
                "minimum": 2
            },
            {
                "kind": "sliding_window_capacity",
                "constraint_id": "cap",
                "version": "executor-v1",
                "window_size": 2,
                "maximum": 2
            }
        ]
    }));

    let result = evaluate_schedule(&request);

    assert!(result.valid_input);
    assert!(!result.feasible);
    assert!(!result.complete_oracle_certificate);
    assert_eq!(result.total_energy, Some(1));
    assert_eq!(result.terms.len(), 2);
    assert_eq!(result.terms[0].energy, 0);
    assert_eq!(result.terms[1].energy, 1);
    assert_eq!(result.error, None);
}

// REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-INVALID
#[test]
fn empty_invalid_slot_version_and_overflow_fail_closed() {
    let base = serde_json::json!({
        "schema": "carnot.constraint_kernel.request.v1",
        "executor_version": "executor-v1",
        "slot_min": 0,
        "slot_max": 3,
        "schedule": [{"activity": "a", "slot": 0}, {"activity": "b", "slot": 2}],
        "constraints": [{
            "kind": "pairwise_separation",
            "constraint_id": "sep",
            "version": "executor-v1",
            "left": "a",
            "right": "b",
            "minimum": 2
        }]
    });

    let mut empty = base.clone();
    empty["schedule"] = serde_json::json!([]);
    assert_eq!(
        evaluate_schedule(&parse(empty)).error.as_deref(),
        Some("empty_schedule")
    );

    let mut invalid_slot = base.clone();
    invalid_slot["schedule"][0]["slot"] = serde_json::json!(9);
    assert_eq!(
        evaluate_schedule(&parse(invalid_slot)).error.as_deref(),
        Some("slot_out_of_domain:a")
    );

    let mut version = base.clone();
    version["constraints"][0]["version"] = serde_json::json!("executor-v2");
    assert_eq!(
        evaluate_schedule(&parse(version)).error.as_deref(),
        Some("version_mismatch:sep")
    );

    let overflow = parse(serde_json::json!({
        "schema": "carnot.constraint_kernel.request.v1",
        "executor_version": "executor-v1",
        "slot_min": i64::MAX,
        "slot_max": i64::MAX,
        "schedule": [{"activity": "a", "slot": i64::MAX}],
        "constraints": [{
            "kind": "sliding_window_capacity",
            "constraint_id": "cap",
            "version": "executor-v1",
            "window_size": 1,
            "maximum": 1
        }]
    }));
    assert_eq!(
        evaluate_schedule(&overflow).error.as_deref(),
        Some("window_end_overflow:cap")
    );
}

// REQ-VERIFY-7326 / SCENARIO-VERIFY-7326-ENERGY
#[test]
fn zero_acquired_energy_never_claims_complete_oracle_authority() {
    let request = parse(serde_json::json!({
        "schema": "carnot.constraint_kernel.request.v1",
        "executor_version": "executor-v1",
        "slot_min": 0,
        "slot_max": 1,
        "schedule": [{"activity": "a", "slot": 0}],
        "constraints": []
    }));

    let result = evaluate_schedule(&request);
    assert!(result.valid_input);
    assert!(result.feasible);
    assert_eq!(result.total_energy, Some(0));
    assert!(!result.complete_oracle_certificate);
}
