"""Tests for REQ-VERIFY-7477 and SCENARIO-VERIFY-7477-*.

The tests use controlled readout rows. They never load the current model.
"""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

from carnot import experiment_7477_v655_native_readout_pilot as exp


def test_req_verify_7477_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7477 and each required scenario exist in the capability spec."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "REQ-VERIFY-7477" in text
    for suffix in ("IDENTITY", "READOUT", "CONTROLS", "NULL", "COST", "E2E"):
        assert f"SCENARIO-VERIFY-7477-{suffix}" in text


def test_scenario_verify_7477_readout_freezes_twelve_disjoint_groups() -> None:
    """SCENARIO-VERIFY-7477-READOUT freezes identities before model outcomes."""

    first = exp.freeze_development_groups()
    second = exp.freeze_development_groups()
    sealed = {
        json.loads(line)["group_id"]
        for line in (exp.REPO_ROOT / exp.V654_GROUPS_PATH).read_text(encoding="utf-8").splitlines()
    }
    assert first == second
    assert first is not second
    assert len(first) == len({row["group_id"] for row in first}) == 12
    assert not ({row["group_id"] for row in first} & sealed)
    assert all(row["group_hash"] == exp.canonical_hash(row["prompt_inputs"]) for row in first)
    first[0]["source"] = "changed"
    assert exp.freeze_development_groups()[0]["source"] != "changed"


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    (
        ("label_swap", "label_mapping_valid"),
        ("token_position", "token_position_valid"),
        ("constant_vector", "constant_stub_absent"),
        ("state_reuse", "fresh_state_valid"),
    ),
)
def test_scenario_verify_7477_controls_reject_known_mutations(
    mutation: str, failed_check: str
) -> None:
    """SCENARIO-VERIFY-7477-CONTROLS rejects each named transport defect."""

    rows = exp.controlled_transport_fixture()
    assert exp.reduce_transport_rows(rows, expected_forwards=4)["passed"] is True
    altered = exp.mutate_transport_fixture(rows, mutation)
    reduction = exp.reduce_transport_rows(altered, expected_forwards=4)
    assert reduction["passed"] is False
    assert reduction[failed_check] is False


def test_scenario_verify_7477_null_does_not_gate_on_predictions() -> None:
    """SCENARIO-VERIFY-7477-NULL accepts natural ties and order disagreement."""

    rows = exp.controlled_transport_fixture()
    rows[0]["raw_logits_by_option_id"] = {
        "supported": 2.0,
        "contains_unsupported": 2.0,
    }
    rows[0]["prediction_correct"] = False
    rows[1]["predicted_option_id"] = "contains_unsupported"
    reduction = exp.reduce_transport_rows(rows, expected_forwards=4)
    assert reduction["passed"] is True
    assert reduction["uniform_rows"] == 1
    assert reduction["predictive_accuracy_is_gate"] is False
    assert reduction["real_order_agreement_is_gate"] is False


def test_scenario_verify_7477_cost_preserves_both_capture_caps() -> None:
    """SCENARIO-VERIFY-7477-COST keeps the fixed rosters and 3,300-second caps."""

    feasible = exp.project_capture_costs([5.0, 4.0, 6.0], load_s=8.0)
    assert [row["planned_forwards"] for row in feasible] == [520, 628]
    assert [row["group_roster"] for row in feasible] == [
        {"training": 180, "calibration_tuning": 60},
        {"internal_test": 60, "online": 160, "external": 74},
    ]
    assert all(row["hard_cap_s"] == 3300.0 for row in feasible)
    assert all(row["feasible"] is True for row in feasible)

    short = exp.project_capture_costs([10.0], load_s=100.0)
    assert short[0]["feasible"] is False
    assert short[0]["unstarted_forwards"] == 200
    assert short[1]["unstarted_forwards"] == 308


def test_invocation_accounting_balances_without_generation() -> None:
    """REQ-VERIFY-7477 balances every attempted native operation."""

    events = exp.fixture_invocation_events(forwards=4)
    counts = exp.reduce_invocation_events(events)
    assert counts["model_loads"]["attempted"] == counts["model_loads"]["completed"] == 1
    assert counts["forward_calls"]["attempted"] == counts["forward_calls"]["completed"] == 4
    assert counts["generation_calls"]["attempted"] == 0
    assert counts["generation_calls"]["in_flight"] == 0
    assert exp.invocation_counts_balanced(counts) is True
    incomplete = deepcopy(counts)
    incomplete["forward_calls"]["in_flight"] = 1
    assert exp.invocation_counts_balanced(incomplete) is False
    assert exp.invocation_counts_balanced({}) is False


def test_acceptance_gates_keep_validity_readiness_and_benefit_separate() -> None:
    """REQ-VERIFY-7477 gives every gate a failure-prevention principle."""

    gates = exp.build_acceptance_gates(
        transport_passed=True,
        validation_passed=True,
        benefit_measured=False,
    )
    assert {row["category"] for row in gates} == {
        "required_validity",
        "readiness",
        "scientific_benefit",
    }
    assert all(row["passed"] is True and row["principle"] for row in gates)
    failed = exp.build_acceptance_gates(
        transport_passed=False,
        validation_passed=True,
        benefit_measured=False,
    )
    assert exp.gate_check_summary(failed)["failed_check"] == "authenticated_transport"


def test_artifact_fixture_cold_reduces_and_detects_drift() -> None:
    """SCENARIO-VERIFY-7477-E2E recomputes rows, readiness, and checksum."""

    artifact = exp.build_artifact_for_test()
    assert exp.validate_artifact(artifact, require_validation=False) == []
    assert artifact["native_readout_ready_score"] == 1
    assert artifact["scored_runtime_parity_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null")
    assert artifact["MODEL_SPECS"] == [exp.MODEL_HF_ID]
    assert artifact["model_specs"] == artifact["MODEL_SPECS"]
    assert artifact["inference_substrate_class"] == "model_load_no_generation"
    assert artifact["field_principles"].keys() >= exp.REQUIRED_PRINCIPLE_FIELDS

    drifted = deepcopy(artifact)
    drifted["rows"][0]["actual_last_evaluated_position"] += 1
    assert "transport_reduction_mismatch" in exp.validate_artifact(
        drifted, require_validation=False
    )


def test_validation_manifest_is_scoped_and_frozen() -> None:
    """SCENARIO-VERIFY-7477-E2E excludes every repository-wide test target."""

    manifest = exp.VALIDATION_MANIFEST
    assert manifest.test_paths == (exp.TEST_PATH.as_posix(),)
    assert manifest.changed_modules == (exp.MODULE_PATH.as_posix(),)
    assert manifest.static_paths == (exp.WRAPPER_PATH.as_posix(),)
    assert all(path not in {"tests", "tests/python"} for path in manifest.test_paths)


def test_defensive_reducers_reject_unknown_or_malformed_inputs() -> None:
    """REQ-VERIFY-7477 fails closed on malformed fixtures and cost inputs."""

    with pytest.raises(ValueError, match="unknown_mutation"):
        exp.mutate_transport_fixture(exp.controlled_transport_fixture(), "unknown")
    with pytest.raises(ValueError, match="positive_finite_prefill"):
        exp.project_capture_costs([], load_s=1.0)
    malformed = exp.controlled_transport_fixture()
    malformed[0]["raw_logits_by_option_id"] = {"supported": 1.0}
    assert exp.reduce_transport_rows(malformed, expected_forwards=4)["finite_logits_valid"] is False
    assert exp._validation_names_passed({}, ("one",)) is False
    assert exp._validation_names_passed([{"name": "one", "passed": True}], ("one",)) is True
    assert exp.validate_artifact([], require_validation=False) == ["artifact_not_object"]


def test_cold_validator_names_each_corrupted_artifact_boundary() -> None:
    """SCENARIO-VERIFY-7477-E2E reports the exact invalid evidence boundary."""

    cases: list[tuple[str, object]] = []

    wrong_identity = exp.build_artifact_for_test()
    wrong_identity["schema"] = "wrong"
    cases.append(("schema_mismatch", wrong_identity))

    missing_rows = exp.build_artifact_for_test()
    missing_rows["rows"] = None
    cases.append(("rows_invalid", missing_rows))

    missing_events = exp.build_artifact_for_test()
    missing_events["current_invocation_events"] = None
    cases.append(("invocation_evidence_invalid", missing_events))

    wrong_counts = exp.build_artifact_for_test()
    wrong_counts["invocation_counts"]["forward_calls"]["completed"] = 3
    cases.append(("invocation_counts_mismatch", wrong_counts))

    unfinished = exp.build_artifact_for_test()
    unfinished["current_invocation_events"].pop()
    unfinished["invocation_counts"] = exp.reduce_invocation_events(
        unfinished["current_invocation_events"]
    )
    cases.append(("invocation_counts_unbalanced", unfinished))

    no_principles = exp.build_artifact_for_test()
    no_principles["field_principles"] = {}
    cases.append(("field_principles_missing", no_principles))

    bad_gates = exp.build_artifact_for_test()
    bad_gates["acceptance_gate_results"] = None
    cases.append(("acceptance_gates_invalid", bad_gates))

    bad_summary = exp.build_artifact_for_test()
    bad_summary["gate_check_summary"] = {"passed": False}
    cases.append(("gate_summary_mismatch", bad_summary))

    bad_ready = exp.build_artifact_for_test()
    bad_ready["native_readout_ready_score"] = 0
    cases.append(("native_readout_ready_score_mismatch", bad_ready))

    for expected, artifact in cases:
        assert expected in exp.validate_artifact(artifact, require_validation=False)

    no_receipts = exp.build_artifact_for_test()
    assert "required_validation_failed" in exp.validate_artifact(
        no_receipts, require_validation=True
    )
