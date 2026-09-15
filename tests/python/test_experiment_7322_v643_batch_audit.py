"""Tests for the independent V643 batch audit.

Spec refs: REQ-VERIFY-7322 and SCENARIO-VERIFY-7322-*.
"""

from __future__ import annotations

import base64
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7322_v643_batch_audit as audit
from carnot.reporting import experiment_7303_validation_scope as scoped


ROOT = Path(__file__).resolve().parents[2]


def _bundle() -> audit.JsonDict:
    """Load the sealed current measurement once for a real raw-evidence test."""

    return audit.load_bundle(ROOT)


def _passing_validation() -> audit.JsonDict:
    """Make named receipts that stand in only for artifact unit tests."""

    names = (*scoped.REQUIRED_CHECK_NAMES, *audit.TERMINAL_CHECK_NAMES)
    receipts = [
        {
            "name": name,
            "command": f"fixture:{name}",
            "command_argv": [name],
            "scope": "unit_test_only",
            "exit_code": 0,
            "duration_s": 0.001,
            "log_path": f"/tmp/{name}.log",
            "log_sha256": "sha256:" + "a" * 64,
            "passed": True,
            "timed_out": False,
        }
        for name in names
    ]
    return {
        "validation_receipts": receipts,
        "required_checks_passed": True,
        "missing_required_commands": [],
        "failed_required_commands": [],
        "duplicate_required_commands": [],
        "repository_health": scoped.build_repository_health([]),
        "validation_entrypoint_receipt": {
            "runner": audit.SCOPED_RUNNER,
            "called": True,
            "test_paths": [audit.TEST_PATH.as_posix()],
            "changed_modules": [audit.MODULE_PATH.as_posix()],
            "static_paths": [audit.WRAPPER_PATH.as_posix()],
            "legacy_launcher_called": False,
            "repository_wide_target_present": False,
        },
    }


def test_scenario_verify_7322_dependency_admits_complete_null() -> None:
    """SCENARIO-VERIFY-7322-DEPENDENCY audits a complete null capture."""

    measurement = json.loads(audit.MEASUREMENT_PATH.read_text(encoding="utf-8"))
    checks = audit.dependency_gate_rows(measurement)
    assert all(row["passed"] for row in checks)
    assert measurement["verdict_class"] == "null"
    assert measurement["batch_value_score"] == 0

    changed = deepcopy(measurement)
    changed["verdict_class"] = "disqualified"
    summary = audit.gate_check_summary(audit.dependency_gate_rows(changed))
    assert summary == {
        "failed_check": "exp7321_terminal_class",
        "upstream": "exp7321-batch-measurement",
        "field": "verdict_class",
        "expected_value": "not_in:['blocked', 'disqualified', 'partial']",
        "observed_value": "disqualified",
    }
    changed["verdict_class"] = "null"
    changed["quarantined"] = True
    assert audit.gate_check_summary(audit.dependency_gate_rows(changed))["failed_check"] == (
        "exp7321_quarantine"
    )
    assert audit.gate_check_summary(audit.dependency_gate_rows(None))["observed_value"] == (
        "missing_artifact"
    )
    assert audit.gate_check_summary([])["failed_check"] is None


def test_scenario_verify_7322_raw_authentication_and_independent_reduction() -> None:
    """SCENARIO-VERIFY-7322-RAW rebuilds all rows from native response bytes."""

    bundle = _bundle()
    checks = audit.authenticate_bundle(bundle)
    assert all(row["passed"] for row in checks), audit.gate_check_summary(checks)

    reduced = audit.independent_reduce(bundle, draws=10_000, seed=audit.BOOTSTRAP_SEED)
    rows = reduced["rows"]
    assert len(rows) == 384
    assert {row["arm"] for row in rows} == set(audit.ARMS)
    assert all(sum(row["arm"] == arm for row in rows) == 128 for arm in audit.ARMS)
    assert reduced["denominators"] == {
        "planned_rows": 384,
        "attempted_rows": 384,
        "complete_rows": 384,
        "censored_rows": 0,
        "planned_units_per_arm": 128,
        "group_count": 16,
    }
    comparisons = {row["comparison"]: row for row in reduced["independent_comparison_rows"]}
    assert comparisons["batched_verifier_vs_joint_direct"]["accuracy_difference"] == -0.171875
    assert comparisons["batched_verifier_vs_joint_direct"]["coverage_difference"] == -0.1875
    assert comparisons["serial_vs_batched_verifier"]["prediction_discrepancies"] == 0
    assert (
        reduced["paired_intervals"]["metrics"]["accuracy_difference_vs_direct"][
            "one_sided_95_lower"
        ]
        == -0.296875
    )
    assert reduced["zero_error_population_risk_claimed"] is False


def test_scenario_verify_7322_cost_uses_disjoint_spans_and_equal_cache() -> None:
    """SCENARIO-VERIFY-7322-COST charges each measured span exactly once."""

    bundle = _bundle()
    reduced = audit.independent_reduce(bundle, draws=20, seed=audit.BOOTSTRAP_SEED)
    cost = reduced["cost_summary"]
    assert cost["shared_initialization_count"] == 1
    assert cost["shared_initialization_counted_in_warm_cost"] is False
    assert (
        cost["cold_total_wall_s"] == cost["shared_initialization_s"] + cost["warm_complete_cost_s"]
    )
    assert cost["disjoint_span_reconstruction_passed"] is True
    assert reduced["direct_native_cache_equal_opportunity"] is True
    assert reduced["paired_intervals"]["bootstrap_unit"] == "source_group"


def test_scenario_verify_7322_interference_separates_live_and_cpu_evidence() -> None:
    """SCENARIO-VERIFY-7322-INTERFERENCE does not invent model outputs."""

    bundle = _bundle()
    reduced = audit.independent_reduce(bundle, draws=20, seed=audit.BOOTSTRAP_SEED)
    controls = audit.interference_controls(bundle, reduced)
    assert all(row["passed"] for row in controls)
    assert {
        row["control"] for row in controls if row["evidence_kind"] == "live_retained_output"
    } == {"sealed_arm_order_effect"}
    assert {
        row["control"] for row in controls if row["evidence_kind"] == "cpu_parser_executor"
    } == {
        "mutate_one_claim",
        "duplicate_claim_identifier",
        "swap_source_version",
        "unsupported_neighbor_instruction",
    }
    assert all(row["counterfactual_model_output_invented"] is False for row in controls)


def test_scenario_verify_7322_attacks_all_fail_closed() -> None:
    """SCENARIO-VERIFY-7322-ATTACKS rejects every named altered bundle."""

    attacks = audit.adversarial_attacks(_bundle())
    assert [row["attack"] for row in attacks] == [
        "missing_slow_row",
        "relabeled_abstention",
        "excluded_malformed_batch",
        "wrong_group_denominator",
        "forged_source_version",
        "adjusted_cost",
    ]
    assert all(row["attack_rejected"] and row["passed"] for row in attacks)
    assert all(row["failed_check"] for row in attacks)


def test_scenario_verify_7322_terminal_complete_null_and_blocked() -> None:
    """SCENARIO-VERIFY-7322-TERMINAL separates audit work from promotion."""

    bundle = _bundle()
    checks = audit.authenticate_bundle(bundle)
    reduced = audit.independent_reduce(bundle, draws=100, seed=audit.BOOTSTRAP_SEED)
    controls = audit.interference_controls(bundle, reduced)
    attacks = audit.adversarial_attacks(bundle)
    artifact = audit.assemble_artifact(
        "20260915", bundle, checks, reduced, controls, attacks, _passing_validation()
    )
    assert audit.validate_artifact(artifact) == []
    assert artifact["batch_audit_complete_score"] == 1
    assert artifact["batch_promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null")
    assert artifact["verifier_is_oracle"] is True
    assert artifact["retirement_decision"]["stop_same_mechanism"] is True
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert set(artifact["field_principles"]) >= set(audit.REQUIRED_FIELDS)

    failed_validation = _passing_validation()
    failed_validation["required_checks_passed"] = False
    failed_validation["validation_receipts"][0]["passed"] = False
    disqualified = audit.assemble_artifact(
        "20260915", bundle, checks, reduced, controls, attacks, failed_validation
    )
    assert disqualified["batch_audit_complete_score"] == 0
    assert disqualified["batch_promotion_score"] == 0
    assert disqualified["verdict_class"] == "disqualified"

    blocked = audit.blocked_artifact("20260915", audit.dependency_gate_rows(None), 0.01)
    assert audit.validate_artifact(blocked, require_validation=False) == []
    assert blocked["verdict_class"] == "blocked"
    assert blocked["batch_audit_complete_score"] == 0
    assert blocked["gate_check_summary"]["field"] == "artifact"


def test_scenario_verify_7322_e2e_scope_and_cold_validation() -> None:
    """SCENARIO-VERIFY-7322-E2E fixes explicit paths and validates checksums."""

    commands = audit.validation_commands(ROOT)
    assert [row.name for row in commands] == list(scoped.REQUIRED_CHECK_NAMES)
    assert all("tests/python" not in row.argv for row in commands)
    assert all(audit.TEST_PATH.as_posix() in row.argv for row in commands if "pytest" in row.name)

    bundle = _bundle()
    artifact = audit.assemble_artifact(
        "20260915",
        bundle,
        audit.authenticate_bundle(bundle),
        audit.independent_reduce(bundle, draws=50, seed=audit.BOOTSTRAP_SEED),
        audit.interference_controls(
            bundle, audit.independent_reduce(bundle, draws=10, seed=audit.BOOTSTRAP_SEED)
        ),
        audit.adversarial_attacks(bundle),
        _passing_validation(),
    )
    changed = deepcopy(artifact)
    changed["rows"][0]["prediction"] = "forged"
    assert "reproducibility_checksum" in audit.validate_artifact(changed)
    missing = deepcopy(artifact)
    missing.pop("retirement_decision")
    assert "required_fields" in audit.validate_artifact(missing)


def test_scenario_verify_7322_defensive_input_paths_fail_closed(tmp_path: Path) -> None:
    """REQ-VERIFY-7322 rejects malformed rosters, bytes, pointers, and batches."""

    not_mapping = tmp_path / "list.json"
    not_mapping.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping required"):
        audit._read_json(not_mapping)

    bundle = _bundle()
    public = bundle["public"]
    bad_groups = deepcopy(public)
    bad_groups["evaluation_groups"][0]["source_versions"] = []
    assert any(
        error.startswith("source_version_count") for error in audit._public_roster(bad_groups)[1]
    )
    bad_claims = deepcopy(public)
    bad_claims["evaluation_groups"][0]["claims"] = []
    assert any(error.startswith("claim_count") for error in audit._public_roster(bad_claims)[1])
    bad_hashes = deepcopy(public)
    group = bad_hashes["evaluation_groups"][0]
    group["source_versions"][0]["source_hash"] = "sha256:forged"
    group["claims"][0]["claim_hash"] = "sha256:forged"
    group["claims"][1]["source_version"] = 9
    roster_errors = audit._public_roster(bad_hashes)[1]
    assert any(error.startswith("source_hash") for error in roster_errors)
    assert any(error.startswith("claim_hash") for error in roster_errors)
    assert any(error.startswith("claim_version_count") for error in roster_errors)
    assert "unit_identity_count" in roster_errors

    structural = {**bundle, "schedule": deepcopy(bundle["schedule"])}
    structural["schedule"][0]["call_order"] = 9
    structural["schedule"][0]["arm"] = "wrong"
    structural["schedule"] = structural["schedule"][:-1]
    structural["predictions"] = structural["predictions"][:-1]
    structural["scored"] = structural["scored"][:-1]
    structural["evaluator"] = {"labels": None}
    structural["public"] = deepcopy(bundle["public"])
    structural["public"]["authority_note"] = "expected_decision"
    structural_errors = audit._structural_errors(structural)
    assert {
        "schedule_call_count",
        "prediction_row_count",
        "scored_row_count",
        "call_order",
    }.issubset(structural_errors)
    assert "evaluator_label_identity" in structural_errors
    assert "public_authority_leakage" in structural_errors
    assert any(error.startswith("arm_call_count") for error in structural_errors)

    row = bundle["calls"][0]
    assert audit._native_payload({}) == (None, ["native_byte_decode"])
    altered = deepcopy(row)
    altered.update(
        {
            "request_payload_sha256": "sha256:bad",
            "raw_response_sha256": "sha256:bad",
            "request_payload": {},
            "raw_response": {},
            "raw_completion": "bad",
            "raw_completion_sha256": "sha256:bad",
        }
    )
    _payload, byte_errors = audit._native_payload(altered)
    assert {
        "request_byte_hash",
        "response_byte_hash",
        "request_byte_object",
        "response_byte_object",
        "native_completion_text",
        "native_completion_hash",
    }.issubset(byte_errors)
    no_choice = {"choices": []}
    no_choice_bytes = audit.canonical_json(no_choice).encode("utf-8")
    incomplete = deepcopy(row)
    incomplete["raw_response_bytes_b64"] = base64.b64encode(no_choice_bytes).decode("ascii")
    incomplete["raw_response_sha256"] = audit.sha256_bytes(no_choice_bytes)
    incomplete["raw_response"] = no_choice
    assert "native_completion_parse" in audit._native_payload(incomplete)[1]

    source = public["evaluation_groups"][0]["source_versions"][0]["document"]
    good_relation = {
        "subject_pointer": "m000",
        "predicate": "precedes",
        "object_pointer": "m001",
        "polarity": "positive",
    }
    assert audit._compile_relations(source, None, source=True)[1] == ["completion_shape"]
    assert audit._compile_relations(source, {"outcome": "unknown", "relations": []}, source=True)[
        1
    ] == ["explicit_unknown"]
    assert audit._compile_relations(source, {"outcome": "bad", "relations": []}, source=True)[
        1
    ] == ["completion_shape"]
    assert audit._compile_relations(source, {"outcome": "known", "relations": []}, source=True)[
        1
    ] == ["relation_count"]
    assert audit._compile_relations(
        {}, {"outcome": "known", "relations": [good_relation]}, source=True
    )[1] == ["document_shape"]
    assert audit._compile_relations(source, {"outcome": "known", "relations": [{}]}, source=True)[
        1
    ] == ["relation_shape"]
    missing_pointer = {**good_relation, "subject_pointer": "missing"}
    assert audit._compile_relations(
        source, {"outcome": "known", "relations": [missing_pointer]}, source=True
    )[1] == ["unresolved_relation"]
    bad_surface = deepcopy(source)
    bad_surface["mentions"][0]["surface_text"] = "wrong"
    assert audit._compile_relations(
        bad_surface, {"outcome": "known", "relations": [good_relation]}, source=True
    )[1] == ["pointer_surface"]
    bad_polarity = {**good_relation, "polarity": "maybe"}
    assert audit._compile_relations(
        source, {"outcome": "known", "relations": [bad_polarity]}, source=True
    )[1] == ["polarity"]
    assert audit._execute_relations(
        [("a", "precedes", "b", "positive")], [("a", "other", "b", "positive")]
    ) == ("unknown")
    assert audit._joint_values(None, ["u1"], "decision")["u1"][1] == ["malformed_batch"]
    assert audit._joint_values({"items": [{"claim_id": "u1"}]}, ["u1"], "decision")["u1"][1] == [
        "malformed_item"
    ]

    missing_call = {**bundle, "calls": bundle["calls"][1:]}
    replayed, _errors = audit._replay_rows(missing_call)
    assert any(row["censored"] for row in replayed)
    bad_measurement = {**bundle, "measurement": deepcopy(bundle["measurement"])}
    bad_measurement["measurement"]["honest_verdict"] = "forged"
    assert audit.authenticate_bundle(bad_measurement)[0]["passed"] is False


def test_scenario_verify_7322_schema_guards_and_circular_cap() -> None:
    """REQ-VERIFY-7322 covers every terminal schema guard and oracle cap."""

    bundle = _bundle()
    reduced = audit.independent_reduce(bundle, draws=20, seed=audit.BOOTSTRAP_SEED)
    controls = audit.interference_controls(bundle, reduced)
    attacks = audit.adversarial_attacks(bundle)
    checks = audit.authenticate_bundle(bundle)
    passing = _passing_validation()

    passing_values = deepcopy(reduced)
    for metric in passing_values["paired_intervals"]["metrics"].values():
        metric["one_sided_95_lower"] = 2.0
    circular = audit.assemble_artifact(
        "20260915", bundle, checks, passing_values, controls, attacks, passing
    )
    assert circular["verdict_class"] == "circular_positive"
    assert circular["batch_promotion_score"] == 1

    assert audit.validate_artifact(None) == ["artifact_mapping"]
    corrupt = deepcopy(circular)
    corrupt.update(
        {
            "schema": "wrong",
            "run_date": "wrong",
            "MODEL_SPECS": [{}],
            "invocation_counts": {},
            "inference_substrate": "wrong",
            "inference_substrate_class": "wrong",
            "execution_venue": "wrong",
            "verdict_class": "wrong",
            "rows": [],
            "batch_audit_complete_score": 0,
            "validation_receipts": [],
        }
    )
    errors = audit.validate_artifact(corrupt)
    assert {
        "identity",
        "date_or_milestone",
        "model_declaration",
        "invocation_counts",
        "inference_substrate",
        "inference_substrate_class",
        "execution_venue",
        "verdict_class",
        "row_count",
        "arm_denominators",
        "audit_complete_score",
        "validation_receipts",
    }.issubset(errors)
    failed_scores = deepcopy(circular)
    failed_scores["verdict_class"] = "blocked"
    assert "failed_scores" in audit.validate_artifact(failed_scores, require_validation=False)
    oracle_positive = deepcopy(circular)
    oracle_positive["verdict_class"] = "positive"
    assert "oracle_positive" in audit.validate_artifact(oracle_positive)
    assert audit._date_argument("20260915") == "20260915"
    with pytest.raises(Exception, match="date must be 20260915"):
        audit._date_argument("20260914")
