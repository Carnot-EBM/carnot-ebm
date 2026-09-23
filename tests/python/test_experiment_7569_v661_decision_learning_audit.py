"""Tests for REQ-REPORT-7569 and SCENARIO-REPORT-7569-*.

Private fixtures exercise corruptions. The real-data test reads producer bytes
without changing them or importing a producer reduction function.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7569_v661_decision_learning_audit as exp
from carnot.reporting import experiment_7303_validation_scope as validation_scope


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build a private command receipt without claiming a subprocess ran."""

    return {
        "name": name,
        "command": f"private {name}",
        "command_argv": ["private", name],
        "scope": "private_fixture",
        "exit_code": 0 if passed else 1,
        "duration_s": 0.01,
        "log_path": f"/tmp/{name}.log",
        "log_sha256": "sha256:" + "a" * 64,
        "output_tail": "private fixture",
        "passed": passed,
        "timed_out": False,
    }


def _receipts() -> list[dict[str, Any]]:
    """Represent every fixed check for pure artifact tests."""

    names = [*validation_scope.REQUIRED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES]
    return [_receipt(name) for name in names]


def test_inventory_retains_authentic_exp7568_gate_diagnostic() -> None:
    """REQ-REPORT-7569; SCENARIO-REPORT-7569-INVENTORY."""

    inventory = exp.inventory_producers(ROOT)
    assert [row["producer_id"] for row in inventory] == list(exp.PRODUCER_IDS)
    assert [row["state"] for row in inventory[:5]] == ["present"] * 5
    learning = inventory[-1]
    assert learning["state"] == "blocked_pre_gate"
    assert learning["declared_path"] == (
        "results/experiment_7568_v661_continuous_recalibration.json"
    )
    assert learning["evidence_path"] == "results/experiment_7568_continuous_recalibration.json"
    assert learning["honest_verdict"] == "blocked_gate_check_failed"
    failure = learning["gate_failure"]
    assert failure == {
        "upstream": "exp7561-recalibration-prototype",
        "path": (
            "/home/ianblenke/github.com/ianblenke/carnot/"
            "results/experiment_7561_v661_recalibration_prototype.json"
        ),
        "field": "recalibration_ready_score",
        "op": "==",
        "expected": 1,
        "observed": 0,
    }


def test_real_static_rows_reconstruct_from_features_and_coefficients() -> None:
    """REQ-REPORT-7569; SCENARIO-REPORT-7569-SOURCE."""

    inputs = exp.load_static_inputs(ROOT)
    reduced = exp.reconstruct_static(**inputs, draws=2_000)
    assert reduced["errors"] == []
    assert reduced["source_count"] == 80
    assert reduced["row_count"] == 480
    assert reduced["strongest_comparator"] == "temperature_original"
    assert reduced["max_probability_error"] <= exp.PROBABILITY_TOLERANCE
    assert reduced["max_metric_error"] <= exp.METRIC_TOLERANCE
    assert reduced["producer_agreement"] is True
    assert reduced["source_claims_qualified"] is True
    assert reduced["producer_probability_benefit_score"] == 0
    assert reduced["producer_decision_benefit_score"] == 0
    assert set(reduced["paired_intervals"]["brier"]) == set(exp.BRIER_COMPARATORS)


def test_eight_private_corruptions_fail_closed() -> None:
    """REQ-REPORT-7569; SCENARIO-REPORT-7569-MUTATIONS."""

    rows = exp.run_private_mutations()
    assert [row["mutation"] for row in rows] == list(exp.MUTATION_NAMES)
    assert all(row["passed"] is True for row in rows)
    assert all(row["observed_errors"] for row in rows)
    assert all(row["corrupted_fixture_published"] is False for row in rows)


def test_learning_fixture_checks_chronology_shuffle_restart_and_tail() -> None:
    """REQ-REPORT-7569; SCENARIO-REPORT-7569-LEARNING and -MUTATIONS."""

    fixture = exp.causal_fixture()
    reduced = exp.audit_causal_fixture(fixture)
    assert reduced["errors"] == []
    assert reduced["chronology_passed"] is True
    assert reduced["shuffle_passed"] is True
    assert reduced["restart_passed"] is True
    assert reduced["tail_passed"] is True
    assert reduced["retention_passed"] is True
    assert reduced["primal_constraints_passed"] is True
    assert reduced["optimizer_residuals_passed"] is True


def test_numerical_helpers_enforce_probability_and_metric_contract() -> None:
    """REQ-REPORT-7569; SCENARIO-REPORT-7569-SOURCE."""

    assert exp.sigmoid(0.0) == pytest.approx(0.5)
    assert exp.sigmoid(-1_000.0) == pytest.approx(0.0)
    assert exp.sigmoid(1_000.0) == pytest.approx(1.0)
    assert exp.losses(0.0, 1)["brier"] == 1.0
    assert exp.losses(1.0, 0)["log_loss"] > 20.0
    assert exp.typed_decision(0.03, 0)["action"] == "accept"
    assert exp.typed_decision(0.04, 0)["action"] == "escalate"
    assert exp.typed_decision(0.20, 1)["action"] == "escalate"
    assert exp.typed_decision(0.90, 1)["action"] == "reject"
    with pytest.raises(ValueError, match="probability_or_label_invalid"):
        exp.losses(float("nan"), 0)


def test_blocked_artifact_keeps_source_qualification_and_learning_block(tmp_path: Path) -> None:
    """REQ-REPORT-7569; SCENARIO-REPORT-7569-INVENTORY and -LEARNING."""

    artifact = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["source_claims_qualified_score"] == 1
    assert artifact["learning_claims_qualified_score"] == 0
    assert artifact["qualified_source_benefit_score"] == 0
    assert artifact["qualified_learning_benefit_score"] == 0
    assert artifact["branch_dispositions"]["source"]["verdict_class"] == "null"
    assert artifact["branch_dispositions"]["learning"]["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["first_failure"]["upstream"]
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["model_invoked"] is False
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["execution_venue"] == "host"
    assert set(artifact) <= set(artifact["field_principles"])
    assert exp.validate_artifact(artifact, root=tmp_path, verify_sources=False) == []

    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)
    assert exp.cold_replay(path, root=tmp_path, verify_sources=False) == []
    assert exp.independent_replay(path) == []


def test_reader_rejects_score_receipt_row_and_checksum_drift(tmp_path: Path) -> None:
    """REQ-REPORT-7569; SCENARIO-REPORT-7569-E2E."""

    artifact = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    changed = deepcopy(artifact)
    changed["qualified_source_benefit_score"] = 1
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "source_benefit_score_mismatch" in exp.validate_artifact(
        changed, root=tmp_path, verify_sources=False
    )

    changed = deepcopy(artifact)
    changed["validation_receipts"][0]["passed"] = False
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "required_validation_failed" in exp.validate_artifact(
        changed, root=tmp_path, verify_sources=False
    )

    changed = deepcopy(artifact)
    changed["rows"][0]["state"] = "positive"
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    assert "producer_rows_mismatch" in exp.validate_artifact(
        changed, root=tmp_path, verify_sources=False
    )

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    assert "reproducibility_checksum_mismatch" in exp.validate_artifact(
        changed, root=tmp_path, verify_sources=False
    )


def test_sidecar_and_json_readers_fail_on_changed_or_malformed_bytes(tmp_path: Path) -> None:
    """REQ-REPORT-7569; SCENARIO-REPORT-7569-INVENTORY."""

    path = tmp_path / "rows.jsonl"
    path.write_text('{"value":1}\n', encoding="utf-8")
    receipt = exp.sidecar_reference(path, tmp_path, rows=1)
    assert exp.read_sidecar(tmp_path, receipt, "rows") == [{"value": 1}]
    path.write_text('{"value":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar_hash_mismatch:rows"):
        exp.read_sidecar(tmp_path, receipt, "rows")

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[", encoding="utf-8")
    assert exp.load_json(malformed) == {}
    assert exp.cold_replay(malformed, root=tmp_path) == ["artifact_not_object"]
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_row_not_object"):
        exp.load_jsonl(path)


def test_preconditions_and_scoped_commands_are_exact(tmp_path: Path) -> None:
    """REQ-REPORT-7569; SCENARIO-REPORT-7569-E2E."""

    preconditions = exp.collect_preconditions(ROOT)
    assert all(row["passed"] for row in preconditions["rows"] if row["required"])
    assert preconditions["missing_external"] == [
        "results/experiment_7568_v661_continuous_recalibration.json"
    ]
    commands = exp.build_validation_commands(ROOT, tmp_path)
    assert [row.name for row in commands] == list(validation_scope.REQUIRED_CHECK_NAMES)
    assert exp.AFFECTED_MANIFEST.test_paths == (
        "tests/python/test_experiment_7569_v661_decision_learning_audit.py",
    )
    terminal = exp.terminal_commands(tmp_path / "candidate.json")
    assert [row.spec.name for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)
    args = exp.parse_args(["--date", "20260923", "--root", str(ROOT)])
    assert args.date == "20260923"
    assert args.root == ROOT


def test_checksum_binds_stable_content() -> None:
    """REQ-REPORT-7569; SCENARIO-REPORT-7569-E2E."""

    value = {"alpha": 1, "duration_s": 1.0, "reproducibility_checksum": "old"}
    first = exp.reproducibility_checksum(value)
    value["duration_s"] = 2.0
    assert exp.reproducibility_checksum(value) == first
    value["alpha"] = 2
    assert exp.reproducibility_checksum(value) != first


def test_wrong_run_date_is_rejected() -> None:
    """REQ-REPORT-7569 binds the exact run date."""

    with pytest.raises(ValueError, match="run_date_must_equal_20260923"):
        exp.parse_args(["--date", "20260922"])


def test_private_source_guards_name_exact_corruption(tmp_path: Path) -> None:
    """REQ-REPORT-7569 rejects changed custody and row-level source drift."""

    path = tmp_path / "rows.jsonl"
    path.write_text('{"value":1}\n', encoding="utf-8")
    reference = exp.sidecar_reference(path, tmp_path, rows=1)
    changed = {**reference, "bytes": reference["bytes"] + 1}
    with pytest.raises(ValueError, match="sidecar_size_mismatch:rows"):
        exp.read_sidecar(tmp_path, changed, "rows")
    changed = {**reference, "rows": 2}
    with pytest.raises(ValueError, match="sidecar_row_count_mismatch:rows"):
        exp.read_sidecar(tmp_path, changed, "rows")
    assert exp._diagnostic_failure({}) == {}
    with pytest.raises(ValueError, match="undeclared_or_changed_sidecar"):
        exp._authenticate_declared(tmp_path, Path("missing.jsonl"), [])

    fixture = exp.static_fixture()
    heads = deepcopy(fixture["heads"])
    heads["selected_heads"][exp.CANDIDATE_ARM]["checkpoint_sha256"] = "changed"
    with pytest.raises(ValueError, match="checkpoint_hash_mismatch"):
        exp._trained_probability(exp.CANDIDATE_ARM, [0.0, 0.0, 0.0], heads)

    labels = deepcopy(fixture["labels"])
    labels.append(deepcopy(labels[0]))
    labels[0]["role"] = "invalid"
    features = deepcopy(fixture["features"])
    features[0]["original_unsupported_probability"] = 1.0
    features.append(
        {
            **deepcopy(features[0]),
            "component_hash": "component-without-label",
        }
    )
    _rows, errors = exp._reconstruct_rows(features, labels, fixture["heads"])
    assert any(error.startswith("duplicate_label_component") for error in errors)
    assert any(error.startswith("label_row_invalid") for error in errors)
    assert any(error.startswith("source_label_missing") for error in errors)
    assert any(error.startswith("option_orientation_probability_mismatch") for error in errors)


def test_private_published_row_comparator_fails_every_stable_field() -> None:
    """REQ-REPORT-7569 checks exact rows before trusting producer aggregates."""

    fixture = exp.static_fixture()
    rebuilt, assert_errors = exp._reconstruct_rows(
        fixture["features"], fixture["labels"], fixture["heads"]
    )
    assert assert_errors == []
    published = deepcopy(fixture["published_rows"])
    published[0]["option_probabilities"]["supported_first"] += 0.1
    published[0]["probability"] += 0.1
    published[0]["brier"] += 0.1
    published[0]["action"] = "wrong"
    published[0]["head_identity"] = "wrong"
    published[0]["prediction_freeze_sha256"] = "wrong"
    published.append(deepcopy(published[1]))
    published.pop(2)
    errors, probability_error, metric_error = exp._compare_rows(
        rebuilt, published, fixture["heads"], fixture["report"]
    )
    assert any(error.startswith("duplicate_published_component_arm") for error in errors)
    assert "published_row_roster_mismatch" in errors
    assert any(error.startswith("action_mismatch") for error in errors)
    assert any(error.startswith("head_hash_mismatch") for error in errors)
    assert any(error.startswith("prediction_hash_mismatch") for error in errors)
    assert "probability_tolerance_exceeded" in errors
    assert "metric_tolerance_exceeded" in errors
    assert probability_error > exp.PROBABILITY_TOLERANCE
    assert metric_error > exp.METRIC_TOLERANCE
    assert exp._nested_error({"a": 1}, {"b": 1}) == float("inf")
    assert exp._nested_error([1], [1, 2]) == float("inf")


def test_static_and_causal_control_failures_remain_disqualifying() -> None:
    """REQ-REPORT-7569 keeps invalid evidence separate from an honest null."""

    fixture = exp.static_fixture()
    producer = {
        "static_measurement_complete_score": 0,
        "flagged_adversarial": True,
        "validation_summary": {"required_checks_passed": False},
    }
    fixture["heads"]["strongest_comparator"]["family"] = "wrong"
    fixture["report"]["reduction"]["probability_metrics"] = {}
    reduced = exp.reconstruct_static(
        **fixture,
        references={},
        producer=producer,
        draws=16,
    )
    assert {
        "producer_aggregate_mismatch",
        "strongest_comparator_mismatch",
        "static_count_mismatch",
        "producer_measurement_incomplete",
        "producer_critical_flag",
        "producer_validation_failed",
    } <= set(reduced["errors"])

    causal = exp.causal_fixture()
    causal["predictions"].append(deepcopy(causal["predictions"][0]))
    causal["predictions"][0]["prediction_time"] = 8
    causal["predictions"][0]["label_available"] = True
    causal["predictions"][-1]["prediction_time"] = 8
    causal["predictions"][-1]["label_available"] = True
    causal["updates"][0]["update_time"] = 0
    causal["updates"].append(deepcopy(causal["updates"][0]))
    causal["states"][0]["state_hash"] = "wrong"
    causal["states"][0]["primal_violation"] = 1.0
    causal["states"][0]["optimizer_residual"] = 1.0
    causal["retention_rows"][0]["label_returned"] = True
    audit = exp.audit_causal_fixture(causal)
    assert {
        "duplicate_prediction",
        "prediction_not_before_release:event-0",
        "label_visible_at_prediction:event-0",
        "update_before_release:event-0",
        "exactly_once_update_invalid",
        "state_hash_chain_invalid",
        "primal_constraint_failed",
        "optimizer_residual_failed",
        "retention_isolation_failed",
    } <= set(audit["errors"])
    assert audit["chronology_passed"] is False
    assert audit["retention_passed"] is False


def test_validator_reports_all_schema_and_source_custody_failures(tmp_path: Path) -> None:
    """REQ-REPORT-7569 cold validation fails closed on every required boundary."""

    artifact = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    existing = tmp_path / "existing.json"
    existing.write_text("{}\n", encoding="utf-8")
    other = tmp_path / "other.json"
    other.write_text("{}\n", encoding="utf-8")
    artifact["source_artifact_hashes"] = [
        {"path": "missing.json", "sha256": "missing", "bytes": 0},
        {"path": "existing.json", "sha256": "wrong", "bytes": existing.stat().st_size},
        {
            "path": "other.json",
            "sha256": exp.sha256_file(other),
            "bytes": other.stat().st_size + 1,
        },
    ]
    artifact["schema"] = "wrong"
    artifact["milestone"] = "wrong"
    artifact.pop("sample_size_budget")
    artifact["MODEL_SPECS"] = ["wrong"]
    artifact["model_invoked"] = True
    artifact["invocation_counts"] = {}
    artifact["inference_substrate_class"] = "wrong"
    artifact["execution_venue"] = "wrong"
    artifact["source_claims_qualified_score"] = 0
    artifact["learning_claims_qualified_score"] = 1
    artifact["verdict_class"] = "null"
    artifact["mutation_rows"] = []
    artifact["field_principles"] = {}
    errors = exp.validate_artifact(artifact, root=tmp_path)
    assert "artifact_identity_mismatch" in errors
    assert "task_binding_mismatch" in errors
    assert any(error.startswith("required_fields_missing") for error in errors)
    assert "model_specs_not_empty" in errors
    assert "model_invocation_mismatch" in errors
    assert "invocation_counts_mismatch" in errors
    assert "inference_substrate_mismatch" in errors
    assert "execution_venue_mismatch" in errors
    assert "source_claims_score_mismatch" in errors
    assert "learning_score_mismatch" in errors
    assert "terminal_verdict_mismatch" in errors
    assert "mutation_contract_failed" in errors
    assert "field_principles_incomplete" in errors
    assert "source_missing:missing.json" in errors
    assert "source_hash_mismatch:existing.json" in errors
    assert "source_size_mismatch:other.json" in errors


def test_independent_reader_reduces_embedded_and_sealed_rows(tmp_path: Path) -> None:
    """REQ-REPORT-7569 replays serialized rows and authentic sealed operands."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[", encoding="utf-8")
    assert exp.independent_replay(malformed) == ["artifact_not_object"]

    private = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    changed = deepcopy(private)
    changed["rows"] = [
        row for index, row in enumerate(changed["rows"]) if index != len(changed["rows"]) - 1
    ]
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    private_path = tmp_path / "private-missing-arm.json"
    exp.atomic_json(private_path, changed)
    assert any(
        error.startswith("embedded_source_reduction_failed")
        for error in exp.independent_replay(private_path)
    )

    changed = deepcopy(private)
    changed["static_reconstruction"]["probability_metrics"][exp.CANDIDATE_ARM]["brier"] += 0.1
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    private_path = tmp_path / "private-bad-summary.json"
    exp.atomic_json(private_path, changed)
    assert "embedded_source_reduction_mismatch:probability_metrics" in exp.independent_replay(
        private_path
    )

    preconditions = exp.collect_preconditions(ROOT)
    static = exp.reconstruct_static(**exp.load_static_inputs(ROOT), draws=exp.BOOTSTRAP_DRAWS)
    sealed = exp._build_artifact(
        root=ROOT,
        preconditions=preconditions,
        static=static,
        mutations=exp.run_private_mutations(),
        validation_receipts=_receipts(),
        evidence_mode="sealed_real",
        duration_s=0.1,
    )
    sealed_path = tmp_path / "sealed.json"
    exp.atomic_json(sealed_path, sealed)
    assert exp.independent_replay(sealed_path) == []

    changed = deepcopy(sealed)
    source_row = next(row for row in changed["rows"] if row["row_kind"] == "static_source_arm")
    source_row["probability"] += 0.01
    changed["static_reconstruction"]["probability_metrics"][exp.CANDIDATE_ARM]["brier"] += 0.1
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    changed_path = tmp_path / "sealed-changed.json"
    exp.atomic_json(changed_path, changed)
    replay_errors = exp.independent_replay(changed_path)
    assert "sealed_source_rows_mismatch" in replay_errors
    assert "sealed_source_reduction_mismatch:probability_metrics" in replay_errors

    blocked = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    blocked["evidence_mode"] = "sealed_real"
    blocked["reproducibility_checksum"] = exp.reproducibility_checksum(blocked)
    blocked_path = tmp_path / "sealed-missing.json"
    exp.atomic_json(blocked_path, blocked)
    assert any(
        error.startswith("sealed_source_replay_failed")
        for error in exp.independent_replay(blocked_path)
    )
