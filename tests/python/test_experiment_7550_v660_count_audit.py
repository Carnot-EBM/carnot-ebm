"""Tests for REQ-REPORT-7550 and SCENARIO-REPORT-7550-*.

The tests use compact private evidence for mutations. One test reads the real
V660 sidecars to prove that the public reducer does not depend on fixtures.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_7550_v660_count_audit as exp
from carnot.reporting import experiment_7303_validation_scope as validation_scope


ROOT = Path(__file__).resolve().parents[2]


def _receipt(name: str, passed: bool = True) -> dict[str, Any]:
    """Build one command receipt without claiming that a child process ran."""

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


def _receipts(*, terminal: bool = True) -> list[dict[str, Any]]:
    """Represent every fixed check for pure artifact tests."""

    names = list(validation_scope.REQUIRED_CHECK_NAMES)
    if terminal:
        names.extend(exp.TERMINAL_CHECK_NAMES)
    return [_receipt(name) for name in names]


def test_direct_beta_arithmetic_and_binary_normalization() -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-CAUSALITY."""

    probability, energies = exp.direct_beta_prediction(
        0.8, successes=5.0, failures=4.0, prior_mean=0.5
    )
    assert probability == pytest.approx(5.0 / 6.0)
    assert exp.normalized_binary_probability(energies) == pytest.approx(probability)
    assert exp.brier(probability, 1) == pytest.approx(1.0 / 36.0)
    decision = exp.typed_decision(0.8, 1)
    assert decision["action"] == "escalate"
    assert decision["realized_cost"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="binary_energy_requires_two_values"):
        exp.normalized_binary_probability([0.0])
    with pytest.raises(ValueError, match="probability_not_finite"):
        exp.direct_beta_prediction(float("nan"), 4.0, 4.0, 0.5)


def test_private_fixture_reconstructs_counts_chronology_and_retention() -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-CAUSALITY and -REDUCTION."""

    evidence = exp.fixture_evidence()
    reduced = exp.audit_evidence(**evidence)
    assert reduced["audit_errors"] == []
    assert reduced["arithmetic_passed"] is True
    assert reduced["chronology_passed"] is True
    assert reduced["restart_passed"] is True
    assert reduced["retention_isolation_passed"] is True
    assert reduced["complete_online_source_count"] == 4
    assert reduced["label_counts"] == {"0": 2, "1": 2}
    assert set(reduced["absolute_brier"]) == set(exp.ARMS)
    assert set(reduced["absolute_primary_cost"]) == set(exp.ARMS)


def test_six_private_mutations_fail_their_named_checks() -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-MUTATIONS."""

    rows = exp.run_private_mutations()
    assert [row["mutation"] for row in rows] == list(exp.MUTATION_NAMES)
    assert all(row["passed"] is True for row in rows)
    assert all(row["observed_errors"] for row in rows)
    assert all(row["corrupted_fixture_published"] is False for row in rows)


def test_sidecar_custody_rejects_changed_bytes(tmp_path: Path) -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-INVENTORY."""

    path = tmp_path / "rows.jsonl"
    path.write_text('{"unit":1}\n', encoding="utf-8")
    receipt = exp.sidecar_receipt(path, tmp_path, rows=1)
    assert exp.read_sidecar(tmp_path, receipt, "rows") == [{"unit": 1}]
    path.write_text('{"unit":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="raw_sidecar_hash_mismatch:rows"):
        exp.read_sidecar(tmp_path, receipt, "rows")


def test_missing_upstream_builds_exact_complete_blocked_record(tmp_path: Path) -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-INVENTORY."""

    checks, hashes, upstreams = exp.collect_preconditions(tmp_path)
    assert hashes == {}
    assert upstreams == {}
    failed = next(row for row in checks if row["required"] and not row["passed"])
    artifact = exp.build_blocked_artifact(failed, checks, duration_s=0.25)
    assert artifact["honest_verdict"].startswith("complete_blocked_")
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"]["first_failure"]["upstream"]
    assert artifact["gate_check_summary"]["first_failure"]["path_or_field"]
    assert artifact["count_claims_qualified_score"] == 0
    assert artifact["qualified_exploratory_effect_score"] == 0
    assert artifact["confirmatory_benefit_score"] == 0
    exp.validate_artifact(artifact, root=tmp_path, require_terminal=False)


def test_real_upstreams_reconstruct_the_valid_null() -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-REDUCTION and -NULL."""

    checks, _hashes, upstreams = exp.collect_preconditions(ROOT)
    assert all(row["passed"] for row in checks if row["required"])
    evidence = exp.load_upstream_evidence(ROOT, upstreams)
    reduced = exp.audit_evidence(**evidence)
    assert reduced["audit_errors"] == []
    assert reduced["complete_online_source_count"] == 159
    assert reduced["label_counts"] == {"0": 91, "1": 68}
    assert reduced["absolute_brier"]["local_count"] == pytest.approx(0.14997917983634546)
    assert reduced["absolute_primary_cost"]["local_count"] >= 0.0
    assert reduced["count_claims_qualified"] is True
    assert reduced["effect_passed"] is False
    assert reduced["retention_passed"] is False
    assert reduced["producer_aggregate_agreement"] is True


def test_null_artifact_stays_qualified_without_promotion(tmp_path: Path) -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-NULL and -E2E."""

    artifact = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    assert artifact["verdict_class"] == "null"
    assert artifact["honest_verdict"].startswith("complete_null_")
    assert artifact["count_claims_qualified_score"] == 1
    assert artifact["qualified_exploratory_effect_score"] == 0
    assert artifact["confirmatory_benefit_score"] == 0
    assert artifact["positive_claim"] is False
    assert artifact["MODEL_SPECS"] == artifact["model_specs"] == []
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["execution_venue"] == "host"
    assert artifact["rows"]
    assert set(artifact) <= set(artifact["field_principles"])
    assert exp.validate_artifact(artifact, root=tmp_path)["effect_passed"] is False

    path = tmp_path / "candidate.json"
    exp.atomic_json(path, artifact)
    assert exp.cold_replay(path, root=tmp_path)["count_claims_qualified"] is True


def test_artifact_reader_rejects_score_receipt_and_checksum_drift(tmp_path: Path) -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-E2E."""

    artifact = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())
    changed = deepcopy(artifact)
    changed["confirmatory_benefit_score"] = 1
    with pytest.raises(ValueError, match="confirmatory_benefit_score_mismatch"):
        exp.validate_artifact(changed, root=tmp_path)

    changed = deepcopy(artifact)
    changed["validation_receipts"][0]["passed"] = False
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="required_validation_failed"):
        exp.validate_artifact(changed, root=tmp_path)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="reproducibility_checksum_mismatch"):
        exp.validate_artifact(changed, root=tmp_path)


def test_manifest_commands_and_cli_are_exact(tmp_path: Path) -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-E2E."""

    commands = exp.build_validation_commands(ROOT, tmp_path)
    assert [row.name for row in commands] == list(validation_scope.REQUIRED_CHECK_NAMES)
    assert exp.AFFECTED_MANIFEST.test_paths == (
        "tests/python/test_experiment_7550_v660_count_audit.py",
    )
    terminal = exp.terminal_commands(tmp_path / "candidate.json", ROOT)
    assert [row.name for row in terminal] == list(exp.TERMINAL_CHECK_NAMES)
    assert all(row.timeout_s == 300.0 for row in terminal)
    args = exp.parse_args(["--date", "20260923", "--root", str(ROOT)])
    assert args.date == "20260923"
    assert args.root == ROOT


def test_json_readers_reject_malformed_content(tmp_path: Path) -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-INVENTORY."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("[", encoding="utf-8")
    assert exp.load_json(malformed) == {}
    with pytest.raises(ValueError, match="artifact_unreadable_or_not_object"):
        exp.cold_replay(malformed, root=tmp_path)

    rows = tmp_path / "rows.jsonl"
    rows.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_row_not_object"):
        exp.load_jsonl(rows)


def test_defensive_causal_and_retention_branches() -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-MUTATIONS."""

    base = exp.fixture_evidence()
    changed = deepcopy(base)
    changed["protocol"]["labels_read"] = True
    changed["protocol"]["count_config"]["kappa"] = 7.0
    changed["prediction_rows"].append(deepcopy(changed["prediction_rows"][0]))
    changed["persistence_rows"].append(deepcopy(changed["persistence_rows"][0]))
    changed["prediction_rows"][1]["base_probability"] = 0.7
    changed["prediction_rows"][2]["label"] = 0
    errors, _snapshots = exp._audit_orders(
        changed["protocol"],
        changed["prediction_rows"],
        changed["release_rows"],
        changed["persistence_rows"],
    )
    assert "protocol_labels_read_before_freeze" in errors
    assert "count_config_mismatch" in errors
    assert any(error.startswith("duplicate_prediction_arm") for error in errors)
    assert any(error.startswith("duplicate_checkpoint") for error in errors)
    assert any(error.startswith("prediction_operand_mismatch") for error in errors)

    changed = deepcopy(base)
    row = changed["prediction_rows"][0]
    row["label_available_at_prediction"] = True
    row["prediction_time"] = 99
    row["bin_index"] = 0
    row["feedback_update_id"] = "wrong"
    row["feedback_availability_time"] = 99
    changed["release_rows"][0]["release_time"] = 3
    changed["release_rows"][0]["source_to_label_permutation"].pop()
    changed["release_rows"][1]["event_ids"] = ["wrong"]
    changed["release_rows"][1]["update_id"] = "unexpected"
    checkpoint = changed["persistence_rows"][0]
    checkpoint["checkpoint_sha256"] = "bad"
    checkpoint["payload_equal_after_reload"] = False
    checkpoint["uninterrupted_state_equal"] = False
    checkpoint["duplicate_feedback_rejected"] = False
    errors, _snapshots = exp._audit_orders(
        changed["protocol"],
        changed["prediction_rows"],
        changed["release_rows"],
        changed["persistence_rows"],
    )
    expected_fragments = (
        "future_label_access",
        "prediction_time_mismatch",
        "bin_mismatch",
        "release_schedule_mismatch",
        "shuffled_group_roster_mismatch",
        "checkpoint_hash_invalid",
        "checkpoint_payload_mismatch",
        "checkpoint_restart_mismatch",
        "checkpoint_duplicate_guard_missing",
        "censored_schedule_mismatch",
        "censored_update_present",
    )
    assert all(
        any(error.startswith(fragment) for error in errors) for fragment in expected_fragments
    )

    changed = deepcopy(base)
    errors, snapshots = exp._audit_orders(
        changed["protocol"],
        changed["prediction_rows"],
        changed["release_rows"],
        changed["persistence_rows"],
    )
    assert errors == []
    counts = snapshots[(1, 0)]
    probability, energies, _index = exp._predict_arm(
        "frozen", 0.6, changed["protocol"]["count_config"], counts
    )
    retention = {
        "seed": 1,
        "arrival_checkpoint": 0,
        "group_id": "retained",
        "arm": "frozen",
        "base_probability": 0.6,
        "probability": probability + 0.1,
        "energies": [energies[0], energies[1] + 0.1],
        "normalized_probability": 0.1,
        "label": 1,
        "brier": 0.9,
        "decision": {"action": "wrong"},
        "label_returned_to_learner": True,
    }
    retention_errors = exp._audit_retention(
        changed["protocol"],
        [retention, {**retention, "arrival_checkpoint": 999}],
        snapshots,
    )
    assert any(error.startswith("retention_snapshot_missing") for error in retention_errors)
    for fragment in (
        "retention_label_leak",
        "retention_probability_mismatch",
        "retention_energy_mismatch",
        "retention_normalization_mismatch",
        "retention_brier_mismatch",
        "retention_cost_mismatch",
    ):
        assert any(error.startswith(fragment) for error in retention_errors)


def test_low_level_duplicate_and_unknown_mutation_guards() -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-MUTATIONS."""

    counts = exp._initial_counts({"bin_means": [0.5] * 8, "global_mean": 0.5, "kappa": 8.0})
    assert exp._update_count(counts, "local", "event", 0.5, 1) is None
    assert exp._update_count(counts, "local", "event", 0.5, 1).startswith("duplicate_update")
    assert exp._update_count(counts, "local", "other", 0.5, 2).startswith("nonbinary_label")
    with pytest.raises(ValueError, match="unknown_private_mutation"):
        exp._mutate_evidence(exp.fixture_evidence(), "unknown")


def test_sidecar_shape_and_json_object_guards(tmp_path: Path) -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-INVENTORY."""

    path = tmp_path / "rows.jsonl"
    path.write_text('{"unit":1}\n', encoding="utf-8")
    receipt = exp.sidecar_receipt(path, tmp_path, rows=1)
    with pytest.raises(ValueError, match="raw_sidecar_size_mismatch"):
        exp.read_sidecar(tmp_path, {**receipt, "bytes": 0}, "rows")
    with pytest.raises(ValueError, match="raw_sidecar_row_count_mismatch"):
        exp.read_sidecar(tmp_path, {**receipt, "rows": 2}, "rows")

    object_path = tmp_path / "object.json"
    exp.atomic_json(object_path, {"ok": True})
    object_receipt = exp._json_receipt(object_path, tmp_path)
    with pytest.raises(ValueError, match="raw_sidecar_hash_mismatch"):
        exp._read_json_sidecar(
            tmp_path, {**object_receipt, "sha256": "sha256:" + "0" * 64}, "object"
        )
    with pytest.raises(ValueError, match="raw_sidecar_size_mismatch"):
        exp._read_json_sidecar(tmp_path, {**object_receipt, "bytes": 0}, "object")
    object_path.write_text("[]\n", encoding="utf-8")
    malformed_receipt = exp._json_receipt(object_path, tmp_path)
    with pytest.raises(ValueError, match="raw_sidecar_object_invalid"):
        exp._read_json_sidecar(tmp_path, malformed_receipt, "object")


def test_resample_callback_and_reducer_row_guards() -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-REDUCTION."""

    checks, _hashes, upstreams = exp.collect_preconditions(ROOT)
    assert all(row["passed"] for row in checks if row["required"])
    evidence = exp.load_upstream_evidence(ROOT, upstreams)
    completed: list[int] = []
    rows = exp.source_cluster_resamples(
        evidence["protocol"],
        evidence["prediction_rows"],
        evidence["retention_rows"],
        replicates=1,
        progress_hook=completed.append,
    )
    assert completed == [1]
    assert len(rows) == 1

    changed = exp.fixture_evidence()
    changed["bootstrap_rows"] = [{"unexpected": True}]
    changed["prediction_rows"].append(
        {
            "seed": 1,
            "event_id": "unknown",
            "arm": "unknown",
            "label": 0,
            "brier": 0.0,
            "decision": {"realized_cost": 0.0},
        }
    )
    reduced = exp.audit_evidence(**changed)
    assert "source_resample_mismatch" in reduced["audit_errors"]
    assert any(error.startswith("unknown_arm") for error in reduced["audit_errors"])


def test_artifact_all_defensive_reader_boundaries(tmp_path: Path) -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-E2E."""

    artifact = exp.build_test_artifact(tmp_path, validation_receipts=_receipts())

    def rejected(field: str, value: Any, message: str) -> None:
        changed = deepcopy(artifact)
        changed[field] = value
        changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
        with pytest.raises(ValueError, match=message):
            exp.validate_artifact(changed, root=tmp_path)

    rejected("schema", "wrong", "artifact_identity_mismatch")
    rejected("run_date", "20260922", "artifact_date_or_milestone_mismatch")
    rejected("MODEL_SPECS", ["model"], "model_specs_must_be_empty")
    rejected("model_invoked", True, "current_model_invocation_mismatch")
    rejected("invocation_counts", {}, "current_invocation_counts_mismatch")
    rejected("inference_substrate_class", "no_model_load", "inference_substrate_class_mismatch")
    rejected("inference_substrate", "wrong", "inference_substrate_mismatch")
    rejected("execution_venue", "host_cpu", "execution_venue_mismatch")
    rejected("positive_claim", True, "positive_claim_mismatch")
    rejected("field_principles", {}, "field_principles_incomplete")

    blocked = exp.build_blocked_artifact(
        {
            "check": "missing",
            "expected": True,
            "observed": None,
            "op": "eq",
            "upstream": "",
            "path_or_field": "",
        },
        [],
        duration_s=0.1,
    )
    with pytest.raises(ValueError, match="blocked_gate_summary_incomplete"):
        exp.validate_artifact(blocked, root=tmp_path, require_terminal=False)

    changed = deepcopy(artifact)
    changed["independent_reduction"]["absolute_brier"]["local_count"] += 0.1
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="independent_reduction_mismatch"):
        exp.validate_artifact(changed, root=tmp_path)

    changed = deepcopy(artifact)
    changed["mutation_rows"] = []
    changed["reproducibility_checksum"] = exp.reproducibility_checksum(changed)
    with pytest.raises(ValueError, match="private_mutation_validation_failed"):
        exp.validate_artifact(changed, root=tmp_path)

    for field, value, message in (
        ("count_claims_qualified_score", 0, "count_claims_qualified_score_mismatch"),
        ("qualified_exploratory_effect_score", 1, "qualified_exploratory_effect_score_mismatch"),
        ("verdict_class", "positive", "verdict_class_mismatch"),
    ):
        rejected(field, value, message)


def test_terminal_classification_and_manifest_helpers(tmp_path: Path) -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-NULL and -E2E."""

    evidence = exp.fixture_evidence()
    raw = exp.write_fixture_evidence(tmp_path, evidence)
    reduction = exp.audit_evidence(**evidence)
    mutations = exp.run_private_mutations()
    failed_receipts = _receipts()
    failed_receipts[0]["passed"] = False
    failed_receipts[0]["exit_code"] = 1
    disqualified = exp.build_artifact(
        reduction=reduction,
        mutation_rows=mutations,
        validation_receipts=failed_receipts,
        preconditions_checked=[],
        source_hashes={},
        duration_s=0.1,
        phase_spans=[],
        raw_evidence=raw,
        process_root=tmp_path,
    )
    assert disqualified["verdict_class"] == "disqualified"

    beneficial = deepcopy(reduction)
    beneficial["effect_passed"] = True
    circular = exp.build_artifact(
        reduction=beneficial,
        mutation_rows=mutations,
        validation_receipts=_receipts(),
        preconditions_checked=[],
        source_hashes={},
        duration_s=0.1,
        phase_spans=[],
        raw_evidence=raw,
        process_root=tmp_path,
    )
    assert circular["verdict_class"] == "circular_positive"

    manifest = tmp_path / "manifest.json"
    exp._write_affected_manifest(manifest)
    assert json.loads(manifest.read_text())["experiment_id"] == exp.EXPERIMENT_ID
    checks, _hashes, upstreams = exp.collect_preconditions(ROOT)
    assert all(row["passed"] for row in checks if row["required"])
    actual = exp._actual_raw_receipts(ROOT, upstreams)
    assert actual["producer"]["sha256"] == exp.sha256_file(ROOT / exp.PRODUCER_PATH)
    source_hashes = exp._source_hashes(ROOT, ROOT / exp.SPEC_PATH)
    assert exp.MODULE_PATH.as_posix() in source_hashes


def test_release_duplicate_and_missing_checkpoint_are_rejected() -> None:
    """REQ-REPORT-7550; SCENARIO-REPORT-7550-CAUSALITY."""

    evidence = exp.fixture_evidence()
    release = evidence["release_rows"][0]
    release["event_ids"] = ["group-0", "group-0"]
    release["source_to_label_permutation"] = [
        {"event_id": "group-0", "label_origin": "group-0", "label": 1}
    ]
    release["label_multiset"] = [1, 1]
    release["permuted_label_multiset"] = [1, 1]
    errors, _snapshots = exp._audit_orders(
        evidence["protocol"],
        evidence["prediction_rows"],
        evidence["release_rows"],
        [],
    )
    assert any(error.startswith("duplicate_update") for error in errors)
    assert any(error.startswith("checkpoint_missing") for error in errors)
    assert "checkpoint_count_mismatch" in errors
