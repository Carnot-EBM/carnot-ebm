"""Tests for the V663 evidence and delayed-update audit.

Spec refs: REQ-REPORT-7596 and SCENARIO-REPORT-7596-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7596_v663_evidence_audit as audit


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


# REQ-REPORT-7596 / SCENARIO-REPORT-7596-CUSTODY
def test_source_custody_distinguishes_all_dispositions(tmp_path: Path) -> None:
    producer = audit.SourceSpec(
        "producer", Path("results/producer.json"), Path("results/producer_gate.json")
    )
    gate = audit.SourceSpec("gate", Path("results/gate.json"), Path("results/gate_diag.json"))
    flagged = audit.SourceSpec(
        "flagged", Path("results/flagged.json"), Path("results/flagged_gate.json")
    )
    missing = audit.SourceSpec(
        "missing", Path("results/missing.json"), Path("results/missing_gate.json")
    )
    _write_json(
        tmp_path / producer.producer_path,
        {"honest_verdict": "complete_null", "verdict_class": "null", "flagged_adversarial": False},
    )
    _write_json(
        tmp_path / gate.conductor_path,
        {"schema": "blocked_gate_check_v1", "honest_verdict": "blocked_gate_check_failed"},
    )
    _write_json(
        tmp_path / flagged.producer_path,
        {
            "honest_verdict": "complete_positive",
            "verdict_class": "positive",
            "flagged_adversarial": True,
        },
    )

    receipts = [
        audit.classify_source(tmp_path, spec) for spec in (producer, gate, flagged, missing)
    ]
    assert [row["disposition"] for row in receipts] == [
        "authenticated_producer",
        "conductor_pre_gate",
        "flagged_producer",
        "missing_producer",
    ]
    assert receipts[0]["sha256"].startswith("sha256:")
    assert receipts[1]["path"] == gate.conductor_path.as_posix()
    assert receipts[2]["eligible_for_science"] is False
    assert receipts[3]["sha256"] is None


# REQ-REPORT-7596 / SCENARIO-REPORT-7596-MUTATIONS
def test_all_seven_private_mutations_change_bytes_and_fail_closed() -> None:
    assert audit.validate_private_fixture(audit.private_fixture()) == []
    receipts = audit.run_private_mutations()
    assert {row["mutation"] for row in receipts} == set(audit.MUTATIONS)
    assert all(row["passed"] is True for row in receipts)
    assert all(row["before_sha256"] != row["after_sha256"] for row in receipts)
    assert all(row["observed_failures"] for row in receipts)


# REQ-REPORT-7596 / SCENARIO-REPORT-7596-BLOCKED
def test_actual_missing_producers_build_complete_blocked_artifact() -> None:
    checks, receipts = audit.collect_preconditions(audit.REPO_ROOT)
    by_name = {row["upstream"]: row for row in receipts}
    assert by_name["exp7590-evidence-pilot"]["disposition"] == "conductor_pre_gate"
    assert by_name["exp7594-decision-evaluation"]["disposition"] == "missing_producer"
    assert by_name["exp7595-guarded-learning"]["disposition"] == "missing_producer"

    artifact = audit.build_blocked_artifact(
        audit.REPO_ROOT,
        checks,
        receipts,
        validation_receipts=audit.provisional_validation_receipts(),
        duration_s=0.1,
        phase_spans=[],
    )
    assert artifact["honest_verdict"] == "complete_blocked_missing_v663_evidence_producers"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["static_audit_ready_score"] == 0
    assert artifact["online_audit_ready_score"] == 0
    assert artifact["rows"] == []
    first = artifact["gate_check_summary"]["first_failure"]
    assert first == {
        "check": "required_scientific_producer",
        "upstream": "exp7594-decision-evaluation",
        "path": "results/experiment_7594_v663_decision_evaluation.json",
        "field": "exists_and_eligible",
        "op": "eq",
        "expected": True,
        "observed": False,
    }
    assert {row["branch"] for row in artifact["branch_conclusions"]} == {
        "incremental_evidence",
        "probability",
        "decision_cost",
        "causal_update_behavior",
        "retention",
        "exposure",
    }


# REQ-REPORT-7596 / SCENARIO-REPORT-7596-TERMINAL
def test_blocked_artifact_checksum_and_cold_replay(tmp_path: Path) -> None:
    checks, receipts = audit.collect_preconditions(audit.REPO_ROOT)
    artifact = audit.build_blocked_artifact(
        audit.REPO_ROOT,
        checks,
        receipts,
        validation_receipts=audit.provisional_validation_receipts(),
        duration_s=0.2,
        phase_spans=[],
    )
    assert audit.validate_artifact(artifact, root=audit.REPO_ROOT) == {"valid": True}
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert audit.cold_replay(candidate, root=audit.REPO_ROOT) == {"valid": True}
    assert audit.independent_replay(candidate, root=audit.REPO_ROOT)["row_count"] == 0

    changed = deepcopy(artifact)
    changed["MODEL_SPECS"] = ["forbidden"]
    changed["static_audit_ready_score"] = 1
    changed["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError) as exc:
        audit.validate_artifact(changed, root=audit.REPO_ROOT)
    assert "model_specs_not_empty" in str(exc.value)
    assert "blocked_readiness_nonzero" in str(exc.value)
    assert "checksum_mismatch" in str(exc.value)


# REQ-REPORT-7596 / SCENARIO-REPORT-7596-CUSTODY
def test_source_receipt_detects_byte_drift_and_invalid_json(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    _write_json(source, {"value": 1})
    receipt = audit.source_receipt(source, tmp_path, "fixture")
    assert audit.authenticate_source_receipt(receipt, tmp_path) == source
    source.write_text('{"value": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        audit.authenticate_source_receipt(receipt, tmp_path)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    spec = audit.SourceSpec("invalid", Path("invalid.json"), Path("gate.json"))
    classified = audit.classify_source(tmp_path, spec)
    assert classified["disposition"] == "invalid_producer"
    assert classified["eligible_for_science"] is False


# REQ-REPORT-7596 / SCENARIO-REPORT-7596-BLOCKED
def test_check_rows_and_gate_summary_are_exact() -> None:
    passed = audit.check_row("x", "up", "p", "f", [1, 2], 2, "in")
    failed = audit.check_row("y", "up", "p", "f", True, False, "eq")
    assert passed["passed"] is True
    assert audit.blocked_summary([passed, failed])["first_failure"]["check"] == "y"
    with pytest.raises(ValueError, match="unknown_check_op"):
        audit.check_row("x", "up", "p", "f", 1, 1, "bad")
    with pytest.raises(ValueError, match="requires_failed_check"):
        audit.blocked_summary([passed])


# REQ-REPORT-7596 / SCENARIO-REPORT-7596-TERMINAL
def test_required_fields_and_validation_plan_are_scoped(tmp_path: Path) -> None:
    assert set(audit.REQUIRED_PRINCIPLE_FIELDS) <= set(audit.field_principles())
    commands = audit.build_validation_commands(audit.REPO_ROOT, tmp_path / "private")
    assert [command.name for command in commands] == list(audit.AFFECTED_CHECK_NAMES)
    assert all(command.scope != "repository" for command in commands)
    assert any(
        audit.TEST_PATH.as_posix() in command.argv
        for command in commands
        if command.name in {"focused_pytest", "changed_module_coverage"}
    )
    assert all("tests/python" not in command.argv for command in commands)
    terminal = audit.terminal_commands(tmp_path / "candidate.json", audit.REPO_ROOT)
    assert [command.name for command in terminal] == list(audit.TERMINAL_CHECK_NAMES)

    arguments = audit.parse_args(["--root", str(tmp_path), "--date", "20260924"])
    assert arguments.root == tmp_path.resolve()
    with pytest.raises(ValueError, match="run_date"):
        audit.parse_args(["--root", str(tmp_path), "--date", "20260923"])


# REQ-REPORT-7596 / SCENARIO-REPORT-7596-MUTATIONS
def test_unknown_mutation_and_defensive_fixture_errors() -> None:
    fixture = audit.private_fixture()
    with pytest.raises(ValueError, match="unknown_mutation"):
        audit.mutate_private_fixture(fixture, "unknown")
    broken = deepcopy(fixture)
    broken["declared_static_count"] = 99
    broken["declared_online_count"] = 99
    broken["brier_improvement"] = -1.0
    broken["control_predictions"] = list(broken["treatment_predictions"])
    broken["shuffle_pairs"][0]["source_release"] = 9
    broken["gradient_roles"].append("evaluator")
    broken["accepted_update_ids"].append(broken["accepted_update_ids"][0])
    assert set(audit.validate_private_fixture(broken)) == set(audit.MUTATION_FAILURES.values())


# REQ-REPORT-7596 / SCENARIO-REPORT-7596-TERMINAL
def test_all_terminal_reader_failure_branches(tmp_path: Path) -> None:
    checks, receipts = audit.collect_preconditions(audit.REPO_ROOT)
    artifact = audit.build_blocked_artifact(
        audit.REPO_ROOT,
        checks,
        receipts,
        validation_receipts=audit.provisional_validation_receipts(),
        duration_s=0.3,
        phase_spans=[],
    )
    invalid = deepcopy(artifact)
    invalid.update(
        {
            "schema": "bad",
            "milestone": "bad",
            "honest_verdict": "bad",
            "verdict_class": "null",
            "flagged_adversarial": True,
            "MODEL_SPECS": ["bad"],
            "model_invoked": True,
            "inference_substrate_class": "bad",
            "inference_substrate": "bad",
            "field_principles": {},
            "static_audit_ready_score": 1,
            "branch_conclusions": [],
            "gate_check_summary": {},
            "mutation_receipts": [],
            "validation_receipts": [],
            "rows": [{"invented": True}],
            "reproducibility_checksum": "sha256:bad",
        }
    )
    missing_source = deepcopy(invalid["source_artifact_hashes"][0])
    missing_source.update(path="missing-source.json", sha256="sha256:bad", bytes=1)
    invalid["source_artifact_hashes"] = [missing_source]
    with pytest.raises(ValueError) as exc:
        audit.validate_artifact(invalid, root=tmp_path)
    message = str(exc.value)
    for expected in (
        "identity_mismatch",
        "run_identity_mismatch",
        "terminal_prefix_missing",
        "blocked_class_required",
        "terminal_adversarial_outcome_invalid",
        "model_specs_not_empty",
        "current_model_calls_nonzero",
        "substrate_class_mismatch",
        "substrate_mismatch",
        "field_principles_incomplete",
        "blocked_readiness_nonzero",
        "branch_conclusions_incomplete",
        "blocked_gate_summary_invalid",
        "mutation_receipts_invalid",
        "validation_receipts_failed",
        "blocked_rows_must_be_empty",
        "checksum_mismatch",
        "source_missing",
    ):
        assert expected in message

    outside = Path("/outside-exp7596.json")
    assert audit._path_label(outside, tmp_path) == str(outside)
    source = tmp_path / "s.json"
    _write_json(source, {"x": 1})
    receipt = audit.source_receipt(source, tmp_path, "fixture")
    wrong_size = {**receipt, "bytes": receipt["bytes"] + 1}
    with pytest.raises(ValueError, match="source_size_mismatch"):
        audit.authenticate_source_receipt(wrong_size, tmp_path)
    with pytest.raises(ValueError, match="source_missing"):
        audit.authenticate_source_receipt({**receipt, "path": "gone.json"}, tmp_path)

    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_not_object"):
        audit.cold_replay(unreadable, root=audit.REPO_ROOT)

    for upstream, error in (
        ("exp7594-decision-evaluation", "static_absence_disposition_mismatch"),
        ("exp7595-guarded-learning", "online_absence_disposition_mismatch"),
    ):
        changed = deepcopy(artifact)
        target = next(
            row for row in changed["source_artifact_hashes"] if row["upstream"] == upstream
        )
        target["disposition"] = "authenticated_producer"
        changed["reproducibility_checksum"] = audit.reproducibility_checksum(changed)
        candidate = tmp_path / f"{upstream}.json"
        candidate.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match=error):
            audit.independent_replay(candidate, root=audit.REPO_ROOT)
