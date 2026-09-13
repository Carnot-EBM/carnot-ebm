"""Verify the cold admission audit required by REQ-CL-7283."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import sys

import pytest

from carnot import experiment_7282_v640_admission_learning as learning
from carnot import experiment_7283_v640_admission_audit as audit


def test_req_cl_7283_authenticates_complete_null_inputs(tmp_path: Path) -> None:
    """SCENARIO-CL-7283-PRECONDITIONS: a complete null remains auditable."""

    paths = audit.ExperimentPaths.under(tmp_path)
    checks, hashes, fixture_artifact, upstream = audit.collect_preconditions(audit.REPO_ROOT, paths)

    assert audit.gate_summary(checks)["passed"] is True
    assert fixture_artifact["admission_fixture_ready_score"] == 1
    assert upstream["admission_run_complete_score"] == 1
    assert upstream["admission_value_score"] == 0
    assert hashes[str(audit.REPO_ROOT / audit.DEFAULT_LEARNING_ARTIFACT)] == (
        audit.EXPECTED_LEARNING_SHA256
    )
    assert audit.MODEL_SPECS == []
    assert audit.MODEL_INVOKED is False
    assert audit.INVOCATION_COUNTS == {
        "attempted_model_loads": 0,
        "completed_model_loads": 0,
        "attempted_generation_calls": 0,
        "completed_generation_calls": 0,
        "usable_answers": 0,
    }


def test_external_absence_is_terminal_blocked(tmp_path: Path) -> None:
    """SCENARIO-CL-7283-PRECONDITIONS: an absent input blocks with no rows."""

    paths = audit.ExperimentPaths.under(tmp_path / "out")
    checks, hashes, fixture_artifact, upstream = audit.collect_preconditions(
        audit.REPO_ROOT,
        paths,
        learning_path=tmp_path / "missing.json",
    )
    artifact = audit.build_blocked_artifact(
        checks,
        hashes,
        fixture_artifact,
        upstream,
        ("prospective-01",),
        started_at="2026-09-13T00:00:00+00:00",
        duration_s=0.01,
    )

    assert artifact["status"] == "blocked"
    assert artifact["rows"] == []
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["admission_audit_complete_score"] == 0
    assert artifact["admission_promotion_score"] == 0
    assert artifact["gate_check_summary"]["first_failure"]["observed_value"] is None
    assert audit.validate_artifact(artifact) == []


def test_cold_reducer_preserves_stream_numerators_and_null_arms() -> None:
    """SCENARIO-CL-7283-REDUCTION: raw evidence controls every numerator."""

    result = audit.audit_raw_evidence(
        audit.REPO_ROOT,
        stream_ids=("prospective-01",),
    )

    assert result["process_receipt"]["fresh_process"] is True
    assert len(result["rows"]) == len(learning.ARMS)
    assert {row["arm"] for row in result["rows"]} == set(learning.ARMS)
    assert all(row["opportunity_count"] == learning.MAX_OPPORTUNITIES for row in result["rows"])
    assert result["prequential_error_count"] == 0
    assert result["opportunity_error_count"] == 0
    assert result["stream_numerator_rows"]
    assert any(row["accepted_update_count"] == 0 for row in result["rows"])


def test_opportunity_reduction_keeps_zero_admission_and_no_headroom() -> None:
    """SCENARIO-CL-7283-REDUCTION: all-reject safety cannot erase misses."""

    rows = [
        {
            "stream_id": "s1",
            "arm": "paired_gated",
            "opportunity_index": 1,
            "decision": "reject",
            "harmful_admission": False,
            "missed_beneficial_opportunity": True,
            "zero_available_gain": False,
            "available_gain": 2,
            "admitted_state_change": False,
            "acquisition_label_cost": 8,
            "admission_label_cost": 8,
            "alpha": 0.00625,
            "nomination_case_ids_sha256": "sha256:n",
            "admission_case_ids_sha256": "sha256:a",
            "label_overlap_count": 0,
            "memory_bytes": 10,
            "censored": False,
        },
        {
            "stream_id": "s1",
            "arm": "paired_gated",
            "opportunity_index": 2,
            "decision": "reject",
            "harmful_admission": False,
            "missed_beneficial_opportunity": False,
            "zero_available_gain": True,
            "available_gain": 0,
            "admitted_state_change": False,
            "acquisition_label_cost": 8,
            "admission_label_cost": 8,
            "alpha": 0.00625,
            "nomination_case_ids_sha256": "sha256:n2",
            "admission_case_ids_sha256": "sha256:a2",
            "label_overlap_count": 0,
            "memory_bytes": 11,
            "censored": False,
        },
    ]

    reduced = audit.reduce_opportunities(rows)
    assert reduced["opportunity_denominator"] == 2
    assert reduced["admission_count"] == 0
    assert reduced["harmful_admission_count"] == 0
    assert reduced["missed_beneficial_opportunity_count"] == 1
    assert reduced["zero_available_gain_count"] == 1
    assert reduced["no_headroom_count"] == 1
    assert reduced["paid_query_count"] == 32
    assert reduced["nomination_admission_overlap_count"] == 0


def test_lifecycle_controls_and_causal_deletion_preserve_chronology(tmp_path: Path) -> None:
    """SCENARIO-CL-7283-CONTROLS and CAUSAL: attacks fail and effects are later."""

    controls = audit.run_mutation_controls(tmp_path / "controls")
    interventions = audit.build_causal_intervention_rows()

    assert {row["control"] for row in controls} == set(audit.CONTROL_NAMES)
    assert all(row["passed"] is True for row in controls)
    assert all(row["parent_bytes_preserved"] is True for row in controls)
    assert any(row["accepted_update_deleted"] is True for row in interventions)
    assert any(row["later_changed_prediction_count"] > 0 for row in interventions)
    assert any(row["later_changed_prediction_count"] == 0 for row in interventions)
    assert all(row["pre_commit_changed_prediction_count"] == 0 for row in interventions)


def test_e2e_has_own_statistical_admission_receipt(tmp_path: Path) -> None:
    """SCENARIO-CL-7283-E2E: update, reload, reject, retain, and rollback run."""

    rows = audit.run_e2e_controls(tmp_path / "e2e")

    assert {row["stage"] for row in rows} == set(audit.E2E_STAGES)
    assert all(row["passed"] is True for row in rows)
    assert all(row["smgi_certificate_claimed"] is False for row in rows)
    assert (
        next(row for row in rows if row["stage"] == "model_weight_immutability")["observed"] is True
    )


def test_terminal_scores_separate_safety_causality_and_efficacy() -> None:
    """SCENARIO-CL-7283-TERMINAL: a complete efficacy null stays complete."""

    safety = {"passed": True, "verdict": "pass"}
    causal = {"passed": True, "verdict": "influence_observed"}
    opportunity = {"passed": True, "verdict": "complete"}

    null = audit.derive_terminal_scores(0, safety, causal, opportunity)
    promoted = audit.derive_terminal_scores(1, safety, causal, opportunity)
    failed = audit.derive_terminal_scores(1, {"passed": False}, causal, opportunity)

    assert null == (
        1,
        0,
        "null",
        "complete_null: cold admission audit completed; upstream efficacy value gate failed",
    )
    assert promoted[0:3] == (1, 1, "circular_positive")
    assert failed[0:3] == (1, 0, "null")


def test_terminal_artifact_cold_validates_and_detects_tampering(tmp_path: Path) -> None:
    """REQ-CL-7283 seals a complete one-stream audit and rejects changed evidence."""

    paths = audit.ExperimentPaths.under(tmp_path / "audit")
    artifact = audit.build_and_seal(
        audit.REPO_ROOT,
        paths,
        stream_ids=("prospective-01",),
        progress=True,
    )
    assert artifact["status"] == "complete"
    assert artifact["admission_audit_complete_score"] == 1
    assert artifact["admission_promotion_score"] == 0
    assert artifact["verdict_class"] == "null"
    assert artifact["mechanical_safety_verdict"]["passed"] is True
    assert artifact["causal_influence_verdict"]["passed"] is True
    assert artifact["efficacy_verdict"]["passed"] is False
    assert (
        audit.validate_artifact(
            artifact,
            repo_root=audit.REPO_ROOT,
            expected_stream_ids=("prospective-01",),
            check_files=True,
        )
        == []
    )

    changed = deepcopy(artifact)
    changed["opportunity_reduction"][0]["opportunity_denominator"] += 1
    assert "checksum" in audit.validate_artifact(changed)

    artifact = audit.attach_validation_receipts(
        artifact,
        [
            {
                "command": "focused-test",
                "exit_code": 0,
                "classification": "passed",
                "duration_s": 0.1,
                "log_sha256": "sha256:" + "0" * 64,
            }
        ],
    )
    audit.write_artifact(
        paths.artifact,
        artifact,
        repo_root=audit.REPO_ROOT,
        expected_stream_ids=("prospective-01",),
    )
    assert json.loads(paths.artifact.read_text(encoding="utf-8"))["status"] == "complete"


def test_defensive_validation_and_cli_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-CL-7283 rejects malformed evidence and supports private worker modes."""

    with pytest.raises(ValueError, match="validation_receipt_schema"):
        audit.attach_validation_receipts({}, [{"command": "missing fields"}])
    with pytest.raises(ValueError, match="empty_opportunity_rows"):
        audit.reduce_opportunities([])
    assert audit._receipt_matches(tmp_path, {}) is False
    assert (
        audit.main(["--date", audit.RUN_DATE, "--e2e-worker", "--output-root", str(tmp_path)]) == 0
    )
    monkeypatch.setattr(
        audit,
        "_audit_raw_evidence_impl",
        lambda _root, _selected: {"rows": []},
    )
    assert (
        audit.main(
            [
                "--date",
                audit.RUN_DATE,
                "--audit-worker",
                "--stream-ids",
                "prospective-01",
            ]
        )
        == 0
    )
    assert audit.RESULT_PREFIX in capsys.readouterr().out
    with pytest.raises(SystemExit, match="run_date_must_be"):
        audit.main(["--date", "20260912"])


def test_blocked_build_unknown_status_and_invalid_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-CL-7283-PRECONDITIONS: internal builders remain fail closed."""

    paths = audit.ExperimentPaths.under(tmp_path)
    original = audit.collect_preconditions

    def failed(
        *args: object, **kwargs: object
    ) -> tuple[
        list[dict[str, object]],
        dict[str, str | None],
        dict[str, object],
        dict[str, object],
    ]:
        checks, hashes, fixture_artifact, upstream = original(*args, **kwargs)
        checks[0] = {**checks[0], "observed_value": False, "passed": False}
        return checks, hashes, fixture_artifact, upstream

    monkeypatch.setattr(audit, "collect_preconditions", failed)
    blocked = audit.build_and_seal(
        audit.REPO_ROOT,
        paths,
        stream_ids=("prospective-01",),
        progress=True,
    )
    assert blocked["status"] == "blocked"

    unknown = deepcopy(blocked)
    unknown["status"] = "partial"
    unknown["reproducibility_checksum"] = audit.reproducibility_checksum(unknown)
    assert "status" in audit.validate_artifact(unknown)
    with pytest.raises(ValueError, match="artifact_validation_failed"):
        audit.write_artifact(tmp_path / "invalid.json", unknown)


def test_validation_command_manifest_is_focused(tmp_path: Path) -> None:
    """REQ-CL-7283 keeps coverage, affected tests, E2E, and lints bounded."""

    commands = audit._validation_commands(tmp_path / "candidate.json")
    text = [" ".join(command) for command in commands]

    assert any("--fail-under=100" in command for command in text)
    assert any("test_experiment_7283_v640_admission_audit.py" in command for command in text)
    assert any("adversarial_verify.py" in command for command in text)
    assert any("verdict_row_consistency_lint.py" in command for command in text)
    assert all("tests/python -q" not in command for command in text)


def test_selected_rows_and_worker_failures_are_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-CL-7283-REDUCTION: failed child evidence never becomes zero."""

    raw = tmp_path / "rows.jsonl"
    raw.write_text('{"stream_id":"other"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="selected_raw_rows_unavailable"):
        audit._read_selected(raw, ("wanted",))
    with pytest.raises(TimeoutError, match="timeout"):
        audit._spawn_worker(
            [sys.executable, "-c", "import time; time.sleep(1)"],
            "timeout",
            timeout_s=-1,
        )
    with pytest.raises(RuntimeError, match="failed"):
        audit._spawn_worker([sys.executable, "-c", "raise SystemExit(2)"], "failed")
    with pytest.raises(RuntimeError, match="missing_result"):
        audit._spawn_worker([sys.executable, "-c", "print('ordinary output')"], "missing")


def test_build_rejects_its_own_invalid_terminal_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7283 refuses publication when its terminal check fails."""

    worker = {
        "rows": [],
        "opportunity_reduction": [],
        "stream_numerator_rows": [],
        "prequential_error_count": 0,
        "opportunity_error_count": 0,
        "producer_rows_match": True,
        "process_receipt": {
            "command": "cold-worker",
            "exit_code": 0,
            "duration_s": 0.01,
            "log_sha256": "sha256:" + "1" * 64,
        },
    }
    controls = [
        {"control": name, "passed": True, "parent_bytes_preserved": True}
        for name in audit.CONTROL_NAMES
    ]
    e2e = [
        {"stage": stage, "passed": True, "smgi_certificate_claimed": False}
        for stage in audit.E2E_STAGES
    ]
    interventions = [
        {
            "pre_commit_changed_prediction_count": 0,
            "later_changed_prediction_count": value,
        }
        for value in (1, 0)
    ]
    monkeypatch.setattr(audit, "audit_raw_evidence", lambda *_args, **_kwargs: worker)
    monkeypatch.setattr(audit, "run_mutation_controls", lambda _root: controls)
    monkeypatch.setattr(audit, "run_e2e_controls", lambda _root: e2e)
    monkeypatch.setattr(audit, "build_causal_intervention_rows", lambda: interventions)
    monkeypatch.setattr(audit, "validate_artifact", lambda *_args, **_kwargs: ["forced"])

    with pytest.raises(ValueError, match="artifact_validation_failed:forced"):
        audit.build_and_seal(
            audit.REPO_ROOT,
            audit.ExperimentPaths.under(tmp_path),
            stream_ids=("prospective-01",),
            progress=True,
        )


def test_main_validates_and_dispatches_terminal_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-CL-7283 validates the candidate before its atomic terminal write."""

    paths = audit.ExperimentPaths.under(tmp_path)
    checks, hashes, fixture_artifact, upstream = audit.collect_preconditions(
        audit.REPO_ROOT,
        paths,
        learning_path=tmp_path / "missing.json",
    )
    blocked = audit.build_blocked_artifact(
        checks,
        hashes,
        fixture_artifact,
        upstream,
        ("prospective-01",),
        started_at="2026-09-13T00:00:00+00:00",
        duration_s=0.01,
    )
    audit._atomic_write(paths.artifact, audit._canonical_bytes(blocked))
    assert (
        audit.main(["--date", audit.RUN_DATE, "--validate", "--artifact-path", str(paths.artifact)])
        == 0
    )
    changed = deepcopy(blocked)
    changed["duration_s"] = 99
    audit._atomic_write(paths.artifact, audit._canonical_bytes(changed))
    with pytest.raises(SystemExit, match="artifact_validation_failed:checksum"):
        audit.main(["--date", audit.RUN_DATE, "--validate", "--artifact-path", str(paths.artifact)])

    written: list[str] = []
    monkeypatch.setattr(audit, "build_and_seal", lambda *_args, **_kwargs: blocked)
    monkeypatch.setattr(
        audit,
        "write_artifact",
        lambda *_args, **_kwargs: {"sha256": "sha256:" + "2" * 64},
    )
    assert (
        audit.main(
            [
                "--date",
                audit.RUN_DATE,
                "--output-root",
                str(tmp_path / "blocked"),
                "--stream-ids",
                "prospective-01",
            ]
        )
        == 0
    )

    complete = {"status": "complete", "validation_receipts": []}
    monkeypatch.setattr(audit, "build_and_seal", lambda *_args, **_kwargs: complete)
    monkeypatch.setattr(audit, "_validation_commands", lambda _candidate: [["focused"]])
    monkeypatch.setattr(
        audit.fixture,
        "_command_receipt",
        lambda _command: {
            "command": "focused",
            "exit_code": 0,
            "classification": "passed",
            "duration_s": 0.01,
            "log_sha256": "sha256:" + "3" * 64,
        },
    )
    monkeypatch.setattr(audit, "_atomic_write", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(audit, "_write_evidence", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        audit,
        "attach_validation_receipts",
        lambda artifact, receipts: {**artifact, "validation_receipts": list(receipts)},
    )
    monkeypatch.setattr(
        audit,
        "write_artifact",
        lambda *_args, **_kwargs: written.append("final") or {"sha256": "sha256:" + "4" * 64},
    )
    complete_args = [
        "--date",
        audit.RUN_DATE,
        "--output-root",
        str(tmp_path / "complete"),
        "--stream-ids",
        "prospective-01",
    ]
    assert audit.main(complete_args) == 0
    assert written == ["final"]

    monkeypatch.setattr(
        audit.fixture,
        "_command_receipt",
        lambda _command: {
            "command": "focused",
            "exit_code": 1,
            "classification": "failed",
            "duration_s": 0.01,
            "log_sha256": "sha256:" + "5" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="focused_validation_failed"):
        audit.main(complete_args)
