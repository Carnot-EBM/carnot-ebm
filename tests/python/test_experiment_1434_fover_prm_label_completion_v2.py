"""Tests for Exp 1434 FoVer PRM label completion v2.

Spec: REQ-VERIFY-1434, SCENARIO-VERIFY-1434.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import carnot.reporting.fover_prm_label_completion_v2 as mod


def _exp1395(promoted_ids: list[str]) -> dict[str, Any]:
    return {
        "status": "complete",
        "memory_updates": {
            "promoted": [f"dvi_v2:fover:{case_id}" for case_id in promoted_ids],
        },
    }


def _exp1423(training_traces_used: int, missing_trace_labels: int) -> dict[str, Any]:
    return {
        "status": "complete",
        "training_traces_used": training_traces_used,
        "missing_trace_labels": missing_trace_labels,
        "prmv1_trained": True,
    }


def _fover_row(case_id: str, label: str, text: str) -> dict[str, Any]:
    return {
        "question_id": case_id,
        "step_text": text,
        "label": label,
        "source": "unit_fover",
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_req_verify_1434_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-VERIFY-1434: bootstrap output exists before label recovery."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["missing_labels_before"] == 0
    assert written["missing_labels_filled"] == 0
    assert written["missing_labels_remaining"] == 0
    assert written["prmv2_trained"] is False
    assert written["headline_label_coverage_ready"] is False


def test_req_verify_1434_recovers_only_ordinal_replay_labels(tmp_path: Path) -> None:
    """REQ-VERIFY-1434: duplicate-ID replay recovers labels without inventing them."""

    labels = [
        mod.prm_v1.StepLabel(
            case_id="raw",
            text="raw positive local label",
            correct=True,
            label_source="unit",
        )
    ]
    missing = mod.missing_trace_ids(
        _exp1395(["raw", "raw_1", "absent"]),
        labels,
        expected_promoted_count=3,
    )

    recovery = mod.recover_with_ordinal_replay(
        missing,
        [
            _fover_row("raw", "correct", "first raw row"),
            _fover_row("raw", "incorrect", "second raw row"),
        ],
    )
    ledger_path = tmp_path / mod.LEDGER_FILE
    mod.write_label_blocker_ledger(
        ledger_path,
        missing_ids=missing,
        recovered_labels=recovery.recovered_labels,
        blockers=recovery.blockers,
        project_root="/repo",
    )

    assert [label.case_id for label in recovery.recovered_labels] == ["raw_1"]
    assert recovery.recovered_labels[0].correct is False
    assert recovery.recovered_labels[0].label_source == mod.ORDINAL_REPLAY_LABEL_SOURCE
    assert recovery.blockers == [
        {
            "case_id": "absent",
            "blocker": "no_local_ordinal_replay_source_row",
            "recovery_scope": "local_recovery_scope",
        }
    ]
    ledger = ledger_path.read_text(encoding="utf-8")
    assert "| absent | no_local_ordinal_replay_source_row | local_recovery_scope |" in ledger
    assert "raw_1" in ledger


def test_req_verify_1434_headline_gate_and_validation_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-1434: coverage gates and artifact validation are explicit."""

    assert mod.headline_ready(0, []) is True
    assert (
        mod.headline_ready(
            1,
            [
                {
                    "case_id": "outside",
                    "blocker": "non_fover_trace",
                    "recovery_scope": "outside_local_recovery_scope",
                }
            ],
        )
        is True
    )
    assert (
        mod.headline_ready(
            1,
            [
                {
                    "case_id": "local",
                    "blocker": "no_local_ordinal_replay_source_row",
                    "recovery_scope": "local_recovery_scope",
                }
            ],
        )
        is False
    )

    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact({})

    bad_status = dict.fromkeys(mod.REQUIRED_ARTIFACT_FIELDS, None)
    bad_status["status"] = "weird"
    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(bad_status)

    complete = dict.fromkeys(mod.REQUIRED_ARTIFACT_FIELDS, None)
    complete.update(
        {
            "status": "complete",
            "missing_labels_before": 1,
            "missing_labels_filled": 1,
            "missing_labels_remaining": 0,
            "label_blocker_ledger_path": str(tmp_path / "missing.md"),
            "training_traces_used": 2,
            "prmv2_trained": False,
            "prmv2_auroc": 0.5,
            "prmv2_precision": 0.5,
            "prmv2_recall": 0.5,
            "headline_label_coverage_ready": True,
            "honest_verdict": "bad",
            "checkpoint_path": str(tmp_path / "missing.pt"),
        }
    )
    with pytest.raises(AssertionError, match="requires prmv2_trained=true"):
        mod.validate_artifact(complete)

    complete["prmv2_trained"] = True
    complete["prmv2_auroc"] = None
    with pytest.raises(AssertionError, match="requires prmv2_auroc"):
        mod.validate_artifact(complete)

    complete["prmv2_auroc"] = 0.5
    with pytest.raises(AssertionError, match="requires an existing checkpoint"):
        mod.validate_artifact(complete)

    checkpoint = tmp_path / "exists.pt"
    checkpoint.touch()
    complete["checkpoint_path"] = str(checkpoint)
    with pytest.raises(AssertionError, match="requires an existing label blocker ledger"):
        mod.validate_artifact(complete)

    blocked = dict(complete, status="blocked", checkpoint_path=str(tmp_path / "bad.pt"))
    with pytest.raises(AssertionError, match="must not expose a checkpoint"):
        mod.validate_artifact(blocked)

    empty_ledger = tmp_path / "empty.md"
    mod.write_label_blocker_ledger(
        empty_ledger,
        missing_ids=[],
        recovered_labels=[],
        blockers=[],
        project_root="/repo",
    )
    empty_text = empty_ledger.read_text(encoding="utf-8")
    assert "No labels were recovered." in empty_text
    assert "- none" in empty_text

    blocked_artifact = mod.build_artifact(
        exp1423_artifact=_exp1423(training_traces_used=0, missing_trace_labels=1),
        labels=[],
        missing_ids=["local"],
        recovery=mod.LabelRecovery(
            recovered_labels=[],
            blockers=[
                {
                    "case_id": "local",
                    "blocker": "no_local_ordinal_replay_source_row",
                    "recovery_scope": "local_recovery_scope",
                }
            ],
        ),
        training_result=mod.prm_v1.TrainingResult(
            trained=False,
            auroc=None,
            precision=None,
            recall=None,
            checkpoint_path=None,
            threshold=0.5,
            loss_history=[],
            train_labels_used=0,
            heldout_labels_used=0,
        ),
        ledger_path=empty_ledger,
        started_at="2026-05-06T00:00:00+00:00",
        duration_s=0.0,
        tests_run=[],
        project_root="/repo",
    )
    assert blocked_artifact["status"] == "blocked"
    assert blocked_artifact["prmv2_auroc"] is None
    assert blocked_artifact["honest_verdict"] == "prmv2_blocked_insufficient_trainable_local_labels"
    assert (
        mod._honest_verdict(trained=True, headline=False, missing_remaining=2)
        == "prmv2_trained_with_2_promoted_traces_still_blocked"
    )


def test_scenario_verify_1434_run_retrains_and_writes_complete_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-1434: runner fills ordinal labels and retrains PRM v2."""

    results = tmp_path / "results"
    results.mkdir()
    docs = tmp_path / "docs" / "research"
    exp1395_path = results / mod.EXP1395_FILE
    exp1423_path = results / mod.EXP1423_FILE
    exp1397_path = results / mod.EXP1397_FILE
    fover_path = tmp_path / "fover.jsonl"
    step_prm_path = tmp_path / "step_prm.jsonl"
    out_path = results / mod.OUTPUT_FILE
    ledger_path = docs / mod.LEDGER_FILE
    checkpoint_path = tmp_path / "models" / "prmv2_checkpoint.pt"
    promoted = ["p0", "p0_1", "p1", "p1_1", "n0", "n0_1", "n1", "n1_1"]
    _write_json(exp1395_path, _exp1395(promoted))
    _write_json(exp1423_path, _exp1423(training_traces_used=4, missing_trace_labels=4))
    _write_json(exp1397_path, {"certificate_rows": [], "generation_rows": []})
    _write_jsonl(
        fover_path,
        [
            _fover_row("p0", "correct", "valid correct proof p0 first"),
            _fover_row("p0", "correct", "valid correct proof p0 second"),
            _fover_row("p1", "correct", "valid correct proof p1 first"),
            _fover_row("p1", "correct", "valid correct proof p1 second"),
            _fover_row("n0", "incorrect", "invalid wrong repair n0 first"),
            _fover_row("n0", "incorrect", "invalid wrong repair n0 second"),
            _fover_row("n1", "incorrect", "invalid wrong repair n1 first"),
            _fover_row("n1", "incorrect", "invalid wrong repair n1 second"),
        ],
    )
    _write_jsonl(step_prm_path, [])

    artifact = mod.run(
        exp1395_path=exp1395_path,
        exp1423_path=exp1423_path,
        exp1397_path=exp1397_path,
        fover_path=fover_path,
        step_prm_path=step_prm_path,
        out_path=out_path,
        ledger_path=ledger_path,
        checkpoint_path=checkpoint_path,
        project_root=tmp_path,
        expected_promoted_count=8,
        n_epochs=2,
        tests_run=["pytest targeted"],
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["missing_labels_before"] == 4
    assert artifact["missing_labels_filled"] == 4
    assert artifact["missing_labels_remaining"] == 0
    assert artifact["training_traces_used"] == 8
    assert artifact["prmv2_trained"] is True
    assert artifact["headline_label_coverage_ready"] is True
    assert artifact["label_blocker_ledger_path"] == str(ledger_path)
    assert artifact["tests_run"] == ["pytest targeted"]
    assert checkpoint_path.exists()
    assert "No unrecovered labels remain." in ledger_path.read_text(encoding="utf-8")
