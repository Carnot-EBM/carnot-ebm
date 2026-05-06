"""Tests for Exp 1423 FoVer process reward model v1.

Spec: REQ-VERIFY-1423, SCENARIO-VERIFY-1423.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.reporting import process_reward_model_v1_fover_1508 as mod


def _exp1395(promoted_ids: list[str]) -> dict[str, Any]:
    return {
        "status": "complete",
        "fresh_verified_sample_count": len(promoted_ids),
        "memory_updates": {
            "promoted": [f"dvi_v2:fover:{case_id}" for case_id in promoted_ids],
        },
    }


def _fover_row(case_id: str, label: str, text: str | None = None) -> dict[str, Any]:
    return {
        "question_id": case_id,
        "step_text": text or f"{label} arithmetic step for {case_id}",
        "label": label,
        "source": "unit_fover",
        "verifier": "unit",
    }


def _step_prm_row(case_id: str, label: str, text: str | None = None) -> dict[str, Any]:
    return {
        "question_id": case_id,
        "partial_cot": text or f"{label} partial reasoning for {case_id}",
        "step_label": label,
        "full_cot_correct": label == "correct",
        "prefix_fraction": 1.0,
    }


def _exp1397() -> dict[str, Any]:
    return {
        "certificate_rows": [
            {
                "case_id": "from_certificate",
                "expected_state": "SAT",
                "parseable": True,
                "truthful": True,
            }
        ],
        "generation_rows": [
            {
                "case_id": "from_certificate",
                "reasoning_text": "certificate reasoning says this step is SAT",
            }
        ],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_req_verify_1423_writes_in_progress_artifact_first(tmp_path: Path) -> None:
    """REQ-VERIFY-1423: bootstrap output exists before traces are loaded."""

    out_path = tmp_path / mod.OUTPUT_FILE

    artifact = mod.write_in_progress_artifact(out_path, project_root="/repo")

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert written["status"] == "in_progress"
    assert written["training_traces_used"] == 0
    assert written["step_labels_available"] == 0
    assert written["prmv1_trained"] is False
    assert written["honest_verdict"] == "in_progress"


def test_req_verify_1423_collects_labels_from_fover_prm_and_certificates() -> None:
    """REQ-VERIFY-1423: promoted IDs are joined to local step/certificate labels."""

    promoted = ["from_fover", "from_prm", "from_certificate", "missing"]
    labels, coverage = mod.collect_promoted_step_labels(
        _exp1395(promoted),
        fover_rows=[_fover_row("from_fover", "incorrect")],
        step_prm_rows=[_step_prm_row("from_prm", "correct")],
        exp1397_artifact=_exp1397(),
        expected_promoted_count=4,
    )

    assert [label.case_id for label in labels] == [
        "from_fover",
        "from_prm",
        "from_certificate",
    ]
    assert [label.correct for label in labels] == [False, True, True]
    assert coverage.promoted_traces == 4
    assert coverage.training_traces_used == 3
    assert coverage.missing_trace_labels == 1
    assert coverage.positive_step_labels == 2
    assert coverage.negative_step_labels == 1
    assert {label.label_source for label in labels} == {
        "fover_corpus_label",
        "step_level_prm_training",
        "exp1397_certificate_label",
    }


def test_req_verify_1423_rejects_bad_exp1395_promoted_shape() -> None:
    """REQ-VERIFY-1423: malformed promoted memory does not train silently."""

    with pytest.raises(ValueError, match="memory_updates.promoted must be a list"):
        mod.promoted_case_ids({"memory_updates": {"promoted": "bad"}})

    with pytest.raises(ValueError, match="fresh verified count mismatch"):
        mod.promoted_case_ids(_exp1395(["one"]), expected_count=2)

    with pytest.raises(ValueError, match="contain duplicates"):
        mod.promoted_case_ids(_exp1395(["dup", "dup"]), expected_count=2)


def test_req_verify_1423_metrics_are_tie_aware_and_thresholded() -> None:
    """REQ-VERIFY-1423: AUROC, precision, and recall are deterministic."""

    assert mod.tie_aware_auroc([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert mod.tie_aware_auroc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0
    assert mod.tie_aware_auroc([1, 0], [0.5, 0.5]) == 0.5
    assert mod.tie_aware_auroc([1, 1], [0.1, 0.2]) == 0.5

    metrics = mod.classification_metrics([1, 0, 1, 0], [0.8, 0.7, 0.4, 0.1], threshold=0.5)
    assert metrics == {"precision": 0.5, "recall": 0.5}
    assert mod.classification_metrics([1, 0], [0.2, 0.1], threshold=0.9) == {
        "precision": 0.0,
        "recall": 0.0,
    }


def test_req_verify_1423_defensive_edges_are_explicit(tmp_path: Path) -> None:
    """REQ-VERIFY-1423: malformed inputs and invalid artifacts fail honestly."""

    non_object = tmp_path / "non_object.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact must be a JSON object"):
        mod.load_json(non_object)

    jsonl = tmp_path / "rows.jsonl"
    jsonl.write_text('\nnot-json\n[]\n{"question_id": "ok"}\n', encoding="utf-8")
    assert mod.load_jsonl_rows(jsonl) == [{"question_id": "ok"}]

    labels, coverage = mod.collect_promoted_step_labels(
        _exp1395(["keep", "dup", "bool_case", "step_bool", "numeric_case"]),
        fover_rows=[
            _fover_row("not_promoted", "correct"),
            {"question_id": "keep", "step_text": "", "label": "correct"},
            {"question_id": "keep", "step_text": "unknown label", "label": "maybe"},
            _fover_row("dup", "correct", text="same text"),
            _fover_row("dup", "correct", text="same text"),
            {"question_id": "bool_case", "step_text": "bool label", "is_correct": False},
            {"question_id": "step_bool", "step_text": "step bool label", "step_correct": True},
            {"question_id": "numeric_case", "step_text": "numeric label", "label": 0},
        ],
        step_prm_rows=[
            {
                "question_id": "keep",
                "partial_cot": "valid fallback prefix",
                "step_label": "correct",
                "prefix_fraction": "bad",
            }
        ],
        exp1397_artifact={
            "generation_rows": [],
            "certificate_rows": [
                "bad row",
                {"case_id": "keep", "parseable": False, "expected_state": "SAT"},
            ],
        },
        expected_promoted_count=5,
    )
    assert len(labels) == 5
    assert coverage.training_traces_used == 5
    assert labels[-1].prefix_fraction == 1.0

    insufficient = mod.train_and_evaluate(
        [mod.StepLabel(case_id="only", text="correct only", correct=True)],
        checkpoint_path=tmp_path / "unused.pt",
    )
    assert insufficient.trained is False

    tiny = mod.train_and_evaluate(
        [
            mod.StepLabel(case_id="p", text="valid sat", correct=True),
            mod.StepLabel(case_id="n", text="invalid repair", correct=False),
        ],
        checkpoint_path=tmp_path / "tiny.pt",
        n_epochs=1,
    )
    assert tiny.trained is True

    assert mod._label_value_is_correct(True) is True
    assert mod._label_value_is_correct(1) is True
    assert mod._label_value_is_correct("unknown") is None
    assert mod._best_threshold([], []) == 0.5
    assert (
        mod._honest_verdict(
            True,
            mod.LabelCoverage(
                promoted_traces=2,
                training_traces_used=1,
                missing_trace_labels=1,
                positive_step_labels=1,
                negative_step_labels=1,
            ),
        )
        == "prmv1_trained_on_available_step_labels_with_1_promoted_traces_missing_local_labels"
    )
    assert mod._honest_verdict(
        False,
        mod.LabelCoverage(
            promoted_traces=2,
            training_traces_used=0,
            missing_trace_labels=2,
            positive_step_labels=0,
            negative_step_labels=0,
        ),
    ) == "prmv1_blocked_missing_positive_step_labels_and_negative_step_labels_and_labeled_traces"

    with pytest.raises(AssertionError, match="missing required fields"):
        mod.validate_artifact({})
    bad_status = dict.fromkeys(mod.REQUIRED_ARTIFACT_FIELDS, None)
    bad_status["status"] = "weird"
    with pytest.raises(AssertionError, match="unsupported status"):
        mod.validate_artifact(bad_status)
    complete = {
        "status": "complete",
        "training_traces_used": 1,
        "step_labels_available": 2,
        "prmv1_trained": False,
        "prmv1_auroc": 0.5,
        "prmv1_step_precision": 0.5,
        "prmv1_step_recall": 0.5,
        "checkpoint_path": str(tmp_path / "missing.pt"),
        "honest_verdict": "bad",
    }
    with pytest.raises(AssertionError, match="requires prmv1_trained=true"):
        mod.validate_artifact(complete)
    complete["prmv1_trained"] = True
    complete["prmv1_auroc"] = None
    with pytest.raises(AssertionError, match="requires prmv1_auroc"):
        mod.validate_artifact(complete)
    complete["prmv1_auroc"] = 0.5
    with pytest.raises(AssertionError, match="requires an existing checkpoint"):
        mod.validate_artifact(complete)
    blocked = dict(complete, status="blocked", checkpoint_path=str(tmp_path / "bad.pt"))
    with pytest.raises(AssertionError, match="must not expose a checkpoint"):
        mod.validate_artifact(blocked)


def test_req_verify_1423_trains_feature_classifier_and_checkpoint(tmp_path: Path) -> None:
    """REQ-VERIFY-1423: CPU feature classifier saves replayable model state."""

    labels = [
        mod.StepLabel(case_id=f"correct_{i}", text=f"correct proof {i} valid sat", correct=True)
        for i in range(6)
    ] + [
        mod.StepLabel(case_id=f"wrong_{i}", text=f"wrong proof {i} invalid repair", correct=False)
        for i in range(6)
    ]
    checkpoint_path = tmp_path / "prmv1_checkpoint.pt"

    result = mod.train_and_evaluate(labels, checkpoint_path=checkpoint_path, n_epochs=8)

    assert result.trained is True
    assert checkpoint_path.exists()
    assert 0.0 <= result.auroc <= 1.0
    assert 0.0 <= result.precision <= 1.0
    assert 0.0 <= result.recall <= 1.0
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        assert checkpoint["weights"].shape == (mod.FEATURE_DIM,)
        assert float(checkpoint["threshold"].reshape(-1)[0]) == pytest.approx(result.threshold)


def test_scenario_verify_1423_run_trains_and_writes_complete_artifact(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1423: runner reports held-out metrics and checkpoint path."""

    results = tmp_path / "results"
    results.mkdir()
    exp1395_path = results / mod.EXP1395_FILE
    exp1397_path = results / mod.EXP1397_FILE
    fover_path = tmp_path / "fover.jsonl"
    step_prm_path = tmp_path / "step_prm.jsonl"
    out_path = results / mod.OUTPUT_FILE
    checkpoint_path = tmp_path / "models" / "prmv1_checkpoint.pt"
    promoted = [f"correct_{i}" for i in range(8)] + [f"wrong_{i}" for i in range(8)]
    _write_json(exp1395_path, _exp1395(promoted))
    _write_json(exp1397_path, {"certificate_rows": [], "generation_rows": []})
    _write_jsonl(
        fover_path,
        [
            *[
                _fover_row(case_id, "correct", text=f"valid correct sat reasoning {case_id}")
                for case_id in promoted[:8]
            ],
            *[
                _fover_row(case_id, "incorrect", text=f"invalid wrong repair reasoning {case_id}")
                for case_id in promoted[8:]
            ],
        ],
    )
    _write_jsonl(step_prm_path, [])

    artifact = mod.run(
        exp1395_path=exp1395_path,
        exp1397_path=exp1397_path,
        fover_path=fover_path,
        step_prm_path=step_prm_path,
        out_path=out_path,
        checkpoint_path=checkpoint_path,
        project_root=tmp_path,
        expected_promoted_count=16,
        n_epochs=8,
        tests_run=["pytest targeted"],
    )

    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written == artifact
    mod.validate_artifact(artifact)
    assert artifact["status"] == "complete"
    assert artifact["training_traces_used"] == 16
    assert artifact["step_labels_available"] == 16
    assert artifact["prmv1_trained"] is True
    assert artifact["checkpoint_path"] == str(checkpoint_path)
    assert checkpoint_path.exists()
    assert artifact["tests_run"] == ["pytest targeted"]


def test_scenario_verify_1423_run_blocks_when_labels_are_insufficient(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-1423: missing class labels produce an honest blocked artifact."""

    results = tmp_path / "results"
    results.mkdir()
    exp1395_path = results / mod.EXP1395_FILE
    exp1397_path = results / mod.EXP1397_FILE
    fover_path = tmp_path / "fover.jsonl"
    step_prm_path = tmp_path / "step_prm.jsonl"
    out_path = results / mod.OUTPUT_FILE
    checkpoint_path = tmp_path / "models" / "prmv1_checkpoint.pt"
    promoted = [f"correct_{i}" for i in range(4)]
    _write_json(exp1395_path, _exp1395(promoted))
    _write_json(exp1397_path, {"certificate_rows": [], "generation_rows": []})
    _write_jsonl(fover_path, [_fover_row(case_id, "correct") for case_id in promoted])
    _write_jsonl(step_prm_path, [])

    artifact = mod.run(
        exp1395_path=exp1395_path,
        exp1397_path=exp1397_path,
        fover_path=fover_path,
        step_prm_path=step_prm_path,
        out_path=out_path,
        checkpoint_path=checkpoint_path,
        project_root=tmp_path,
        expected_promoted_count=4,
        n_epochs=2,
    )

    mod.validate_artifact(artifact)
    assert artifact["status"] == "blocked"
    assert artifact["prmv1_trained"] is False
    assert artifact["prmv1_auroc"] is None
    assert artifact["checkpoint_path"] is None
    assert artifact["missing_negative_step_labels"] == 1
    assert not checkpoint_path.exists()
