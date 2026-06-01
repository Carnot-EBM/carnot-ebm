"""Tests for Exp 3643 additivity remeasurement.

Spec: REQ-VERIFY-3643, SCENARIO-VERIFY-3643.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.verify import additivity_second_pair_of_eyes_v4 as exp3643
from carnot.verify.additivity_second_pair_of_eyes_v4 import (
    REQUIRED_ARTIFACT_FIELDS,
    build_artifact,
    conditional_catch_summary,
    negative_tail_calibrated_scores,
    predictions_at_fixed_fpr,
    validate_artifact,
    write_artifact,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _seed_exp3643_fixture(
    root: Path,
    *,
    code_ran: bool = True,
    facts_ran: bool = True,
    at_least_one_nonmath_row_ran: bool = True,
    n_errors: int = 10,
    n_correct: int = 10,
) -> tuple[list[dict], list[dict]]:
    results = root / "results"
    data = root / "data"
    code_rows = [
        {"candidate_code": f"def bad_{idx}(:\n    return {idx}\n", "label": False}
        for idx in range(n_errors)
    ] + [
        {"candidate_code": f"def good_{idx}():\n    return {idx}\n", "label": True}
        for idx in range(n_correct)
    ]
    facts_rows = [
        {
            "answer": f"wrong-{idx}",
            "evidence_passage": f"Evidence supports answer {idx}.",
            "is_hallucination": 1,
            "model_confidence": 0.2,
        }
        for idx in range(n_errors)
    ] + [
        {
            "answer": f"right-{idx}",
            "evidence_passage": f"Evidence supports right-{idx}.",
            "is_hallucination": 0,
            "model_confidence": 0.8,
        }
        for idx in range(n_correct)
    ]
    _write_jsonl(data / "code.jsonl", code_rows)
    _write_jsonl(data / "facts.jsonl", facts_rows)
    _write_json(
        results / "experiment_3641_code_corpus_verifiers_fire_transfer_v3.json",
        {
            "code_corpus_path": "data/code.jsonl",
            "code_verifiers_fire": True,
        },
    )
    _write_json(
        results / "experiment_3640_build_factual_corpus_v3.json",
        {
            "corpus_path_used": "data/facts.jsonl",
            "facts_corpus_validated": True,
        },
    )
    _write_json(
        results / "experiment_3642_corrected_cross_domain_remeasurement_v4.json",
        {
            "at_least_one_nonmath_row_ran": at_least_one_nonmath_row_ran,
            "generalization_table": {
                "code": {
                    "domain": "code",
                    "headroom": code_ran,
                    "ran_or_blocked": "ran" if code_ran else "blocked",
                },
                "facts": {
                    "domain": "facts",
                    "headroom": facts_ran,
                    "ran_or_blocked": "ran" if facts_ran else "blocked",
                },
                "math": {"domain": "math", "ran_or_blocked": "ran"},
            },
        },
    )
    return code_rows, facts_rows


def _confidence_scores(n_errors: int = 10, n_correct: int = 10) -> list[float]:
    return [0.95] * 4 + [0.01] * (n_errors - 4) + [0.90] + [0.05] * (n_correct - 1)


def _ensemble_additive_scores(n_errors: int = 10, n_correct: int = 10) -> list[float]:
    return [0.95] * n_errors + [0.90] + [0.05] * (n_correct - 1)


def test_exp3643_reports_additive_second_pair_and_fusion_win(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3643: corrected runnable rows drive the additivity verdict."""

    _seed_exp3643_fixture(tmp_path, code_ran=True, facts_ran=True)
    artifact = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=3.0,
        score_overrides={
            "code": {
                "ensemble_scores": _ensemble_additive_scores(),
                "confidence_scores": _confidence_scores(),
            },
            "facts": {
                "ensemble_scores": _confidence_scores(),
                "confidence_scores": _confidence_scores(),
            },
        },
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: ensemble_additive_to_confidence_second_pair_of_eyes_real_fusion_wins"
    )
    assert artifact["code_conditional_catch_rate_ensemble_over_confidence"] == 1.0
    assert artifact["factual_conditional_catch_rate_ensemble_over_confidence"] == 0.0
    assert artifact["mcnemar_p_code"] == pytest.approx(0.03125)
    assert artifact["mcnemar_p_factual"] == 1.0
    assert artifact["second_pair_of_eyes_real"] is True
    assert artifact["fusion_beats_confidence_alone"] is True
    assert artifact["fused_detector_auroc"] > artifact["confidence_alone_auroc"]
    assert artifact["fused_detector_recall_at_fixed_fpr"] > artifact[
        "confidence_recall_at_fixed_fpr"
    ]
    assert artifact["n_errors_per_domain"] == {"code": 10, "facts": 10}
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])


def test_exp3643_marks_blocked_domain_not_measured(tmp_path: Path) -> None:
    """REQ-VERIFY-3643: a missing domain does not block a measured non-math row."""

    _seed_exp3643_fixture(tmp_path, code_ran=True, facts_ran=False)
    artifact = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=2.0,
        score_overrides={
            "code": {
                "ensemble_scores": _ensemble_additive_scores(),
                "confidence_scores": _confidence_scores(),
            }
        },
    )

    validate_artifact(artifact)
    assert artifact["code_conditional_catch_rate_ensemble_over_confidence"] == 1.0
    assert artifact["factual_conditional_catch_rate_ensemble_over_confidence"] == "not_measured"
    assert artifact["mcnemar_p_factual"] == "not_measured"
    assert artifact["n_errors_per_domain"] == {"code": 10, "facts": "not_measured"}
    assert artifact["per_domain_additivity"]["facts"]["status"] == "not_measured"


def test_exp3643_blocks_when_exp3642_has_no_nonmath_rows(tmp_path: Path) -> None:
    """REQ-VERIFY-3643: closed upstream gate writes the required blocked verdict."""

    _seed_exp3643_fixture(
        tmp_path,
        code_ran=False,
        facts_ran=False,
        at_least_one_nonmath_row_ran=False,
    )
    artifact = build_artifact(tmp_path, started_s=0.0, now_s=1.0)

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: blocked_no_nonmath_row_ran"
    assert artifact["code_conditional_catch_rate_ensemble_over_confidence"] == "not_measured"
    assert artifact["factual_conditional_catch_rate_ensemble_over_confidence"] == "not_measured"
    assert artifact["fused_detector_auroc"] is None
    assert artifact["fusion_beats_confidence_alone"] is False
    assert artifact["second_pair_of_eyes_real"] is False


def test_exp3643_redundant_when_fusion_does_not_beat_confidence(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-3643: identical signals are classified as redundant."""

    _seed_exp3643_fixture(tmp_path, code_ran=True, facts_ran=True)
    scores = _confidence_scores()
    artifact = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=2.0,
        score_overrides={
            "code": {"ensemble_scores": scores, "confidence_scores": scores},
            "facts": {"ensemble_scores": scores, "confidence_scores": scores},
        },
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: ensemble_redundant_with_confidence_no_additive_value_value_prop_weak"
    )
    assert artifact["fusion_beats_confidence_alone"] is False
    assert artifact["second_pair_of_eyes_real"] is False


def test_exp3643_conditional_summary_handles_edge_cases() -> None:
    """REQ-VERIFY-3643: conditional catch rates expose disagreement honestly."""

    assert conditional_catch_summary([], []) == {
        "baseline_recall": 0.0,
        "ensemble_recall": 0.0,
        "conditional_catch_rate_ensemble_over_confidence": 0.0,
        "conditional_catch_rate_confidence_over_ensemble": 0.0,
        "conditional_catch_rate_ci95": [None, None],
        "confidence_only_count": 0,
        "ensemble_only_count": 0,
        "mcnemar_p": 1.0,
        "n_errors": 0,
    }
    no_miss = conditional_catch_summary([True, True], [True, False])
    assert no_miss["conditional_catch_rate_ensemble_over_confidence"] == 0.0
    assert no_miss["conditional_catch_rate_ci95"] == [None, None]
    with pytest.raises(ValueError, match="same length"):
        conditional_catch_summary([True], [])


def test_exp3643_write_artifact_persists_valid_json(tmp_path: Path) -> None:
    """REQ-VERIFY-3643: the runner entry point writes the terminal artifact."""

    _seed_exp3643_fixture(tmp_path, code_ran=True, facts_ran=False)
    output_path = write_artifact(
        tmp_path,
        output_path="results/custom_exp3643.json",
        started_s=0.0,
        now_s=2.0,
        score_overrides={
            "code": {
                "ensemble_scores": _ensemble_additive_scores(),
                "confidence_scores": _confidence_scores(),
            }
        },
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    validate_artifact(payload)
    assert payload["honest_verdict"].startswith("complete:")


def test_exp3643_helper_edges_and_schema_errors(tmp_path: Path) -> None:
    """REQ-VERIFY-3643: helper edge cases fail closed instead of fabricating metrics."""

    assert predictions_at_fixed_fpr([1, 1], [0.2, 0.1], 0.1) == {
        "threshold": None,
        "fpr": 0.0,
        "recall": 0.0,
        "predictions": [False, False],
        "caught_errors": [],
    }
    with pytest.raises(ValueError, match="no fixed-FPR"):
        predictions_at_fixed_fpr([1, 0], [0.2, 0.1], -0.1)
    assert negative_tail_calibrated_scores([1, 1], [0.2, 0.1]) == [0.0, 0.0]
    assert exp3643._score_or_override({}, "missing", lambda: [1, 2]) == [1.0, 2.0]

    list_path = tmp_path / "list.json"
    list_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        exp3643._read_json_object(list_path)

    _seed_exp3643_fixture(tmp_path, code_ran=True, facts_ran=False)
    artifact = build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=2.0,
        score_overrides={
            "code": {
                "ensemble_scores": _ensemble_additive_scores(),
                "confidence_scores": _confidence_scores(),
            }
        },
    )
    missing = dict(artifact)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        validate_artifact(missing)
    bad_verdict = dict(artifact, honest_verdict="blocked")
    with pytest.raises(ValueError, match="must start"):
        validate_artifact(bad_verdict)
    bad_bool = dict(artifact, fusion_beats_confidence_alone={"value": True})
    with pytest.raises(ValueError, match="bare top-level bool"):
        validate_artifact(bad_bool)
    bad_counts = dict(artifact, n_errors_per_domain=[])
    with pytest.raises(ValueError, match="n_errors_per_domain"):
        validate_artifact(bad_counts)
    bad_duration = dict(artifact, duration_s=-1.0)
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(bad_duration)
