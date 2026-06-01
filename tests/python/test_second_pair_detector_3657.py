"""Tests for the deployable second-pair detector.

Spec: REQ-SPOE-3657, REQ-SPOE-3657-ARTIFACT,
      SCENARIO-SPOE-3657, SCENARIO-SPOE-3658.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot.pipeline.second_pair_detector import (
    CalibratedFusedDetector,
    LabeledDetectorExample,
    brier_score,
    build_artifact,
    build_artifact_from_examples,
    expected_calibration_error,
    load_cached_labeled_examples,
    operating_points_at_fixed_fpr,
    stratified_train_holdout,
    tie_aware_auroc,
    validate_artifact,
    write_artifact,
)


def _synthetic_examples(outcome: str) -> list[LabeledDetectorExample]:
    examples: list[LabeledDetectorExample] = []
    if outcome == "blocked":
        return examples
    for idx in range(80):
        label = 1 if idx < 40 else 0
        if outcome == "fusion_wins":
            ensemble = 0.90 - 0.002 * idx if label else 0.10 + 0.001 * (idx - 40)
            confidence = 0.50
        elif outcome == "fusion_redundant":
            confidence = 0.90 - 0.002 * idx if label else 0.10 + 0.001 * (idx - 40)
            ensemble = confidence
        else:  # pragma: no cover - guarded by parametrization choices
            raise ValueError(outcome)
        examples.append(
            LabeledDetectorExample(
                domain="synthetic",
                label=label,
                ensemble_energy=ensemble,
                confidence_error=confidence,
                example_id=f"{outcome}-{idx}",
            )
        )
    return examples


@pytest.mark.parametrize(
    ("outcome", "expected_verdict", "expected_win"),
    [
        (
            "fusion_wins",
            "complete: deployable_second_pair_of_eyes_detector_built_fusion_wins_calibrated",
            True,
        ),
        (
            "fusion_redundant",
            "complete: deployable_detector_built_fusion_redundant_with_confidence_product_value_weak",
            False,
        ),
        ("blocked", "complete: blocked_no_labeled_corpus_for_fusion", False),
    ],
)
def test_req_spoe_3657_honest_synthetic_outcomes(
    outcome: str,
    expected_verdict: str,
    expected_win: bool,
) -> None:
    """SCENARIO-SPOE-3657: outcomes are classified from measured synthetic data."""

    artifact = build_artifact_from_examples(
        _synthetic_examples(outcome),
        started_s=1.0,
        now_s=3.5,
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert artifact["fusion_beats_confidence_alone"] is expected_win
    assert type(artifact["fusion_beats_confidence_alone"]) is bool
    assert set(artifact["field_principles"]) >= {
        "honest_verdict",
        "fused_detector_auroc",
        "confidence_alone_auroc",
        "calibration_brier_ece",
    }
    if outcome == "blocked":
        assert artifact["fused_detector_auroc"] == {}
        assert artifact["calibration_brier_ece"] == {}
    else:
        assert "synthetic" in artifact["fused_detector_auroc"]
        assert "synthetic" in artifact["confidence_alone_auroc"]
        assert "synthetic" in artifact["ensemble_alone_auroc"]
        assert "synthetic" in artifact["calibration_brier_ece"]


def test_req_spoe_3657_detector_fits_both_features_and_calibrates() -> None:
    """REQ-SPOE-3657: fitted detector emits calibrated probabilities from two features."""

    detector = CalibratedFusedDetector(max_iter=600, learning_rate=0.25)
    detector.fit(_synthetic_examples("fusion_wins"))

    probs = detector.predict_proba(_synthetic_examples("fusion_wins")[:8])

    assert detector.feature_names == ("ensemble_energy", "confidence_error")
    assert detector.coef_ is not None
    assert len(detector.coef_) == 2
    assert all(0.0 <= prob <= 1.0 for prob in probs)
    assert max(probs) > min(probs)

    assert detector.predict_proba([]) == []
    with pytest.raises(ValueError, match="both positive and negative"):
        CalibratedFusedDetector().fit(_synthetic_examples("fusion_wins")[:4])
    with pytest.raises(ValueError, match="must be fitted"):
        CalibratedFusedDetector().predict_proba(_synthetic_examples("fusion_wins")[:1])


def test_scenario_spoe_3658_operating_points_are_per_domain() -> None:
    """SCENARIO-SPOE-3658: fixed-FPR recall table drives the operating point."""

    artifact = build_artifact_from_examples(
        _synthetic_examples("fusion_wins"),
        started_s=0.0,
        now_s=1.0,
    )
    table = artifact["recall_at_fixed_fpr_table"]["synthetic"]
    point = artifact["operating_points"]["synthetic"]

    assert set(table) == {"0.05", "0.10", "0.20"}
    assert point["fpr_budget"] == 0.10
    assert point["threshold"] == table["0.10"]["fused_threshold"]
    assert table["0.10"]["fused_recall"] >= table["0.10"]["confidence_recall"]


def test_req_spoe_3657_operating_points_edge_cases() -> None:
    """REQ-SPOE-3657: operating-point helper handles blocked and invalid inputs."""

    assert operating_points_at_fixed_fpr([1, 1], [0.2, 0.1], [0.05])["0.05"] == {
        "threshold": None,
        "actual_fpr": 0.0,
        "recall": 0.0,
    }
    with pytest.raises(ValueError, match="same length"):
        operating_points_at_fixed_fpr([1, 0], [0.2], [0.1])
    assert tie_aware_auroc([1, 1], [0.1, 0.2]) == 0.5
    assert brier_score([], []) == 0.0
    assert expected_calibration_error([], []) == 0.0
    train, holdout = stratified_train_holdout(
        [LabeledDetectorExample("tiny", 1, 0.5, 0.5, "only")],
        seed=3657,
    )
    assert len(train) == 1
    assert holdout == []


def test_req_spoe_3657_artifact_validation_and_io(tmp_path: Path) -> None:
    """REQ-SPOE-3657-ARTIFACT: artifact schema is validated and written."""

    output = write_artifact(
        tmp_path,
        output_path="results/exp3657.json",
        examples=_synthetic_examples("fusion_wins"),
        started_s=0.0,
        now_s=1.0,
        tests_run=["pytest synthetic"],
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_artifact(payload)
    assert payload["tests_run"] == ["pytest synthetic"]

    missing = dict(payload)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        validate_artifact(missing)
    bad_bool = dict(payload, fusion_beats_confidence_alone={"value": True})
    with pytest.raises(ValueError, match="bare top-level bool"):
        validate_artifact(bad_bool)
    bad_duration = dict(payload, duration_s=-1.0)
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(bad_duration)
    bad_verdict = dict(payload, honest_verdict="complete: unexpected")
    with pytest.raises(ValueError, match="terminal verdict"):
        validate_artifact(bad_verdict)

    blocked_output = write_artifact(
        tmp_path,
        output_path="results/blocked_exp3657.json",
        started_s=0.0,
        now_s=1.0,
    )
    blocked = json.loads(blocked_output.read_text(encoding="utf-8"))
    assert blocked["honest_verdict"] == "complete: blocked_no_labeled_corpus_for_fusion"


def test_req_spoe_3657_missing_cached_corpora_blocks(tmp_path: Path) -> None:
    """REQ-SPOE-3657: absent cached corpora produce the blocked honest outcome."""

    examples, status = load_cached_labeled_examples(tmp_path)
    artifact = build_artifact_from_examples(examples, started_s=0.0, now_s=1.0)

    assert examples == []
    assert status["math"]["status"] == "missing"
    assert status["code"]["status"] == "missing"
    assert artifact["honest_verdict"] == "complete: blocked_no_labeled_corpus_for_fusion"


def test_req_spoe_3657_cached_loader_scores_math_and_code(tmp_path: Path) -> None:
    """REQ-SPOE-3657: cached loader derives labels, ensemble energy, and confidence."""

    data = tmp_path / "data"
    results = tmp_path / "results"
    data.mkdir()
    results.mkdir()
    (data / "fover_corpus_v4.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "m1",
                    "step_text": "1 + 1 = 2",
                    "label": "correct",
                    "confidence": 1.0,
                },
                {
                    "question_id": "m2",
                    "step_text": "1 + 1 = 3",
                    "label": "incorrect",
                    "confidence": "bad",
                },
                ["ignored"],
            ]
        ),
        encoding="utf-8",
    )
    (results / "experiment_3641_code_corpus_verifiers_fire_transfer_v3.json").write_text(
        json.dumps({"code_corpus_path": "data/code.jsonl"}),
        encoding="utf-8",
    )
    (data / "code.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"candidate_code": "def ok():\n    return 1", "label": True}),
                json.dumps({"candidate_code": "def bad(:", "label": False}),
                "",
                json.dumps([1, 2, 3]),
            ]
        ),
        encoding="utf-8",
    )

    examples, status = load_cached_labeled_examples(
        tmp_path,
        score_overrides={
            "math": {
                "ensemble_scores": [0.1, 0.9, 0.4],
                "confidence_scores": [0.0, 0.5, 0.2],
            },
            "code": {"ensemble_scores": [0.2, 0.8], "confidence_scores": [0.1, 0.7]},
        },
    )

    assert status["math"]["status"] == "loaded"
    assert status["code"]["status"] == "loaded"
    assert {(example.domain, example.label) for example in examples} == {
        ("math", 0),
        ("math", 1),
        ("code", 0),
        ("code", 1),
    }

    real_code_root = tmp_path / "real_code"
    (real_code_root / "data").mkdir(parents=True)
    (real_code_root / "data/code_verification_corpus_v1.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"candidate_code": "def ok():\n    return 1", "label": True}),
                json.dumps({"candidate_code": "def bad(:", "label": False}),
            ]
        ),
        encoding="utf-8",
    )
    real_examples, real_status = load_cached_labeled_examples(real_code_root)
    assert real_status["code"]["status"] == "loaded"
    assert len(real_examples) == 2


def test_req_spoe_3657_cached_loader_blocks_empty_files(tmp_path: Path) -> None:
    """REQ-SPOE-3657: empty cached files do not fabricate rows."""

    (tmp_path / "data").mkdir()
    (tmp_path / "data/fover_corpus_v4.json").write_text("[]", encoding="utf-8")
    (tmp_path / "data/code_verification_corpus_v1.jsonl").write_text("", encoding="utf-8")

    examples, status = load_cached_labeled_examples(tmp_path)

    assert examples == []
    assert status["math"]["status"] == "blocked"
    assert status["code"]["status"] == "blocked"


def test_req_spoe_3657_build_artifact_skips_unevaluable_holdout_domain() -> None:
    """REQ-SPOE-3657: one-class held-out domains are skipped, not counted as wins."""

    examples = _synthetic_examples("fusion_wins")
    examples.extend(
        LabeledDetectorExample("one_class", 1, 0.8 + idx * 0.01, 0.5, f"one-{idx}")
        for idx in range(6)
    )

    artifact = build_artifact_from_examples(examples, started_s=0.0, now_s=1.0)

    assert "synthetic" in artifact["fused_detector_auroc"]
    assert "one_class" not in artifact["fused_detector_auroc"]


def test_req_spoe_3657_build_artifact_loads_from_root_with_overrides(tmp_path: Path) -> None:
    """REQ-SPOE-3657-ARTIFACT: build_artifact can use the cached-corpus path."""

    (tmp_path / "data").mkdir()
    (tmp_path / "data/fover_corpus_v4.json").write_text(
        json.dumps(
            [
                {"question_id": "m1", "step_text": "ok", "label": "correct", "confidence": 1.0},
                {"question_id": "m2", "step_text": "bad", "label": "incorrect", "confidence": 1.0},
            ]
        ),
        encoding="utf-8",
    )

    artifact = build_artifact(tmp_path, started_s=0.0, now_s=1.0)

    assert artifact["corpus_status"]["math"]["status"] == "loaded"
    assert artifact["honest_verdict"] == "complete: blocked_no_labeled_corpus_for_fusion"
