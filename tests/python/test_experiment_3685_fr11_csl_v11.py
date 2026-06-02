"""Tests for Exp 3685 FR-11 continuous self-learning v11.

Spec: REQ-LEARN-3685, SCENARIO-LEARN-3685.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.fr11 import continuous_self_learning_v11 as exp3685
from carnot.fr11.continuous_self_learning_v11 import (
    BLOCKED_VERDICT,
    NO_GAIN_VERDICT,
    REQUIRED_ARTIFACT_FIELDS,
    SUCCESS_VERDICT,
    build_artifact,
    build_artifact_from_scores,
    select_honest_verdict,
    validate_artifact,
    write_artifact,
)

VERIFIER_NAMES = exp3685.VERIFIER_NAMES


def _score_fixture(outcome: str) -> tuple[list[int], dict[str, list[float]]]:
    if outcome == "blocked":
        return [], {}

    rng = np.random.default_rng(11 if outcome == "drift_recovered_no_collapse" else 17)
    n_each = 180
    labels_pre = np.asarray([0, 1] * (n_each // 2), dtype=np.int64)
    labels_post = np.asarray([0, 1] * (n_each // 2), dtype=np.int64)
    rng.shuffle(labels_pre)
    rng.shuffle(labels_post)

    if outcome == "drift_recovered_no_collapse":
        pre = np.column_stack(
            [
                0.18 + 0.18 * labels_pre + rng.normal(0.0, 0.035, n_each),
                0.10 + 0.36 * labels_pre + rng.normal(0.0, 0.035, n_each),
                0.36 - 0.25 * labels_pre + rng.normal(0.0, 0.045, n_each),
                0.06 + rng.normal(0.0, 0.020, n_each),
            ]
        )
        post = np.column_stack(
            [
                0.36 + 0.28 * labels_post + rng.normal(0.0, 0.045, n_each),
                0.65 - 0.18 * labels_post + rng.normal(0.0, 0.055, n_each),
                0.34 + rng.normal(0.0, 0.055, n_each),
                0.22 + 0.57 * labels_post + rng.normal(0.0, 0.045, n_each),
            ]
        )
    elif outcome == "no_gain_over_v10":
        pre = np.column_stack(
            [
                0.16 + 0.35 * labels_pre + rng.normal(0.0, 0.035, n_each),
                0.13 + 0.30 * labels_pre + rng.normal(0.0, 0.040, n_each),
                0.30 - 0.18 * labels_pre + rng.normal(0.0, 0.040, n_each),
                0.08 + 0.15 * labels_pre + rng.normal(0.0, 0.035, n_each),
            ]
        )
        post = np.column_stack(
            [
                0.38 + 0.35 * labels_post + rng.normal(0.0, 0.035, n_each),
                0.35 + 0.30 * labels_post + rng.normal(0.0, 0.040, n_each),
                0.56 - 0.18 * labels_post + rng.normal(0.0, 0.040, n_each),
                0.30 + 0.15 * labels_post + rng.normal(0.0, 0.035, n_each),
            ]
        )
    else:  # pragma: no cover - parametrization guards this branch.
        raise ValueError(outcome)

    labels = np.concatenate([labels_pre, labels_post]).tolist()
    matrix = np.clip(np.vstack([pre, post]), 0.0, 1.0)
    return labels, {
        name: matrix[:, index].tolist() for index, name in enumerate(VERIFIER_NAMES)
    }


@pytest.mark.parametrize(
    ("outcome", "expected_verdict"),
    [
        ("drift_recovered_no_collapse", SUCCESS_VERDICT),
        ("no_gain_over_v10", NO_GAIN_VERDICT),
        ("blocked", BLOCKED_VERDICT),
    ],
)
def test_req_learn_3685_honest_synthetic_outcomes(
    outcome: str,
    expected_verdict: str,
) -> None:
    """SCENARIO-LEARN-3685: honest outcomes classify from synthetic fixtures."""

    labels, scores_by_verifier = _score_fixture(outcome)
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=1.0,
        now_s=3.0,
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3685.INFERENCE_SUBSTRATE
    if outcome == "blocked":
        assert artifact["n_online_updates"] == 0
        assert artifact["acceptance_gate"]["passed"] is False
    else:
        assert artifact["n_online_updates"] >= 200
        assert artifact["drift_detected_deploy_arm"] is True
        assert artifact["collapse_detected_deploy_arm"] is False
        assert artifact["collapse_detected_control"] is True
        assert artifact["pass_rate_vs_true_accuracy_distinct_assert"] is True
        assert max(abs(value) for value in artifact["weights_deploy_final"].values()) <= 0.8
        assert max(artifact["weights_control_final"].values()) == 1.0
        assert artifact["metrics_post_drift"]["deploy"]["auroc"] >= artifact[
            "metrics_post_drift"
        ]["static_carnot"]["auroc"]
    if outcome == "drift_recovered_no_collapse":
        assert artifact["post_drift_auroc_gain_over_v10"] > 0.0
        assert artifact["quality_maintained"] is True
    if outcome == "no_gain_over_v10":
        assert artifact["post_drift_auroc_gain_over_v10"] <= 0.0


def test_req_learn_3685_build_artifact_preconditions_and_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3685-1/2: cached drift traces are required and JSON is written."""

    blocked = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    validate_artifact(blocked)
    assert blocked["honest_verdict"] == BLOCKED_VERDICT
    assert blocked["preconditions_checked"][0]["resource"] == "fr11_module"

    labels, scores_by_verifier = _score_fixture("drift_recovered_no_collapse")
    (tmp_path / "python/carnot/fr11").mkdir(parents=True)
    monkeypatch.setattr(
        exp3685,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )
    monkeypatch.setattr(
        exp3685,
        "probe_cached_trace_preconditions",
        lambda root, n_examples: [
            {
                "resource": "cached_traces_with_per_verifier_scores_and_labels",
                "available": True,
                "detail": "fixture",
            }
        ],
    )
    artifact = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    validate_artifact(artifact)
    assert artifact["honest_verdict"] in exp3685.TERMINAL_VERDICTS
    assert artifact["preconditions_checked"][1]["available"] is True

    def _raise_score(root: Path, n_examples: int, random_seed: int) -> tuple[list[int], dict[str, list[float]]]:
        raise RuntimeError("cached traces unavailable")

    monkeypatch.setattr(exp3685, "score_fover_corpus", _raise_score)
    failed_score = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    validate_artifact(failed_score)
    assert failed_score["honest_verdict"] == BLOCKED_VERDICT
    assert failed_score["preconditions_checked"][-1]["resource"] == "cached_trace_scoring"

    labels_no_gain, scores_no_gain = _score_fixture("no_gain_over_v10")
    output = write_artifact(
        tmp_path,
        output_path="results/experiment_3685_fixture.json",
        labels=labels_no_gain,
        scores_by_verifier=scores_no_gain,
        started_s=0.0,
        now_s=1.0,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_artifact(payload)
    assert payload["honest_verdict"] == NO_GAIN_VERDICT


def test_req_learn_3685_validation_and_verdict_edges() -> None:
    """REQ-LEARN-3685-4/6: schema, drift, and verdict selection are strict."""

    assert (
        select_honest_verdict(
            gate_passed=True,
            quality_maintained=True,
            post_drift_auroc_gain_over_v10=0.01,
        )
        == SUCCESS_VERDICT
    )
    assert (
        select_honest_verdict(
            gate_passed=True,
            quality_maintained=True,
            post_drift_auroc_gain_over_v10=0.0,
        )
        == NO_GAIN_VERDICT
    )
    assert (
        select_honest_verdict(
            gate_passed=False,
            quality_maintained=True,
            post_drift_auroc_gain_over_v10=1.0,
        )
        == NO_GAIN_VERDICT
    )

    labels, scores_by_verifier = _score_fixture("drift_recovered_no_collapse")
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=1.0,
    )
    validate_artifact(artifact)

    missing = dict(artifact)
    missing.pop("duration_s")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        validate_artifact(missing)

    bad_verdict = dict(artifact, honest_verdict="complete: unsupported")
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        validate_artifact(bad_verdict)

    no_principles = dict(artifact)
    no_principles.pop("field_principles")
    with pytest.raises(ValueError, match="field_principles"):
        validate_artifact(no_principles)

    missing_principle = dict(artifact)
    missing_principle["field_principles"] = dict(artifact["field_principles"])
    missing_principle["field_principles"].pop("duration_s")
    with pytest.raises(ValueError, match="missing field principles"):
        validate_artifact(missing_principle)

    bad_gate = dict(artifact, acceptance_gate={"passed": "yes"})
    with pytest.raises(ValueError, match="acceptance_gate"):
        validate_artifact(bad_gate)

    bad_duration = dict(artifact, duration_s="fast")
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(bad_duration)

    bad_n = dict(artifact, n_online_updates=199)
    with pytest.raises(ValueError, match="at least"):
        validate_artifact(bad_n)

    bad_bool = dict(artifact, drift_detected_deploy_arm="true")
    with pytest.raises(ValueError, match="drift_detected_deploy_arm"):
        validate_artifact(bad_bool)

    bad_gain = dict(artifact, post_drift_auroc_gain_over_v10=float("nan"))
    with pytest.raises(ValueError, match="post_drift_auroc_gain_over_v10"):
        validate_artifact(bad_gain)

    assert exp3685.vote_distribution_distance(
        np.asarray([[0.8, 0.1], [0.7, 0.2]]),
        np.asarray([[0.1, 0.9], [0.2, 0.8]]),
    ) > exp3685.DRIFT_DISTANCE_THRESHOLD
    assert exp3685.detect_vote_distribution_drift(
        np.asarray([[0.8, 0.1], [0.7, 0.2]]),
        np.asarray([[0.1, 0.9], [0.2, 0.8]]),
    )

    pass_rate, true_accuracy = exp3685.online_metric_trajectories(
        [0, 1],
        [0.2, 0.8],
        n_windows=2,
    )
    assert pass_rate != true_accuracy
    assert exp3685.detect_weight_collapse({}) is False
