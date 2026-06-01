"""Tests for Exp 3673 FR-11 continuous self-learning v10.

Spec: REQ-LEARN-3673, SCENARIO-LEARN-3673.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.fr11 import continuous_self_learning_v10 as exp3673
from carnot.fr11.continuous_self_learning_v10 import (
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

VERIFIER_NAMES = exp3673.VERIFIER_NAMES


def _score_fixture(outcome: str) -> tuple[list[int], dict[str, list[float]]]:
    if outcome == "blocked":
        return [], {}
    if outcome == "no_gain":
        labels = [1 if idx % 2 == 0 else 0 for idx in range(240)]
        scores = [0.8 if label else 0.2 for label in labels]
        return labels, {name: list(scores) for name in VERIFIER_NAMES}
    if outcome != "holds_no_collapse":  # pragma: no cover - parametrization guards this.
        raise ValueError(outcome)

    rng = np.random.default_rng(1)
    labels_arr = np.array([1] * 120 + [0] * 120, dtype=np.int64)
    rng.shuffle(labels_arr)
    signed_label = labels_arr * 2 - 1
    shared = rng.normal(0.0, 0.1, len(labels_arr))
    matrix = np.column_stack(
        [
            np.clip(0.5 + 0.22 * signed_label + shared + rng.normal(0.0, 0.22, len(labels_arr)), 0.0, 1.0),
            np.clip(0.5 + 0.22 * signed_label - shared + rng.normal(0.0, 0.22, len(labels_arr)), 0.0, 1.0),
            np.clip(0.5 - 0.18 * signed_label + 0.5 * shared + rng.normal(0.0, 0.25, len(labels_arr)), 0.0, 1.0),
            np.clip(0.5 + 0.10 * signed_label + rng.normal(0.0, 0.30, len(labels_arr)), 0.0, 1.0),
        ]
    )
    return labels_arr.tolist(), {
        name: matrix[:, index].tolist() for index, name in enumerate(VERIFIER_NAMES)
    }


@pytest.mark.parametrize(
    ("outcome", "expected_verdict"),
    [
        ("holds_no_collapse", SUCCESS_VERDICT),
        ("no_gain", NO_GAIN_VERDICT),
        ("blocked", BLOCKED_VERDICT),
    ],
)
def test_req_learn_3673_honest_synthetic_outcomes(
    outcome: str,
    expected_verdict: str,
) -> None:
    """SCENARIO-LEARN-3673: honest outcomes classify from synthetic fixtures."""

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
    assert artifact["inference_substrate"] == exp3673.INFERENCE_SUBSTRATE
    if outcome == "blocked":
        assert artifact["n_online_updates"] == 0
        assert artifact["acceptance_gate"]["passed"] is False
    else:
        assert artifact["n_online_updates"] == 240
        assert artifact["collapse_detected_deploy_arm"] is False
        assert artifact["collapse_detected_control"] is True
        assert artifact["pass_rate_vs_true_accuracy_distinct_assert"] is True
        assert artifact["quality_maintained"] is True
        assert max(abs(value) for value in artifact["weights_deploy_final"].values()) <= 0.8
        assert max(artifact["weights_control_final"].values()) == 1.0
    if outcome == "holds_no_collapse":
        assert artifact["online_dependency_aware_auroc_gain"] > 0.0
        assert artifact["metrics_after_deploy_online_adaptation"]["auroc"] > artifact[
            "metrics_fixed_dependency_aware"
        ]["auroc"]
    if outcome == "no_gain":
        assert artifact["online_dependency_aware_auroc_gain"] == 0.0


def test_req_learn_3673_build_artifact_preconditions_and_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3673-1/2: cached traces are required and JSON is written."""

    blocked = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    validate_artifact(blocked)
    assert blocked["honest_verdict"] == BLOCKED_VERDICT
    assert blocked["preconditions_checked"][0]["resource"] == "fr11_module"

    (tmp_path / "python/carnot/fr11").mkdir(parents=True)
    monkeypatch.setattr(
        exp3673,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: _score_fixture("holds_no_collapse"),
    )
    monkeypatch.setattr(
        exp3673,
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
    assert artifact["honest_verdict"] in exp3673.TERMINAL_VERDICTS
    assert artifact["preconditions_checked"][1]["available"] is True

    def _raise_score(root: Path, n_examples: int, random_seed: int) -> tuple[list[int], dict[str, list[float]]]:
        raise RuntimeError("cached traces unavailable")

    monkeypatch.setattr(exp3673, "score_fover_corpus", _raise_score)
    failed_score = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    validate_artifact(failed_score)
    assert failed_score["honest_verdict"] == BLOCKED_VERDICT
    assert failed_score["preconditions_checked"][-1]["resource"] == "cached_trace_scoring"

    output = write_artifact(
        tmp_path,
        output_path="results/experiment_3673_fixture.json",
        labels=_score_fixture("no_gain")[0],
        scores_by_verifier=_score_fixture("no_gain")[1],
        started_s=0.0,
        now_s=1.0,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_artifact(payload)
    assert payload["honest_verdict"] == NO_GAIN_VERDICT


def test_req_learn_3673_validation_and_verdict_edges() -> None:
    """REQ-LEARN-3673-4/6: schema and verdict selection are strict."""

    assert (
        select_honest_verdict(gate_passed=True, online_dependency_aware_auroc_gain=0.01)
        == SUCCESS_VERDICT
    )
    assert (
        select_honest_verdict(gate_passed=True, online_dependency_aware_auroc_gain=0.0)
        == NO_GAIN_VERDICT
    )
    assert (
        select_honest_verdict(gate_passed=False, online_dependency_aware_auroc_gain=1.0)
        == NO_GAIN_VERDICT
    )

    labels, scores_by_verifier = _score_fixture("holds_no_collapse")
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

    bad_bool = dict(artifact, collapse_detected_control="true")
    with pytest.raises(ValueError, match="collapse_detected_control"):
        validate_artifact(bad_bool)

    bad_gain = dict(artifact, online_dependency_aware_auroc_gain=float("nan"))
    with pytest.raises(ValueError, match="online_dependency_aware_auroc_gain"):
        validate_artifact(bad_gain)

    with pytest.raises(ValueError, match="same length"):
        exp3673.online_metric_trajectories([0], [0.2, 0.8])
    with pytest.raises(ValueError, match="same length"):
        exp3673.score_metrics([0], [0.2, 0.8])
    with pytest.raises(ValueError, match="same length"):
        build_artifact_from_scores(
            labels=[0, 1],
            scores_by_verifier={name: [0.1] for name in VERIFIER_NAMES},
            started_s=0.0,
            now_s=1.0,
        )

    pass_rate, true_accuracy = exp3673.online_metric_trajectories(
        [0, 1],
        [0.2, 0.8],
        n_windows=2,
    )
    assert pass_rate != true_accuracy
    assert exp3673.detect_weight_collapse({}) is False
