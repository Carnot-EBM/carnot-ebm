"""Tests for Exp 3697 FR-11 continuous self-learning v12.

Spec: REQ-LEARN-3697, SCENARIO-LEARN-3697.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.fr11 import continuous_self_learning_v12 as exp3697
from carnot.fr11.continuous_self_learning_v12 import (
    BLOCKED_VERDICT,
    NO_GAIN_VERDICT,
    REQUIRED_ARTIFACT_FIELDS,
    SUCCESS_VERDICT,
    LearnedStructure,
    build_artifact,
    build_artifact_from_scores,
    persist_structure,
    restore_structure_via_subprocess,
    select_honest_verdict,
    validate_artifact,
    write_artifact,
)

VERIFIER_NAMES = exp3697.VERIFIER_NAMES


def _score_fixture(outcome: str) -> tuple[list[int], dict[str, list[float]]]:
    if outcome == "blocked":
        return [], {}

    rng = np.random.default_rng(3697 if outcome == "reset_and_persist_succeed_no_collapse" else 3976)
    n_each = 220
    labels_pre = np.asarray([0, 1] * (n_each // 2), dtype=np.int64)
    labels_post = np.asarray([0, 1] * (n_each // 2), dtype=np.int64)
    rng.shuffle(labels_pre)
    rng.shuffle(labels_post)

    if outcome == "reset_and_persist_succeed_no_collapse":
        pre = np.column_stack(
            [
                0.08 + 0.12 * labels_pre + rng.normal(0.0, 0.015, n_each),
                0.10 + 0.10 * labels_pre + rng.normal(0.0, 0.015, n_each),
                0.20 - 0.07 * labels_pre + rng.normal(0.0, 0.015, n_each),
                0.09 + rng.normal(0.0, 0.010, n_each),
            ]
        )
        post = np.column_stack(
            [
                0.58 + 0.12 * labels_post + rng.normal(0.0, 0.015, n_each),
                0.57 + 0.10 * labels_post + rng.normal(0.0, 0.015, n_each),
                0.56 + 0.28 * labels_post + rng.normal(0.0, 0.015, n_each),
                0.60 - 0.07 * labels_post + rng.normal(0.0, 0.015, n_each),
            ]
        )
    elif outcome == "no_gain_over_v11":
        pre = np.full((n_each, len(VERIFIER_NAMES)), 0.15, dtype=np.float64)
        post = np.full((n_each, len(VERIFIER_NAMES)), 0.75, dtype=np.float64)
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
        ("reset_and_persist_succeed_no_collapse", SUCCESS_VERDICT),
        ("no_gain_over_v11", NO_GAIN_VERDICT),
        ("blocked", BLOCKED_VERDICT),
    ],
)
def test_req_learn_3697_honest_synthetic_outcomes(
    outcome: str,
    expected_verdict: str,
    tmp_path: Path,
) -> None:
    """SCENARIO-LEARN-3697: honest synthetic outcomes are classified."""

    labels, scores_by_verifier = _score_fixture(outcome)
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=1.0,
        now_s=3.0,
        persistence_dir=tmp_path,
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == expected_verdict
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3697.INFERENCE_SUBSTRATE
    if outcome == "blocked":
        assert artifact["n_online_updates"] == 0
        assert artifact["acceptance_gate"]["passed"] is False
        return

    assert artifact["n_online_updates"] >= 200
    assert artifact["drift_detected_deploy_arm"] is True
    assert artifact["collapse_detected_deploy_arm"] is False
    assert artifact["pass_rate_vs_true_accuracy_distinct_assert"] is True
    assert artifact["deploy_drift_events"][0]["kind"] == "recoverable_drift"
    assert artifact["deploy_drift_events"][1]["kind"] == "transient_drift"
    assert artifact["structure_memory"]["sha256"] == artifact["structure_memory"]["restored_sha256"]
    assert max(abs(value) for value in artifact["weights_deploy_restored"].values()) <= 0.8
    if outcome == "reset_and_persist_succeed_no_collapse":
        assert artifact["reset_triggered_on_transient_drift"] is True
        assert artifact["structure_persisted_and_restored"] is True
        assert artifact["post_transient_drift_quality_gain_over_v11"] > 0.0
        assert artifact["quality_maintained"] is True
        assert artifact["acceptance_gate"]["passed"] is True
    else:
        assert artifact["post_transient_drift_quality_gain_over_v11"] <= 0.0
        assert artifact["acceptance_gate"]["passed"] is False


def test_req_learn_3697_build_artifact_preconditions_and_io(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3697: cached traces are required and JSON is written."""

    blocked = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    validate_artifact(blocked)
    assert blocked["honest_verdict"] == BLOCKED_VERDICT
    assert blocked["preconditions_checked"][0]["resource"] == "fr11_module"

    labels, scores_by_verifier = _score_fixture("reset_and_persist_succeed_no_collapse")
    (tmp_path / "python/carnot/fr11").mkdir(parents=True)
    monkeypatch.setattr(
        exp3697,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )
    monkeypatch.setattr(
        exp3697,
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
    assert artifact["honest_verdict"] in exp3697.TERMINAL_VERDICTS
    assert artifact["preconditions_checked"][1]["available"] is True

    def _raise_score(root: Path, n_examples: int, random_seed: int) -> tuple[list[int], dict[str, list[float]]]:
        raise RuntimeError("cached traces unavailable")

    monkeypatch.setattr(exp3697, "score_fover_corpus", _raise_score)
    failed_score = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    validate_artifact(failed_score)
    assert failed_score["honest_verdict"] == BLOCKED_VERDICT
    assert failed_score["preconditions_checked"][-1]["resource"] == "cached_trace_scoring"

    output = write_artifact(
        tmp_path,
        output_path="results/experiment_3697_fixture.json",
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=1.0,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    validate_artifact(payload)
    assert payload["honest_verdict"] in exp3697.TERMINAL_VERDICTS

    blocked_output = write_artifact(
        tmp_path,
        output_path="results/experiment_3697_blocked_fixture.json",
        started_s=0.0,
        now_s=1.0,
    )
    blocked_payload = json.loads(blocked_output.read_text(encoding="utf-8"))
    assert blocked_payload["honest_verdict"] == BLOCKED_VERDICT


def test_req_learn_3697_blocked_trace_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-LEARN-3697: malformed or non-distinct cached traces block honestly."""

    with pytest.raises(ValueError, match="same length"):
        build_artifact_from_scores(
            labels=[0, 1] * 100,
            scores_by_verifier={name: [0.1] * 199 for name in VERIFIER_NAMES},
            started_s=0.0,
            now_s=1.0,
        )

    one_class = build_artifact_from_scores(
        labels=[0] * 200,
        scores_by_verifier={name: [0.1] * 200 for name in VERIFIER_NAMES},
        started_s=0.0,
        now_s=1.0,
    )
    assert one_class["honest_verdict"] == BLOCKED_VERDICT

    non_distinct = build_artifact_from_scores(
        labels=[0, 1] * 120,
        scores_by_verifier={name: [0.5] * 240 for name in VERIFIER_NAMES},
        started_s=0.0,
        now_s=1.0,
    )
    assert non_distinct["honest_verdict"] == BLOCKED_VERDICT
    assert non_distinct["preconditions_checked"][-1]["resource"] == "distributionally_distinct_drift_slices"

    tiny_stream = exp3697.v11.DriftStream(
        labels=np.asarray([0, 1] * 50, dtype=np.int64),
        score_matrix=np.zeros((100, len(VERIFIER_NAMES)), dtype=np.float64),
        drift_point=50,
        projection="tiny",
        drift_distance=1.0,
        pre_vote_mean=np.zeros(len(VERIFIER_NAMES), dtype=np.float64),
        post_vote_mean=np.ones(len(VERIFIER_NAMES), dtype=np.float64),
    )
    monkeypatch.setattr(
        exp3697.v11,
        "build_drifting_trace_stream",
        lambda **kwargs: tiny_stream,
    )
    with pytest.raises(ValueError, match="four v12 drift phases"):
        exp3697.build_v12_drift_stream(
            labels=[0, 1] * 100,
            score_matrix=np.zeros((200, len(VERIFIER_NAMES)), dtype=np.float64),
            verifier_names=VERIFIER_NAMES,
            random_seed=1,
        )

    assert exp3697._ranking_gate_cleared(  # noqa: SLF001 - intentional edge coverage.
        np.asarray([0, 1], dtype=np.int64),
        np.zeros((2, len(VERIFIER_NAMES)), dtype=np.float64),
    ) is False
    assert exp3697._ranking_gate_cleared(  # noqa: SLF001 - intentional edge coverage.
        np.asarray([0] * 40 + [1], dtype=np.int64),
        np.zeros((41, len(VERIFIER_NAMES)), dtype=np.float64),
    ) is False


def test_req_learn_3697_validation_persistence_and_verdict_edges(tmp_path: Path) -> None:
    """REQ-LEARN-3697: schema, persistence, and verdict selection are strict."""

    assert (
        select_honest_verdict(
            gate_passed=True,
            quality_maintained=True,
            post_transient_drift_quality_gain_over_v11=0.01,
        )
        == SUCCESS_VERDICT
    )
    assert (
        select_honest_verdict(
            gate_passed=True,
            quality_maintained=True,
            post_transient_drift_quality_gain_over_v11=0.0,
        )
        == NO_GAIN_VERDICT
    )
    assert (
        select_honest_verdict(
            gate_passed=False,
            quality_maintained=True,
            post_transient_drift_quality_gain_over_v11=1.0,
        )
        == NO_GAIN_VERDICT
    )

    structure = LearnedStructure(
        verifier_names=("a", "b"),
        weights=np.asarray([0.4, 0.6], dtype=np.float64),
        edges=({"source": "a", "target": "b", "dependency": 0.1},),
        source_window="unit",
    )
    memory = persist_structure(structure, tmp_path / "structure.json")
    restored = restore_structure_via_subprocess(memory["path"])
    assert restored["sha256"] == memory["sha256"]
    assert restored["weights"] == {"a": 0.4, "b": 0.6}

    labels, scores_by_verifier = _score_fixture("reset_and_persist_succeed_no_collapse")
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=1.0,
        persistence_dir=tmp_path,
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

    bad_bool = dict(artifact, reset_triggered_on_transient_drift="true")
    with pytest.raises(ValueError, match="reset_triggered_on_transient_drift"):
        validate_artifact(bad_bool)

    bad_gain = dict(artifact, post_transient_drift_quality_gain_over_v11=float("nan"))
    with pytest.raises(ValueError, match="post_transient_drift_quality_gain_over_v11"):
        validate_artifact(bad_gain)

    no_memory = dict(artifact)
    no_memory.pop("structure_memory")
    with pytest.raises(ValueError, match="structure_memory"):
        validate_artifact(no_memory)

    bad_marker = dict(artifact, inference_substrate="cached CUDA marker")
    with pytest.raises(ValueError, match="forbidden inference marker"):
        validate_artifact(bad_marker)

    bad_memory = dict(artifact)
    bad_memory["structure_memory"] = dict(artifact["structure_memory"], restored_sha256="bad")
    bad_memory["structure_persisted_and_restored"] = True
    with pytest.raises(ValueError, match="structure SHA256 round-trip"):
        validate_artifact(bad_memory)
