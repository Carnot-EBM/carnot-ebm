"""Tests for Exp 3647 FR-11 continuous self-learning v8.

Spec: REQ-LEARN-3647, SCENARIO-LEARN-3647.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carnot.fr11 import continuous_self_learning_v8 as exp3647
from carnot.fr11.continuous_self_learning_v8 import (
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


def _cached_score_fixture() -> tuple[list[int], dict[str, list[float]]]:
    labels_unit = [0, 0, 0, 0, 1, 1, 1, 1]
    score_rows = [
        [0.10, 0.12, 0.65, 0.01],
        [0.10, 0.11, 0.60, 0.01],
        [0.90, 0.88, 0.55, 0.01],
        [0.90, 0.87, 0.50, 0.01],
        [0.60, 0.58, 0.49, 0.02],
        [0.70, 0.72, 0.45, 0.02],
        [0.80, 0.82, 0.40, 0.02],
        [0.90, 0.88, 0.35, 0.02],
    ]
    labels = labels_unit * 30
    tiled = score_rows * 30
    names = [
        "fr11_session_memory",
        "tier0r_curry_howard",
        "tier0s_arithmetic_gap",
        "tier0u_logical_consistency",
    ]
    return labels, {name: [row[i] for row in tiled] for i, name in enumerate(names)}


def test_v8_artifact_holds_no_collapse_with_distinct_metrics() -> None:
    """SCENARIO-LEARN-3647: guarded online weighting prevents collapse."""

    labels, scores_by_verifier = _cached_score_fixture()
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=1.0,
        now_s=3.5,
        random_seed=3647,
    )

    validate_artifact(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["n_online_updates"] == 240
    assert artifact["collapse_detected_deploy_arm"] is False
    assert artifact["collapse_detected_control"] is True
    assert artifact["pass_rate_vs_true_accuracy_distinct_assert"] is True
    assert artifact["quality_maintained"] is True
    assert artifact["calibration_improved"] is True
    assert artifact["acceptance_gate"]["passed"] is True
    assert artifact["inference_substrate"] == exp3647.INFERENCE_SUBSTRATE
    assert artifact["correlation_source"] == "inline_cached_scores"
    assert artifact["honest_verdict"] in {SUCCESS_VERDICT, NO_GAIN_VERDICT}


def test_v8_uses_exp3644_correlation_seed_when_present() -> None:
    """REQ-LEARN-3647-1: Exp 3644 can seed the redundancy penalty."""

    labels, scores_by_verifier = _cached_score_fixture()
    seed_matrix = [
        [1.0, 0.8, 0.0, 0.0],
        [0.8, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=1.0,
        exp3644_artifact={
            "conditional_verifier_correlation_by_label": {
                "correct": seed_matrix,
                "incorrect": seed_matrix,
            }
        },
    )

    validate_artifact(artifact)
    assert artifact["correlation_source"] == "exp3644_artifact"
    assert artifact["redundancy_penalty_by_verifier"]["fr11_session_memory"] > 1.0


def test_v8_blocks_when_fr11_or_cached_traces_are_unavailable(tmp_path: Path) -> None:
    """REQ-LEARN-3647-2: missing preconditions produce the terminal blocked verdict."""

    artifact = build_artifact(tmp_path, started_s=0.0, now_s=0.5)

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == BLOCKED_VERDICT
    assert artifact["n_online_updates"] == 0
    assert artifact["acceptance_gate"]["passed"] is False
    assert artifact["preconditions_checked"][0]["resource"] == "fr11_module"


def test_v8_build_artifact_scores_cached_traces_when_preconditions_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3647: build_artifact runs from cached scorer output."""

    (tmp_path / "python/carnot/fr11").mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fover_corpus.jsonl").write_text(("{}\n" * 240), encoding="utf-8")
    labels, scores_by_verifier = _cached_score_fixture()
    monkeypatch.setattr(
        exp3647,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )

    artifact = build_artifact(tmp_path, started_s=0.0, now_s=1.0)

    validate_artifact(artifact)
    assert artifact["n_online_updates"] == 240
    assert artifact["preconditions_checked"][1]["available"] is True


def test_v8_scoring_failure_blocks_and_write_artifact_persists_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-LEARN-3647-2: scorer failures and blocked writes remain terminal."""

    (tmp_path / "python/carnot/fr11").mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fover_corpus.jsonl").write_text(("{}\n" * 240), encoding="utf-8")

    def _raise_score(*args: object, **kwargs: object) -> tuple[list[int], dict[str, list[float]]]:
        raise RuntimeError("scorer unavailable")

    monkeypatch.setattr(exp3647, "score_fover_corpus", _raise_score)
    artifact = build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    assert artifact["honest_verdict"] == BLOCKED_VERDICT
    assert artifact["preconditions_checked"][-1]["resource"] == "cached_trace_scoring"

    output = write_artifact(tmp_path / "missing-root", output_path="blocked.json", started_s=0.0, now_s=0.1)
    assert output.read_text(encoding="utf-8")


def test_v8_verdict_and_validation_edges() -> None:
    """REQ-LEARN-3647-3/5: terminal verdicts and schema validation are strict."""

    assert select_honest_verdict(gate_passed=True, correlation_aware_auroc_gain=0.01) == SUCCESS_VERDICT
    assert select_honest_verdict(gate_passed=True, correlation_aware_auroc_gain=0.0) == NO_GAIN_VERDICT
    assert select_honest_verdict(gate_passed=False, correlation_aware_auroc_gain=1.0) == NO_GAIN_VERDICT

    labels, scores_by_verifier = _cached_score_fixture()
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=1.0,
    )

    missing = dict(artifact)
    missing.pop("duration_s")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        validate_artifact(missing)

    bad_verdict = dict(artifact)
    bad_verdict["honest_verdict"] = "complete: unsupported"
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

    bad_gate = dict(artifact)
    bad_gate["acceptance_gate"] = {"passed": "yes"}
    with pytest.raises(ValueError, match="acceptance_gate"):
        validate_artifact(bad_gate)

    bad_duration = dict(artifact)
    bad_duration["duration_s"] = "fast"
    with pytest.raises(ValueError, match="duration_s"):
        validate_artifact(bad_duration)

    bad_n = dict(artifact)
    bad_n["n_online_updates"] = 199
    with pytest.raises(ValueError, match="at least"):
        validate_artifact(bad_n)

    bad_bool = dict(artifact)
    bad_bool["collapse_detected_control"] = "true"
    with pytest.raises(ValueError, match="collapse_detected_control"):
        validate_artifact(bad_bool)

    bad_gain = dict(artifact)
    bad_gain["correlation_aware_auroc_gain"] = float("nan")
    with pytest.raises(ValueError, match="correlation_aware_auroc_gain"):
        validate_artifact(bad_gain)

    with pytest.raises(ValueError, match="same length"):
        build_artifact_from_scores(
            labels=labels[:-1],
            scores_by_verifier=scores_by_verifier,
            started_s=0.0,
            now_s=1.0,
        )

    with pytest.raises(ValueError, match="both classes"):
        build_artifact_from_scores(
            labels=[0] * 240,
            scores_by_verifier=scores_by_verifier,
            started_s=0.0,
            now_s=1.0,
        )

    with pytest.raises(ValueError, match="at least 200"):
        build_artifact_from_scores(
            labels=labels[:8],
            scores_by_verifier={name: values[:8] for name, values in scores_by_verifier.items()},
            started_s=0.0,
            now_s=1.0,
        )


def test_v8_private_seed_and_matrix_edge_paths(tmp_path: Path) -> None:
    """REQ-LEARN-3647-1: malformed optional seeds fail closed to inline mode."""

    assert exp3647._load_exp3644_artifact(tmp_path) is None
    artifact_path = tmp_path / "results/experiment_3644_weaver_peer_comparison_v3.json"
    artifact_path.parent.mkdir()
    artifact_path.write_text("{bad json", encoding="utf-8")
    assert exp3647._load_exp3644_artifact(tmp_path) is None
    artifact_path.write_text("[]", encoding="utf-8")
    assert exp3647._load_exp3644_artifact(tmp_path) is None

    assert exp3647._redundancy_from_exp3644({}, expected_dim=2) is None
    assert (
        exp3647._redundancy_from_exp3644(
            {"conditional_verifier_correlation_by_label": []},
            expected_dim=2,
        )
        is None
    )
    assert (
        exp3647._redundancy_from_exp3644(
            {"conditional_verifier_correlation_by_label": {"correct": [[1.0, 0.0], [0.0, 1.0]]}},
            expected_dim=2,
        )
        is None
    )
    assert (
        exp3647._redundancy_from_exp3644(
            {
                "conditional_verifier_correlation_by_label": {
                    "correct": [[1.0]],
                    "incorrect": [[1.0]],
                }
            },
            expected_dim=2,
        )
        is None
    )

    with pytest.raises(ValueError, match="at least one"):
        exp3647._score_matrix({}, [])
    with pytest.raises(ValueError, match="same length"):
        exp3647._score_matrix({"a": [1.0], "b": [1.0, 2.0]}, ["a", "b"])
