"""Tests for Exp 3656 correlation-aware weighting paradox diagnosis.

Spec: REQ-VERIFY-3656, SCENARIO-VERIFY-3656.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.verify import correlation_aware_weighting_paradox_diagnosis as exp3656


def _complementary_fixture() -> tuple[list[int], dict[str, list[float]]]:
    labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    return labels, {
        "memory_signal": [0.10, 0.15, 0.20, 0.12, 0.18, 0.22, 0.70, 0.82, 0.74, 0.88, 0.78, 0.84],
        "semantic_signal": [0.12, 0.16, 0.21, 0.13, 0.20, 0.23, 0.72, 0.83, 0.75, 0.89, 0.80, 0.86],
        "anti_signal": [0.82, 0.74, 0.88, 0.76, 0.84, 0.90, 0.26, 0.20, 0.30, 0.18, 0.24, 0.22],
        "low_variance_hint": [0.02, 0.02, 0.04, 0.02, 0.04, 0.02, 0.05, 0.07, 0.05, 0.07, 0.05, 0.07],
    }


def _prior_exp3644_fixture() -> dict[str, object]:
    return {
        "pearson_verifier_correlation_matrix": [
            [1.0, 0.44, -0.08, 0.05],
            [0.44, 1.0, -0.17, 0.08],
            [-0.08, -0.17, 1.0, -0.01],
            [0.05, 0.08, -0.01, 1.0],
        ],
        "ensemble_auroc_unweighted": 0.736244,
        "ensemble_auroc_weaver_style": 0.87158,
        "ensemble_auroc_carnot": 0.919446,
        "ensemble_auroc_correlation_aware": 0.635312,
        "auroc_delta_correlation_aware_vs_weaver": -0.236268,
    }


@pytest.mark.parametrize(
    ("naive_auroc", "dependency_aware_auroc", "carnot_auroc", "expected_category"),
    [
        pytest.param(0.63, 0.93, 0.92, "dependency_aware_recovers", id="dependency_aware_recovers"),
        pytest.param(0.63, 0.78, 0.92, "penalty_misspecified", id="penalty_misspecified"),
        pytest.param(0.87, 0.872, 0.89, "correlation_harmless", id="correlation_harmless"),
    ],
)
def test_exp3656_classifies_honest_outcomes_without_single_success_string(
    naive_auroc: float,
    dependency_aware_auroc: float,
    carnot_auroc: float,
    expected_category: str,
) -> None:
    """SCENARIO-VERIFY-3656: anti-poison classification covers honest outcomes."""

    classification = exp3656.classify_paradox(
        naive_auroc=naive_auroc,
        dependency_aware_auroc=dependency_aware_auroc,
        carnot_auroc=carnot_auroc,
    )

    assert classification.category == expected_category
    assert classification.category in exp3656.OUTCOME_CATEGORIES
    assert classification.terminal_verdict in exp3656.TERMINAL_VERDICTS


def test_exp3656_builds_dependency_aware_artifact_from_scores() -> None:
    """SCENARIO-VERIFY-3656: learned dependencies separate redundancy from signal."""

    labels, scores_by_verifier = _complementary_fixture()
    artifact = exp3656.build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        prior_exp3644=_prior_exp3644_fixture(),
        started_s=0.0,
        now_s=2.0,
        random_seed=3656,
        baseline_random_seed=3644,
    )

    exp3656.validate_artifact(artifact)
    assert set(exp3656.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3656.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"].startswith("verifier_ensemble_against_cached_candidates")
    assert artifact["n_examples"] == len(labels)
    assert artifact["random_seed"] == 3656
    assert artifact["ensemble_auroc_naive_correlation_aware"] == artifact["ensemble_auroc_correlation_aware"]
    assert artifact["ensemble_auroc_dependency_aware_proper"] >= artifact["ensemble_auroc_naive_correlation_aware"]
    assert artifact["exp3644_baseline_reproduction"]["auroc_delta_correlation_aware_vs_weaver"] == -0.236268
    assert artifact["dependency_aware_training_protocol"]["folds"] == 5
    assert len(artifact["dependency_aware_learned_graph"]["edges"]) == 3
    assert artifact["naive_penalty_diagnosis"]["summary"]
    assert artifact["correlation_harmless_or_penalty_misspecified"] in exp3656.OUTCOME_CATEGORIES


def test_exp3656_dependency_weights_keep_correlated_signal_and_flip_anti_signal() -> None:
    """REQ-VERIFY-3656: graph-aware weights are signed and dependency-informed."""

    labels, scores_by_verifier = _complementary_fixture()
    names = list(scores_by_verifier)
    score_matrix = exp3656.score_matrix(scores_by_verifier, names)

    fit = exp3656.fit_dependency_aware_weights(
        labels=np.asarray(labels),
        score_matrix=score_matrix,
        verifier_names=names,
    )

    assert np.isfinite(fit.weights).all()
    assert pytest.approx(float(np.abs(fit.weights).sum())) == 1.0
    assert fit.weights[names.index("memory_signal")] > 0.0
    assert fit.weights[names.index("semantic_signal")] > 0.0
    assert fit.weights[names.index("anti_signal")] < 0.0
    assert any(edge["pair"] == ["memory_signal", "semantic_signal"] for edge in fit.edges)


def test_exp3656_blocks_when_fover_or_correlation_matrix_unavailable(tmp_path: Path) -> None:
    """REQ-VERIFY-3656: missing corpus or Exp 3644 matrix writes terminal blocked."""

    artifact = exp3656.build_artifact(tmp_path, started_s=0.0, now_s=0.25)

    exp3656.validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: blocked_fover_corpus_or_correlation_matrix_unavailable"
    )
    assert artifact["ensemble_auroc_dependency_aware_proper"] is None
    assert artifact["ensemble_auroc_naive_correlation_aware"] is None
    assert artifact["n_examples"] == 0


def test_exp3656_build_artifact_uses_scoring_paths_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3656: runnable and scoring-failure branches are terminal."""

    labels, scores_by_verifier = _complementary_fixture()
    monkeypatch.setattr(
        exp3656,
        "probe_preconditions",
        lambda root, n_examples: [{"resource": "fixture", "available": True, "detail": "ok"}],
    )
    monkeypatch.setattr(exp3656, "load_prior_exp3644_artifact", lambda root: _prior_exp3644_fixture())
    monkeypatch.setattr(
        exp3656.exp3644,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )

    artifact = exp3656.build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    exp3656.validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete: paradox_resolved_")

    def _raise_score(*args: object, **kwargs: object) -> tuple[list[int], dict[str, list[float]]]:
        raise RuntimeError("score path unavailable")

    monkeypatch.setattr(exp3656.exp3644, "score_fover_corpus", _raise_score)
    blocked = exp3656.build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    exp3656.validate_artifact(blocked)
    assert blocked["honest_verdict"] == (
        "complete: blocked_fover_corpus_or_correlation_matrix_unavailable"
    )
    assert blocked["preconditions_checked"][-1]["resource"] == "fover_scoring"


def test_exp3656_preconditions_check_prior_correlation_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3656: Exp 3644 correlation artifact is a named precondition."""

    monkeypatch.setattr(
        exp3656.exp3644,
        "probe_preconditions",
        lambda root, n_examples: [{"resource": "fover_corpus", "available": True, "detail": "ok"}],
    )
    checks = exp3656.probe_preconditions(tmp_path, n_examples=12)

    assert checks[0]["resource"] == "fover_corpus"
    assert checks[-1]["resource"] == "exp3644_correlation_matrix"
    assert checks[-1]["available"] is False

    result_path = tmp_path / "results" / "experiment_3644_weaver_peer_comparison_v3.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps(_prior_exp3644_fixture()), encoding="utf-8")
    checks = exp3656.probe_preconditions(tmp_path, n_examples=12)
    assert checks[-1]["available"] is True


def test_exp3656_validate_artifact_rejects_schema_errors() -> None:
    """REQ-VERIFY-3656: required fields, principles, and metric ranges are enforced."""

    labels, scores_by_verifier = _complementary_fixture()
    artifact = exp3656.build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        prior_exp3644=_prior_exp3644_fixture(),
        started_s=0.0,
        now_s=1.0,
    )

    missing = dict(artifact)
    missing.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required artifact fields"):
        exp3656.validate_artifact(missing)

    missing_principles = dict(artifact, field_principles={})
    with pytest.raises(ValueError, match="missing field principles"):
        exp3656.validate_artifact(missing_principles)

    bad_verdict = dict(artifact, honest_verdict="complete: unexpected")
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        exp3656.validate_artifact(bad_verdict)

    bad_auc = dict(artifact, ensemble_auroc_carnot=1.5)
    with pytest.raises(ValueError, match="ensemble_auroc_carnot"):
        exp3656.validate_artifact(bad_auc)

    bad_principles_type = dict(artifact, field_principles=[])
    with pytest.raises(ValueError, match="field_principles"):
        exp3656.validate_artifact(bad_principles_type)

    bad_n_examples = dict(artifact, n_examples=-1)
    with pytest.raises(ValueError, match="n_examples"):
        exp3656.validate_artifact(bad_n_examples)

    bad_duration = dict(artifact, duration_s="fast")
    with pytest.raises(ValueError, match="duration_s"):
        exp3656.validate_artifact(bad_duration)

    bad_category = dict(artifact, correlation_harmless_or_penalty_misspecified="made_up")
    with pytest.raises(ValueError, match="correlation_harmless_or_penalty_misspecified"):
        exp3656.validate_artifact(bad_category)

    bad_diagnosis = dict(artifact, naive_penalty_diagnosis=None)
    with pytest.raises(ValueError, match="naive_penalty_diagnosis"):
        exp3656.validate_artifact(bad_diagnosis)

    bad_finite = dict(artifact, ensemble_auroc_carnot=float("nan"))
    with pytest.raises(ValueError, match="ensemble_auroc_carnot"):
        exp3656.validate_artifact(bad_finite)


def test_exp3656_helper_edges_and_write_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3656: helper edge cases remain deterministic and writable."""

    labels, scores_by_verifier = _complementary_fixture()
    artifact = exp3656.build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        prior_exp3644=_prior_exp3644_fixture(),
        started_s=0.0,
        now_s=1.0,
    )
    monkeypatch.setattr(exp3656, "build_artifact", lambda root: artifact)

    output = exp3656.write_artifact(tmp_path)

    assert output == tmp_path / exp3656.OUTPUT_REL_PATH
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["reproducibility_checksum"] == artifact["reproducibility_checksum"]

    with pytest.raises(ValueError, match="same length"):
        exp3656.build_artifact_from_scores(
            labels=labels[:-1],
            scores_by_verifier=scores_by_verifier,
            prior_exp3644=_prior_exp3644_fixture(),
            started_s=0.0,
            now_s=1.0,
        )
    with pytest.raises(ValueError, match="at least two"):
        exp3656.score_matrix({"one": [0.1, 0.9]}, ["one"])
    with pytest.raises(ValueError, match="same length"):
        exp3656.score_matrix({"one": [0.1], "two": [0.1, 0.2]}, ["one", "two"])
    with pytest.raises(ValueError, match="finite"):
        exp3656.score_matrix({"one": [0.1, float("nan")], "two": [0.2, 0.3]}, ["one", "two"])
    with pytest.raises(ValueError, match="two-dimensional"):
        exp3656.fit_dependency_aware_weights(
            labels=np.asarray([0, 1]),
            score_matrix=np.asarray([0.1, 0.9]),
            verifier_names=["one"],
        )
    with pytest.raises(ValueError, match="column count"):
        exp3656.fit_dependency_aware_weights(
            labels=np.asarray([0, 1]),
            score_matrix=np.asarray([[0.1, 0.2], [0.8, 0.7]]),
            verifier_names=["one"],
        )
    with pytest.raises(ValueError, match="same length"):
        exp3656.fit_dependency_aware_weights(
            labels=np.asarray([0, 1, 0]),
            score_matrix=np.asarray([[0.1, 0.2], [0.8, 0.7]]),
            verifier_names=["one", "two"],
        )
    with pytest.raises(ValueError, match="binary classes"):
        exp3656.fit_dependency_aware_weights(
            labels=np.asarray([0, 0]),
            score_matrix=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
            verifier_names=["one", "two"],
        )
    with pytest.raises(ValueError, match="at least two"):
        exp3656.fit_dependency_aware_weights(
            labels=np.asarray([0, 1]),
            score_matrix=np.asarray([[0.1], [0.8]]),
            verifier_names=["one"],
        )
    assert np.allclose(exp3656.normalize_signed_weights([0.0, 0.0]), [0.5, 0.5])

    monkeypatch.setattr(
        exp3656,
        "build_artifact",
        lambda root, started_s=None, now_s=None: artifact,
    )
    assert exp3656.write_artifact(tmp_path, output_path="results/again.json", started_s=0.0)
    assert exp3656._round(None) is None


def test_exp3656_prior_artifact_and_small_fold_edge_cases(tmp_path: Path) -> None:
    """REQ-VERIFY-3656: prior artifact validation and low-fold crossfit are covered."""

    result_path = tmp_path / "results" / "experiment_3644_weaver_peer_comparison_v3.json"
    result_path.parent.mkdir(parents=True)

    result_path.write_text(json.dumps({"pearson_verifier_correlation_matrix": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Pearson correlation matrix"):
        exp3656.load_prior_exp3644_artifact(tmp_path)

    invalid = _prior_exp3644_fixture()
    invalid.pop("ensemble_auroc_carnot")
    result_path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="ensemble_auroc_carnot"):
        exp3656.load_prior_exp3644_artifact(tmp_path)

    result_path.write_text(json.dumps(_prior_exp3644_fixture()), encoding="utf-8")
    assert exp3656.load_prior_exp3644_artifact(tmp_path)["ensemble_auroc_carnot"] == 0.919446

    labels = np.asarray([0, 0, 1, 1])
    matrix = np.asarray(
        [
            [0.1, 0.12],
            [0.2, 0.22],
            [0.8, 0.78],
            [0.9, 0.88],
        ]
    )
    crossfit = exp3656.dependency_aware_crossfit_scores(
        labels=labels,
        score_matrix=matrix,
        verifier_names=["a", "b"],
        random_seed=3656,
        n_folds=1,
    )
    assert crossfit.folds == 2
    assert np.isfinite(crossfit.scores).all()
