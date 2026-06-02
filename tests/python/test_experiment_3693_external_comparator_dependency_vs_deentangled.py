"""Tests for Exp 3693 external de-entangled comparator.

Spec: REQ-VERIFY-3693, SCENARIO-VERIFY-3693.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.verify import dependency_aware_dual_condition_integrity as exp3680
from carnot.verify import external_comparator_dependency_vs_deentangled as exp3693
from carnot.verify import weaver_peer_comparison_v3 as exp3644


def _condition_rows(seed: int, *, n_examples: int = 1000) -> exp3680.ConditionScoreRows:
    """Build a FoVer-like fixture with correlated signal and one anti-signal."""

    rng = np.random.default_rng(seed)
    half = n_examples // 2
    labels = np.asarray([0] * half + [1] * half, dtype=np.int64)
    direction = labels * 2 - 1
    shared = rng.normal(0.0, 0.04, n_examples)
    production_columns = [
        np.clip(0.50 + 0.11 * direction + shared + rng.normal(0.0, 0.15, n_examples), 0.0, 1.0),
        np.clip(0.50 + 0.10 * direction + shared + rng.normal(0.0, 0.15, n_examples), 0.0, 1.0),
        np.clip(0.50 - 0.16 * direction + rng.normal(0.0, 0.20, n_examples), 0.0, 1.0),
        np.clip(0.50 + 0.05 * direction + rng.normal(0.0, 0.17, n_examples), 0.0, 1.0),
    ]
    architecture_columns = [
        np.zeros(n_examples, dtype=np.float64),
        production_columns[1],
        production_columns[2],
        production_columns[3],
    ]
    return exp3680.ConditionScoreRows(
        seed=seed,
        labels=labels.tolist(),
        production_scores_by_verifier={
            name: column.tolist()
            for name, column in zip(exp3644.VERIFIER_NAMES, production_columns, strict=True)
        },
        architecture_scores_by_verifier={
            name: column.tolist()
            for name, column in zip(exp3644.VERIFIER_NAMES, architecture_columns, strict=True)
        },
        production_state_visible_count=3,
        architecture_state_visible_count=0,
        subset_sha256=f"subset-{seed}",
    )


@pytest.mark.parametrize(
    (
        "blocked",
        "dependency_aware_auroc",
        "external_comparator_auroc",
        "delta_ci",
        "expected_category",
        "expected_bool",
    ),
    [
        pytest.param(
            False,
            0.94,
            0.91,
            {"point": 0.03, "ci95": [0.01, 0.05]},
            "candidate_beats_external",
            True,
            id="candidate_beats_external",
        ),
        pytest.param(
            False,
            0.91,
            0.92,
            {"point": -0.01, "ci95": [-0.03, 0.01]},
            "candidate_ties_or_loses_external",
            False,
            id="candidate_ties_or_loses_external",
        ),
        pytest.param(
            True,
            None,
            None,
            None,
            "blocked",
            False,
            id="blocked",
        ),
    ],
)
def test_exp3693_classifies_honest_outcomes_without_single_success_string(
    blocked: bool,
    dependency_aware_auroc: float | None,
    external_comparator_auroc: float | None,
    delta_ci: dict[str, object] | None,
    expected_category: str,
    expected_bool: bool,
) -> None:
    """SCENARIO-VERIFY-3693: anti-poison outcomes include win, null, and blocked."""

    classification = exp3693.classify_outcome(
        blocked=blocked,
        dependency_aware_auroc=dependency_aware_auroc,
        external_comparator_auroc=external_comparator_auroc,
        delta_ci=delta_ci,
    )

    assert classification.category == expected_category
    assert classification.terminal_verdict in exp3693.TERMINAL_VERDICTS
    assert classification.candidate_beats_external_comparator is expected_bool


def test_exp3693_spec_anchor_exists() -> None:
    """REQ-VERIFY-3693: OpenSpec declares the external-comparator contract first."""

    spec = Path("openspec/capabilities/verifiable-reasoning/spec.md").read_text(encoding="utf-8")

    assert "REQ-VERIFY-3693" in spec
    assert "SCENARIO-VERIFY-3693" in spec
    assert "candidate_beats_external_comparator" in spec


def test_exp3693_cig_deentangled_baseline_is_distinct_and_normalized() -> None:
    """REQ-VERIFY-3693: CIG/de-entangled weights are real, finite, and anti-signal aware."""

    row = _condition_rows(3693)
    labels = np.asarray(row.labels, dtype=np.int64)
    matrix = np.column_stack(
        [np.asarray(row.production_scores_by_verifier[name], dtype=np.float64) for name in exp3644.VERIFIER_NAMES]
    )

    fit = exp3693.fit_cig_deentangled_weights(
        labels=labels,
        score_matrix=matrix,
        verifier_names=exp3644.VERIFIER_NAMES,
    )
    crossfit = exp3693.cig_deentangled_crossfit_scores(
        labels=labels,
        score_matrix=matrix,
        verifier_names=exp3644.VERIFIER_NAMES,
        random_seed=3693,
        n_folds=5,
    )
    dependency_scores = exp3693.exp3667.score_weighting_panel(
        labels=labels,
        score_matrix=matrix,
        verifier_names=exp3644.VERIFIER_NAMES,
        random_seed=3693,
    )["dependency_aware_proper"]

    assert np.isfinite(fit.weights).all()
    assert pytest.approx(float(fit.weights.sum())) == 1.0
    assert np.isfinite(crossfit.scores).all()
    assert crossfit.scores.shape == dependency_scores.shape
    assert not np.array_equal(crossfit.scores, dependency_scores)
    assert -1 in fit.orientations.tolist()
    assert all(value >= 0.0 for value in fit.class_information_gain)
    assert len(crossfit.fold_weights) == crossfit.folds


def test_exp3693_builds_external_comparator_artifact_from_synthetic_scores() -> None:
    """SCENARIO-VERIFY-3693: all AUROCs use identical split rows and distinct score vectors."""

    rows = [_condition_rows(seed) for seed in (42, 137, 271, 314, 1729)]
    artifact = exp3693.build_artifact_from_condition_rows(
        rows,
        started_s=0.0,
        now_s=6.0,
        bootstrap_seeds=(42, 137, 271, 314, 1729),
        n_bootstrap=8,
        adversarial_verify_clean=True,
    )

    exp3693.validate_artifact(artifact)
    assert set(exp3693.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3693.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"] == exp3693.INFERENCE_SUBSTRATE
    assert artifact["n_seeds"] == 5
    assert artifact["n_examples"] == 1000
    assert artifact["n_pooled_examples"] == 5000
    assert type(artifact["candidate_beats_external_comparator"]) is bool
    assert artifact["dependency_aware_auroc"] != artifact["external_comparator_auroc"]
    assert artifact["score_vector_checksums"]["dependency_aware"] != artifact[
        "score_vector_checksums"
    ]["external_comparator"]
    assert artifact["dependency_vs_external_delta_ci"]["ci95"][0] <= artifact[
        "dependency_vs_external_delta_ci"
    ]["point"]
    assert artifact["dependency_vs_external_delta_ci"]["point"] <= artifact[
        "dependency_vs_external_delta_ci"
    ]["ci95"][1]
    assert 0.0 <= artifact["delong_p_dependency_vs_external"] <= 1.0
    assert artifact["external_comparator_implementation"]["reference"] == "arXiv:2604.07650"
    artifact_text = json.dumps(artifact)
    assert "GGUF" not in artifact_text
    assert "CUDA" not in artifact_text


def test_exp3693_blocks_when_preconditions_are_unavailable(tmp_path: Path) -> None:
    """REQ-VERIFY-3693: missing FoVer or weighting inputs write the blocked verdict."""

    artifact = exp3693.build_artifact(tmp_path, started_s=0.0, now_s=0.25)

    exp3693.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp3693.BLOCKED_VERDICT
    assert artifact["candidate_beats_external_comparator"] is False
    assert artifact["dependency_aware_auroc"] is None
    assert artifact["external_comparator_auroc"] is None
    assert artifact["n_seeds"] == 0


def test_exp3693_build_artifact_uses_scoring_paths_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3693: success and scorer failure are both terminal paths."""

    rows = [_condition_rows(seed) for seed in (42, 137, 271, 314, 1729)]
    monkeypatch.setattr(
        exp3693,
        "probe_preconditions",
        lambda root, n_examples: [{"resource": "fixture", "available": True, "detail": "ok"}],
    )
    monkeypatch.setattr(exp3693, "load_source_artifacts", lambda root: _source_fixture())
    monkeypatch.setattr(exp3693.exp3680, "discover_fr11_state_files", lambda root: [{"path": "state"}])
    monkeypatch.setattr(
        exp3693.exp3680,
        "score_dual_condition_rows",
        lambda root, seed, n_examples, state_files: rows[(42, 137, 271, 314, 1729).index(seed)],
    )

    artifact = exp3693.build_artifact(
        tmp_path,
        started_s=0.0,
        now_s=2.0,
        n_bootstrap=4,
        adversarial_verify_clean=True,
    )
    exp3693.validate_artifact(artifact)
    assert artifact["honest_verdict"] in exp3693.TERMINAL_VERDICTS

    def _raise_score(*args: object, **kwargs: object) -> exp3680.ConditionScoreRows:
        raise RuntimeError("dual scorer unavailable")

    monkeypatch.setattr(exp3693.exp3680, "score_dual_condition_rows", _raise_score)
    blocked = exp3693.build_artifact(tmp_path, started_s=0.0, now_s=1.0)
    exp3693.validate_artifact(blocked)
    assert blocked["honest_verdict"] == exp3693.BLOCKED_VERDICT
    assert blocked["preconditions_checked"][-1]["resource"] == "dual_condition_scoring"


def test_exp3693_write_artifact_stamps_adversarial_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3693: writer reruns adversarial verification before final JSON."""

    measured = exp3693.build_artifact_from_condition_rows(
        [_condition_rows(seed) for seed in (42, 137, 271, 314, 1729)],
        started_s=0.0,
        now_s=4.0,
        n_bootstrap=4,
        adversarial_verify_clean=False,
    )
    monkeypatch.setattr(exp3693, "build_artifact", lambda root, started_s=None, now_s=None: dict(measured))
    monkeypatch.setattr(exp3693, "run_adversarial_verify_report", lambda path: {"flag_count": 0, "flags": []})

    output = exp3693.write_artifact(tmp_path, output_path="result.json", started_s=0.0, now_s=1.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert saved["adversarial_verify_clean"] is True
    assert saved["acceptance_gate"]["passed"] is True
    assert saved["adversarial_verify_report"]["flag_count"] == 0


def test_exp3693_validation_edges() -> None:
    """REQ-VERIFY-3693: schema guards enforce bare booleans and anti-copy checks."""

    artifact = exp3693.build_artifact_from_condition_rows(
        [_condition_rows(seed) for seed in (42, 137, 271, 314, 1729)],
        started_s=0.0,
        now_s=4.0,
        n_bootstrap=4,
        adversarial_verify_clean=True,
    )

    with pytest.raises(ValueError, match="bare boolean"):
        exp3693.validate_artifact(dict(artifact, candidate_beats_external_comparator=1))
    with pytest.raises(ValueError, match="bare boolean"):
        exp3693.validate_artifact(dict(artifact, adversarial_verify_clean=1))
    with pytest.raises(ValueError, match="missing required artifact fields"):
        missing = dict(artifact)
        missing.pop("honest_verdict")
        exp3693.validate_artifact(missing)
    with pytest.raises(ValueError, match="field_principles must be present"):
        exp3693.validate_artifact(dict(artifact, field_principles=[]))
    with pytest.raises(ValueError, match="missing field principles"):
        exp3693.validate_artifact(dict(artifact, field_principles={}))
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        exp3693.validate_artifact(dict(artifact, honest_verdict="complete: invented"))
    with pytest.raises(ValueError, match="n_seeds"):
        exp3693.validate_artifact(dict(artifact, n_seeds=4))
    with pytest.raises(ValueError, match="n_examples"):
        exp3693.validate_artifact(dict(artifact, n_examples=999))
    with pytest.raises(ValueError, match="duration_s"):
        exp3693.validate_artifact(dict(artifact, duration_s="fast"))
    with pytest.raises(ValueError, match="dependency_aware_auroc"):
        exp3693.validate_artifact(dict(artifact, dependency_aware_auroc=1.2))
    with pytest.raises(ValueError, match="dependency_vs_external_delta_ci"):
        exp3693.validate_artifact(dict(artifact, dependency_vs_external_delta_ci={"point": 0.0}))
    with pytest.raises(ValueError, match="score vector"):
        copied_checksums: dict[str, Any] = dict(artifact["score_vector_checksums"])
        copied_checksums["external_comparator"] = copied_checksums["dependency_aware"]
        exp3693.validate_artifact(dict(artifact, score_vector_checksums=copied_checksums))
    with pytest.raises(ValueError, match="at least one condition"):
        exp3693.build_artifact_from_condition_rows([], started_s=0.0, now_s=1.0)
    with pytest.raises(ValueError, match="same length"):
        bad = _condition_rows(42)
        exp3693.build_artifact_from_condition_rows(
            [
                exp3680.ConditionScoreRows(
                    seed=bad.seed,
                    labels=bad.labels[:-1],
                    production_scores_by_verifier=bad.production_scores_by_verifier,
                    architecture_scores_by_verifier=bad.architecture_scores_by_verifier,
                )
            ],
            started_s=0.0,
            now_s=1.0,
        )
    assert exp3693._round_metric(None) is None


def test_exp3693_helper_and_schema_edge_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3693: helper edge cases fail closed with explicit errors."""

    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    matrix = np.asarray(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    fit = exp3693.fit_cig_deentangled_weights(
        labels=labels,
        score_matrix=matrix,
        verifier_names=["a", "b"],
    )
    assert np.allclose(fit.weights, [0.5, 0.5])

    with pytest.raises(ValueError, match="same length"):
        exp3693.fit_cig_deentangled_weights(labels=[0], score_matrix=matrix, verifier_names=["a", "b"])
    with pytest.raises(ValueError, match="two-dimensional"):
        exp3693.apply_cig_deentangled_fit(np.asarray([0.1, 0.2]), fit)
    with pytest.raises(ValueError, match="column count"):
        exp3693.apply_cig_deentangled_fit(np.ones((2, 3)), fit)
    with pytest.raises(ValueError, match="same length"):
        exp3693.cig_deentangled_crossfit_scores(
            labels=[0],
            score_matrix=matrix,
            verifier_names=["a", "b"],
            random_seed=1,
            n_folds=5,
        )
    with pytest.raises(ValueError, match="at least two examples per class"):
        exp3693.cig_deentangled_crossfit_scores(
            labels=[0, 1],
            score_matrix=np.ones((2, 2)),
            verifier_names=["a", "b"],
            random_seed=1,
            n_folds=5,
        )
    two_fold = exp3693.cig_deentangled_crossfit_scores(
        labels=labels,
        score_matrix=matrix,
        verifier_names=["a", "b"],
        random_seed=1,
        n_folds=1,
    )
    assert two_fold.folds == 2
    with pytest.raises(ValueError, match="two-dimensional"):
        exp3693.difficulty_weights(np.asarray([0.1, 0.2]))

    source_dir = tmp_path / "results"
    source_dir.mkdir()
    (tmp_path / exp3693.EXP3667_REL_PATH).write_text(
        json.dumps(
            {
                "auroc_dependency_aware_proper": 0.933238,
                "auroc_carnot_current": 0.919446,
                "auroc_weaver_style": 0.87158,
                "adversarial_verify_clean": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / exp3693.EXP3680_REL_PATH).write_text(
        json.dumps(
            {
                "production_auroc_dependency_aware": 0.925328,
                "production_auroc_carnot_current": 0.913134,
                "adversarial_verify_clean": True,
                "leak_free": True,
                "n_seeds": 5,
                "n_examples": 1000,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        exp3693.exp3680,
        "probe_preconditions",
        lambda root, n_examples: [{"resource": "fixture", "available": True, "detail": "ok"}],
    )
    checks = exp3693.probe_preconditions(tmp_path, n_examples=1000)
    assert checks[-2]["available"] is True
    assert exp3693.load_source_artifacts(tmp_path)["exp3680"]["leak_free"] is True

    (tmp_path / exp3693.EXP3667_REL_PATH).write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="Exp 3667 source artifact is missing"):
        exp3693.load_source_artifacts(tmp_path)
    (tmp_path / exp3693.EXP3667_REL_PATH).write_text(
        json.dumps(
            {
                "auroc_dependency_aware_proper": 0.933238,
                "auroc_carnot_current": 0.919446,
                "auroc_weaver_style": 0.87158,
                "adversarial_verify_clean": True,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / exp3693.EXP3680_REL_PATH).write_text(json.dumps({}), encoding="utf-8")
    with pytest.raises(ValueError, match="Exp 3680 source artifact is missing"):
        exp3693.load_source_artifacts(tmp_path)

    artifact = exp3693.build_artifact_from_condition_rows(
        [_condition_rows(seed) for seed in (42, 137, 271, 314, 1729)],
        started_s=0.0,
        now_s=4.0,
        n_bootstrap=4,
        adversarial_verify_clean=True,
    )
    with pytest.raises(ValueError, match="cached-verifier sentinel"):
        exp3693.validate_artifact(dict(artifact, inference_substrate="other"))
    with pytest.raises(ValueError, match="delong_p_dependency_vs_external"):
        exp3693.validate_artifact(dict(artifact, delong_p_dependency_vs_external=-0.1))
    with pytest.raises(ValueError, match="arXiv:2604.07650"):
        exp3693.validate_artifact(dict(artifact, external_comparator_implementation={}))
    with pytest.raises(ValueError, match="score_vector_checksums"):
        exp3693.validate_artifact(dict(artifact, score_vector_checksums=[]))
    with pytest.raises(ValueError, match="does not match"):
        exp3693.validate_artifact(dict(artifact, candidate_beats_external_comparator=not artifact["candidate_beats_external_comparator"]))

    assert exp3693.adversarial_report_is_clean(
        {"flags": [{"kind": "TAUTOLOGY", "severity": "warn"}]}
    ) is False
    assert exp3693.adversarial_report_is_clean(
        {"flags": [{"kind": "OTHER", "severity": "critical"}]}
    ) is False
    assert exp3693._weighted_class_information_gain(
        np.asarray([1, 1]),
        np.asarray([0.1, 0.2]),
        np.asarray([1.0, 1.0]),
    ) == 0.0
    assert exp3693._weighted_class_information_gain(
        np.asarray([0, 1, 0, 1]),
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([0.0, 1.0, 0.0, 1.0]),
    ) == 0.0
    assert exp3693._weighted_class_information_gain(
        np.asarray([0, 1, 0, 1]),
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([1.0, 1.0, 0.0, 0.0]),
    ) >= 0.0
    assert exp3693._weighted_binary_entropy(np.asarray([0, 1]), np.asarray([0.0, 0.0])) == 0.0
    assert exp3693._weighted_binary_entropy(np.asarray([0, 0]), np.asarray([1.0, 1.0])) == 0.0
    assert exp3693._weighted_mean(np.asarray([1.0, 2.0]), np.asarray([0.0, 0.0])) == 1.5
    with pytest.raises(ValueError, match="two-dimensional"):
        exp3693._checked_score_matrix(np.asarray([0.1, 0.2]), ["a"])
    with pytest.raises(ValueError, match="column count"):
        exp3693._checked_score_matrix(np.ones((2, 2)), ["a"])
    with pytest.raises(ValueError, match="finite"):
        exp3693._checked_score_matrix(np.asarray([[float("nan")]]), ["a"])
    assert exp3693._candidate_beats_from_metrics(None, 0.9, artifact["dependency_vs_external_delta_ci"]) is False
    assert exp3693._candidate_beats_from_metrics(0.9, 0.8, None) is False
    with pytest.raises(ValueError, match="binary classes"):
        exp3693._require_binary_labels(np.asarray([0, 0]))
    with pytest.raises(ValueError, match="must be an object"):
        exp3693._validate_ci([], "fixture_ci")
    with pytest.raises(ValueError, match="bounds"):
        exp3693._validate_ci({"point": 0.5, "ci95": [float("nan"), 1.0]}, "fixture_ci")
    with pytest.raises(ValueError, match="contain"):
        exp3693._validate_ci({"point": 1.5, "ci95": [0.0, 1.0]}, "fixture_ci")
    assert exp3693._round_p(None) is None
    assert exp3693._round_p(1e-8) == 1e-08
    assert exp3693._round_p(0.1234567) == 0.123457


def _source_fixture() -> dict[str, dict[str, object]]:
    return {
        "exp3667": {
            "auroc_dependency_aware_proper": 0.933238,
            "auroc_carnot_current": 0.919446,
            "auroc_weaver_style": 0.87158,
            "adversarial_verify_clean": True,
        },
        "exp3680": {
            "production_auroc_dependency_aware": 0.925328,
            "production_auroc_carnot_current": 0.913134,
            "adversarial_verify_clean": True,
            "leak_free": True,
            "n_seeds": 5,
            "n_examples": 1000,
        },
    }
