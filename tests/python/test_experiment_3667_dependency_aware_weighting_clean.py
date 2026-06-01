"""Tests for Exp 3667 clean dependency-aware weighting rerun.

Spec: REQ-VERIFY-3667, SCENARIO-VERIFY-3667.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.verify import dependency_aware_weighting_clean as exp3667
from carnot.verify import weaver_peer_comparison_v3 as exp3644


def _synthetic_fover_fixture(n_examples: int = 1000) -> tuple[list[int], dict[str, list[float]]]:
    """Build a deterministic FoVer-like fixture with four distinct AUROCs."""

    rng = np.random.default_rng(7)
    half = n_examples // 2
    labels = np.asarray([0] * half + [1] * half, dtype=np.int64)
    base = np.concatenate(
        [
            rng.normal(0.38, 0.16, half),
            rng.normal(0.62, 0.16, half),
        ]
    )
    columns = [
        np.clip(base + rng.normal(0.0, 0.05, n_examples), 0.0, 1.0),
        np.clip(base * 0.8 + 0.1 + rng.normal(0.0, 0.10, n_examples), 0.0, 1.0),
        np.clip(1.0 - base + rng.normal(0.0, 0.18, n_examples), 0.0, 1.0),
        np.clip(rng.normal(0.5, 0.10, n_examples) + 0.05 * labels, 0.0, 1.0),
    ]
    return labels.tolist(), {
        name: column.tolist() for name, column in zip(exp3644.VERIFIER_NAMES, columns, strict=True)
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
    (
        "blocked",
        "adversarial_clean",
        "dependency_aware_auroc",
        "carnot_auroc",
        "delta_ci",
        "delong_p",
        "expected_category",
    ),
    [
        pytest.param(
            False,
            True,
            0.94,
            0.91,
            {"point": 0.03, "ci95": [0.01, 0.05]},
            0.01,
            "beats_carnot_significant",
            id="beats_carnot_significant",
        ),
        pytest.param(
            False,
            True,
            0.92,
            0.91,
            {"point": 0.01, "ci95": [-0.01, 0.03]},
            0.20,
            "no_significant_gain",
            id="no_significant_gain",
        ),
        pytest.param(
            True,
            False,
            None,
            None,
            None,
            None,
            "blocked",
            id="blocked",
        ),
    ],
)
def test_exp3667_classifies_honest_outcomes_without_single_success_string(
    blocked: bool,
    adversarial_clean: bool,
    dependency_aware_auroc: float | None,
    carnot_auroc: float | None,
    delta_ci: dict[str, object] | None,
    delong_p: float | None,
    expected_category: str,
) -> None:
    """SCENARIO-VERIFY-3667: anti-poison outcomes include win, null, and blocked."""

    classification = exp3667.classify_outcome(
        blocked=blocked,
        adversarial_verify_clean=adversarial_clean,
        dependency_aware_auroc=dependency_aware_auroc,
        carnot_auroc=carnot_auroc,
        delta_ci=delta_ci,
        delong_p=delong_p,
    )

    assert classification.category == expected_category
    assert classification.terminal_verdict in exp3667.TERMINAL_VERDICTS
    assert isinstance(classification.dependency_aware_beats_carnot, bool)


def test_exp3667_builds_clean_nonaliased_significance_artifact() -> None:
    """SCENARIO-VERIFY-3667: the clean panel has no duplicated AUROC aliases."""

    labels, scores_by_verifier = _synthetic_fover_fixture()
    artifact = exp3667.build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        prior_exp3644=_prior_exp3644_fixture(),
        started_s=0.0,
        now_s=3.0,
        random_seed=3668,
        bootstrap_seeds=(3667, 3668, 3669, 3670, 3671),
        n_bootstrap=16,
        adversarial_verify_clean=True,
    )

    exp3667.validate_artifact(artifact)
    assert set(exp3667.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3667.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["n_examples"] == 1000
    assert artifact["n_seeds"] == 5
    assert type(artifact["dependency_aware_beats_carnot"]) is bool
    top_level_aurocs = [artifact[field] for field in exp3667.AUROC_FIELDS]
    assert len({round(float(value), 6) for value in top_level_aurocs}) == len(top_level_aurocs)
    assert artifact["dependency_aware_vs_carnot_delta_ci"]["ci95"][0] <= artifact[
        "dependency_aware_vs_carnot_delta_ci"
    ]["point"] <= artifact["dependency_aware_vs_carnot_delta_ci"]["ci95"][1]
    assert 0.0 <= artifact["delong_p_dependency_vs_carnot"] <= 1.0
    for field in exp3667.AUROC_FIELDS:
        key = field.removeprefix("auroc_")
        assert artifact["auroc_bootstrap"][key]["ci95"][0] <= artifact[field]
        assert artifact["auroc_bootstrap"][key]["ci95"][1] >= artifact[field]
    assert len(artifact["seed_panel"]) == 5


def test_exp3667_blocks_when_preconditions_are_unavailable(tmp_path: Path) -> None:
    """REQ-VERIFY-3667: missing FoVer/dependency inputs produce the blocked verdict."""

    artifact = exp3667.build_artifact(tmp_path, started_s=0.0, now_s=0.25)

    exp3667.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp3667.BLOCKED_VERDICT
    assert artifact["dependency_aware_beats_carnot"] is False
    assert artifact["auroc_dependency_aware_proper"] is None
    assert artifact["n_examples"] == 0


def test_exp3667_build_artifact_uses_scoring_paths_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3667: scoring success and scoring failure are terminal paths."""

    labels, scores_by_verifier = _synthetic_fover_fixture()
    monkeypatch.setattr(
        exp3667,
        "probe_preconditions",
        lambda root, n_examples: [{"resource": "fixture", "available": True, "detail": "ok"}],
    )
    monkeypatch.setattr(exp3667.exp3656, "load_prior_exp3644_artifact", lambda root: _prior_exp3644_fixture())
    monkeypatch.setattr(
        exp3667.exp3644,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )

    artifact = exp3667.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=3.0,
        random_seed=3668,
        n_bootstrap=8,
        adversarial_verify_clean=True,
    )
    exp3667.validate_artifact(artifact)
    assert artifact["honest_verdict"] in exp3667.TERMINAL_VERDICTS

    def _raise_score(*args: object, **kwargs: object) -> tuple[list[int], dict[str, list[float]]]:
        raise RuntimeError("FoVer scorer unavailable")

    monkeypatch.setattr(exp3667.exp3644, "score_fover_corpus", _raise_score)
    blocked = exp3667.build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    exp3667.validate_artifact(blocked)
    assert blocked["honest_verdict"] == exp3667.BLOCKED_VERDICT
    assert blocked["preconditions_checked"][-1]["resource"] == "fover_scoring"


def test_exp3667_statistics_helpers_and_validation_edges() -> None:
    """REQ-VERIFY-3667: bootstrap, DeLong, and schema guards stay deterministic."""

    labels = [0, 0, 0, 1, 1, 1]
    dependency_scores = [0.1, 0.2, 0.3, 0.72, 0.8, 0.9]
    carnot_scores = [0.1, 0.4, 0.6, 0.52, 0.7, 0.8]

    delta = exp3667.paired_delta_ci(
        labels,
        dependency_scores,
        carnot_scores,
        seeds=(11, 12, 13, 14, 15),
        n_bootstrap=8,
    )
    delong = exp3667.paired_delong_test(labels, dependency_scores, carnot_scores)

    assert delta["point"] > 0.0
    assert len(delta["ci95"]) == 2
    assert delong["auc_difference"] > 0.0
    assert 0.0 <= delong["p_value"] <= 1.0
    tiny_bootstrap = exp3667.bootstrap_auroc_ci(
        [0, 1],
        [0.2, 0.8],
        seeds=(1,),
        n_bootstrap=8,
    )
    zero_se_delong = exp3667.paired_delong_test([0, 0, 1, 1], [0.1, 0.1, 0.9, 0.9], [0.1, 0.1, 0.9, 0.9])
    assert tiny_bootstrap["point"] == 1.0
    assert zero_se_delong["p_value"] == 1.0
    assert np.allclose(exp3667.compute_midrank(np.asarray([1.0, 1.0, 2.0])), [1.5, 1.5, 3.0])
    assert np.allclose(exp3667.covariance_matrix(np.ones((2, 1))), np.zeros((2, 2)))
    assert np.allclose(exp3667.covariance_matrix(np.ones((1, 3))), [[0.0]])
    assert exp3667.percentile_ci_or_point([], 0.25) == (0.25, 0.25)
    assert exp3667.adversarial_report_is_clean({"flag_count": 0, "flags": []}) is True
    assert (
        exp3667.adversarial_report_is_clean(
            {"flag_count": 1, "flags": [{"kind": "TAUTOLOGY", "severity": "critical"}]}
        )
        is False
    )

    labels_full, scores_by_verifier = _synthetic_fover_fixture()
    artifact = exp3667.build_artifact_from_scores(
        labels=labels_full,
        scores_by_verifier=scores_by_verifier,
        prior_exp3644=_prior_exp3644_fixture(),
        started_s=0.0,
        now_s=3.0,
        random_seed=3668,
        bootstrap_seeds=(3667, 3668, 3669, 3670, 3671),
        n_bootstrap=8,
        adversarial_verify_clean=True,
    )

    with pytest.raises(ValueError, match="bare boolean"):
        exp3667.validate_artifact(dict(artifact, dependency_aware_beats_carnot=1))
    with pytest.raises(ValueError, match="missing required artifact fields"):
        missing = dict(artifact)
        missing.pop("honest_verdict")
        exp3667.validate_artifact(missing)
    with pytest.raises(ValueError, match="missing field principles"):
        exp3667.validate_artifact(dict(artifact, field_principles={}))
    with pytest.raises(ValueError, match="field_principles"):
        exp3667.validate_artifact(dict(artifact, field_principles=[]))
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        exp3667.validate_artifact(dict(artifact, honest_verdict="complete: invented"))
    with pytest.raises(ValueError, match="adversarial_verify_clean"):
        exp3667.validate_artifact(dict(artifact, adversarial_verify_clean=1))
    with pytest.raises(ValueError, match="n_examples"):
        exp3667.validate_artifact(dict(artifact, n_examples=12))
    with pytest.raises(ValueError, match="n_seeds"):
        exp3667.validate_artifact(dict(artifact, n_seeds=4))
    with pytest.raises(ValueError, match="duration_s"):
        exp3667.validate_artifact(dict(artifact, duration_s="fast"))
    with pytest.raises(ValueError, match="auroc_unweighted"):
        exp3667.validate_artifact(dict(artifact, auroc_unweighted=-0.1))
    with pytest.raises(ValueError, match="aliased AUROC"):
        exp3667.validate_artifact(
            dict(artifact, auroc_dependency_aware_proper=artifact["auroc_carnot_current"])
        )
    with pytest.raises(ValueError, match="delta CI"):
        exp3667.validate_artifact(dict(artifact, dependency_aware_vs_carnot_delta_ci=[]))
    with pytest.raises(ValueError, match="delta CI"):
        exp3667.validate_artifact(dict(artifact, dependency_aware_vs_carnot_delta_ci={"point": 0.0}))
    with pytest.raises(ValueError, match="bounds"):
        exp3667.validate_artifact(
            dict(
                artifact,
                dependency_aware_vs_carnot_delta_ci={"point": 0.0, "ci95": [float("nan"), 1.0]},
            )
        )
    with pytest.raises(ValueError, match="contain"):
        exp3667.validate_artifact(
            dict(artifact, dependency_aware_vs_carnot_delta_ci={"point": 2.0, "ci95": [0.0, 1.0]})
        )
    with pytest.raises(ValueError, match="delong_p"):
        exp3667.validate_artifact(dict(artifact, delong_p_dependency_vs_carnot=1.5))

    with pytest.raises(ValueError, match="same length"):
        exp3667.checked_label_scores([0, 1], [0.1])
    with pytest.raises(ValueError, match="finite"):
        exp3667.checked_label_scores([0, 1], [0.1, float("nan")])
    with pytest.raises(ValueError, match="binary classes"):
        exp3667.paired_delong_test([0, 0], [0.1, 0.2], [0.1, 0.2])
    with pytest.raises(ValueError, match="same length"):
        exp3667.build_artifact_from_scores(
            labels=labels_full[:-1],
            scores_by_verifier=scores_by_verifier,
            prior_exp3644=_prior_exp3644_fixture(),
            started_s=0.0,
            now_s=1.0,
        )
    assert exp3667._significant_digits_match(0.0, 1.0, 5) is False
    assert exp3667._round_metric(None) is None
    assert exp3667._round_p(None) is None
    assert exp3667._round_p(1e-8) == 1e-8


def test_exp3667_write_artifact_stamps_self_adversarial_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3667: writer reruns adversarial verification before final JSON."""

    labels, scores_by_verifier = _synthetic_fover_fixture()
    artifact = exp3667.build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        prior_exp3644=_prior_exp3644_fixture(),
        started_s=0.0,
        now_s=3.0,
        random_seed=3668,
        bootstrap_seeds=(3667, 3668, 3669, 3670, 3671),
        n_bootstrap=8,
        adversarial_verify_clean=False,
    )
    monkeypatch.setattr(exp3667, "build_artifact", lambda root, started_s=None, now_s=None: dict(artifact))
    monkeypatch.setattr(
        exp3667,
        "run_adversarial_verify_report",
        lambda path: {"flag_count": 0, "flags": [], "max_severity": -1},
    )

    output = exp3667.write_artifact(tmp_path, output_path="results/exp3667.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/exp3667.json"
    assert payload["adversarial_verify_clean"] is True
    assert payload["acceptance_gate"]["passed"] is True
    assert payload["adversarial_verify_report"]["flag_count"] == 0
    assert payload["honest_verdict"] in exp3667.TERMINAL_VERDICTS
