"""Tests for Exp 3668 held-out dependency-aware weighting.

Spec: REQ-VERIFY-3668, SCENARIO-VERIFY-3668.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.verify import dependency_aware_weighting_heldout as exp3668
from carnot.verify import weaver_peer_comparison_v3 as exp3644


def _synthetic_fover_fixture(
    n_examples: int = 360,
) -> tuple[list[int], dict[str, list[float]]]:
    """Build a deterministic fixture where an anti-signal column is reusable."""

    rng = np.random.default_rng(3668)
    half = n_examples // 2
    labels = np.asarray([0] * half + [1] * half, dtype=np.int64)
    direction = labels * 2 - 1
    latent = rng.normal(0.0, 0.05, n_examples)
    columns = [
        np.clip(0.50 + 0.12 * direction + latent + rng.normal(0.0, 0.12, n_examples), 0.0, 1.0),
        np.clip(0.50 + 0.10 * direction + latent + rng.normal(0.0, 0.14, n_examples), 0.0, 1.0),
        np.clip(0.50 - 0.28 * direction + rng.normal(0.0, 0.10, n_examples), 0.0, 1.0),
        np.clip(0.50 + 0.04 * direction + rng.normal(0.0, 0.16, n_examples), 0.0, 1.0),
    ]
    return labels.tolist(), {
        name: column.tolist() for name, column in zip(exp3644.VERIFIER_NAMES, columns, strict=True)
    }


@pytest.mark.parametrize(
    (
        "outcome",
        "dependency_aware_auroc",
        "carnot_auroc",
        "delta_ci",
        "expected_bool",
    ),
    [
        pytest.param(
            "generalizes_heldout",
            0.86,
            0.81,
            {"point": 0.05, "ci95": [0.01, 0.09]},
            True,
            id="generalizes_heldout",
        ),
        pytest.param(
            "overfit_train_only",
            0.82,
            0.81,
            {"point": 0.01, "ci95": [-0.02, 0.04]},
            False,
            id="overfit_train_only",
        ),
        pytest.param(
            "blocked",
            None,
            None,
            None,
            False,
            id="blocked",
        ),
    ],
)
def test_exp3668_classifies_honest_outcomes_without_single_success_string(
    outcome: str,
    dependency_aware_auroc: float | None,
    carnot_auroc: float | None,
    delta_ci: dict[str, object] | None,
    expected_bool: bool,
) -> None:
    """SCENARIO-VERIFY-3668: anti-poison outcomes include win, null, and blocked."""

    classification = exp3668.classify_outcome(
        blocked=outcome == "blocked",
        heldout_dependency_aware_auroc=dependency_aware_auroc,
        heldout_carnot_auroc=carnot_auroc,
        delta_ci=delta_ci,
    )

    assert classification.category == outcome
    assert classification.terminal_verdict in exp3668.TERMINAL_VERDICTS
    assert classification.dependency_aware_generalizes_heldout is expected_bool


def test_exp3668_builds_heldout_artifact_from_train_only_fits() -> None:
    """SCENARIO-VERIFY-3668: dependency weights are fit on TRAIN and tested on TEST."""

    labels, scores_by_verifier = _synthetic_fover_fixture()
    artifact = exp3668.build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=4.0,
        random_seed=3668,
        split_seeds=(3668, 3669, 3670, 3671, 3672),
        n_bootstrap=16,
        upstream_exp3667={"dependency_aware_beats_carnot": True},
    )

    exp3668.validate_artifact(artifact)
    assert set(exp3668.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(exp3668.REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["n_splits"] == 5
    assert type(artifact["dependency_aware_generalizes_heldout"]) is bool
    assert artifact["heldout_auroc_dependency_aware"] > artifact["heldout_auroc_carnot"]
    assert artifact["heldout_delta_ci"]["ci95"][0] <= artifact["heldout_delta_ci"]["point"]
    assert artifact["heldout_delta_ci"]["point"] <= artifact["heldout_delta_ci"]["ci95"][1]
    assert 0.0 <= artifact["heldout_delong_p"] <= 1.0
    assert artifact["acceptance_gate"]["passed"] is True

    for split in artifact["split_panel"]:
        train = set(split["train_indices"])
        test = set(split["test_indices"])
        assert train
        assert test
        assert train.isdisjoint(test)
        assert split["test_auroc_dependency_aware"] >= split["test_auroc_carnot"]
        assert len(split["dependency_aware_weights"]) == len(exp3644.VERIFIER_NAMES)


def test_exp3668_blocks_when_upstream_success_is_not_confirmed(tmp_path: Path) -> None:
    """REQ-VERIFY-3668: missing Exp 3667 success emits the blocked verdict."""

    artifact = exp3668.build_artifact(tmp_path, started_s=0.0, now_s=0.5)

    exp3668.validate_artifact(artifact)
    assert artifact["honest_verdict"] == exp3668.BLOCKED_VERDICT
    assert artifact["dependency_aware_generalizes_heldout"] is False
    assert artifact["heldout_auroc_dependency_aware"] is None
    assert artifact["n_splits"] == 0


def test_exp3668_build_artifact_uses_scoring_paths_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3668: scoring success and scoring failure are terminal paths."""

    labels, scores_by_verifier = _synthetic_fover_fixture()
    monkeypatch.setattr(
        exp3668,
        "probe_preconditions",
        lambda root, n_examples: [{"resource": "fixture", "available": True, "detail": "ok"}],
    )
    monkeypatch.setattr(
        exp3668,
        "load_upstream_exp3667_artifact",
        lambda root: {"dependency_aware_beats_carnot": True},
    )
    monkeypatch.setattr(
        exp3668.exp3644,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )

    artifact = exp3668.build_artifact(
        tmp_path,
        started_s=1.0,
        now_s=3.0,
        n_bootstrap=8,
    )
    exp3668.validate_artifact(artifact)
    assert artifact["honest_verdict"] in exp3668.TERMINAL_VERDICTS
    assert artifact["n_splits"] == 5

    def _raise_score(*args: object, **kwargs: object) -> tuple[list[int], dict[str, list[float]]]:
        raise RuntimeError("FoVer scorer unavailable")

    monkeypatch.setattr(exp3668.exp3644, "score_fover_corpus", _raise_score)
    blocked = exp3668.build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    exp3668.validate_artifact(blocked)
    assert blocked["honest_verdict"] == exp3668.BLOCKED_VERDICT
    assert blocked["preconditions_checked"][-1]["resource"] == "fover_scoring"


def test_exp3668_statistics_helpers_and_validation_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3668: split, bootstrap, DeLong, and schema guards are deterministic."""

    labels, scores_by_verifier = _synthetic_fover_fixture(80)
    labels_arr = np.asarray(labels, dtype=np.int64)
    split = exp3668.stratified_train_test_indices(labels_arr, random_seed=11, test_fraction=0.25)
    assert set(split.train_indices.tolist()).isdisjoint(set(split.test_indices.tolist()))
    assert set(labels_arr[split.train_indices].tolist()) == {0, 1}
    assert set(labels_arr[split.test_indices].tolist()) == {0, 1}

    artifact = exp3668.build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=2.0,
        split_seeds=(1, 2, 3, 4, 5),
        n_bootstrap=8,
        upstream_exp3667={"dependency_aware_beats_carnot": True},
    )
    assert len(artifact["reproducibility_checksum"]) == 64

    upstream_path = tmp_path / exp3668.UPSTREAM_EXP3667_REL_PATH
    upstream_path.parent.mkdir(parents=True, exist_ok=True)
    upstream_path.write_text(
        json.dumps({"dependency_aware_beats_carnot": True, "honest_verdict": "complete: fixture"}),
        encoding="utf-8",
    )
    assert exp3668.load_upstream_exp3667_artifact(tmp_path)["dependency_aware_beats_carnot"] is True
    monkeypatch.setattr(exp3668.exp3656, "probe_preconditions", lambda root, n_examples: [])
    checks = exp3668.probe_preconditions(tmp_path, n_examples=2)
    assert checks[0]["resource"] == "exp3667_dependency_aware_beats_carnot"
    assert checks[0]["available"] is True
    upstream_path.write_text(json.dumps({"dependency_aware_beats_carnot": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="bare dependency_aware_beats_carnot"):
        exp3668.load_upstream_exp3667_artifact(tmp_path)

    upstream_blocked = exp3668.build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=1.0,
        upstream_exp3667={"dependency_aware_beats_carnot": False},
    )
    assert upstream_blocked["honest_verdict"] == exp3668.BLOCKED_VERDICT

    with pytest.raises(ValueError, match="bare boolean"):
        exp3668.validate_artifact(dict(artifact, dependency_aware_generalizes_heldout=1))
    with pytest.raises(ValueError, match="missing required artifact fields"):
        missing = dict(artifact)
        missing.pop("honest_verdict")
        exp3668.validate_artifact(missing)
    with pytest.raises(ValueError, match="field_principles must be present"):
        exp3668.validate_artifact(dict(artifact, field_principles=[]))
    with pytest.raises(ValueError, match="missing field principles"):
        exp3668.validate_artifact(dict(artifact, field_principles={}))
    with pytest.raises(ValueError, match="unsupported honest_verdict"):
        exp3668.validate_artifact(dict(artifact, honest_verdict="complete: invented"))
    with pytest.raises(ValueError, match="duration_s"):
        exp3668.validate_artifact(dict(artifact, duration_s="fast"))
    with pytest.raises(ValueError, match="n_splits"):
        exp3668.validate_artifact(dict(artifact, n_splits=4))
    with pytest.raises(ValueError, match="heldout_auroc_dependency_aware"):
        exp3668.validate_artifact(dict(artifact, heldout_auroc_dependency_aware=1.2))
    with pytest.raises(ValueError, match="heldout_delta_ci must be an object"):
        exp3668.validate_artifact(dict(artifact, heldout_delta_ci=[]))
    with pytest.raises(ValueError, match="heldout_delta_ci"):
        exp3668.validate_artifact(dict(artifact, heldout_delta_ci={"point": 0.0}))
    with pytest.raises(ValueError, match="bounds"):
        exp3668.validate_artifact(
            dict(artifact, heldout_delta_ci={"point": 0.0, "ci95": [float("nan"), 1.0]})
        )
    with pytest.raises(ValueError, match="contain"):
        exp3668.validate_artifact(
            dict(artifact, heldout_delta_ci={"point": 2.0, "ci95": [0.0, 1.0]})
        )
    with pytest.raises(ValueError, match="heldout_delong_p"):
        exp3668.validate_artifact(dict(artifact, heldout_delong_p=-0.1))
    with pytest.raises(ValueError, match="same length"):
        exp3668.build_artifact_from_scores(
            labels=labels[:-1],
            scores_by_verifier=scores_by_verifier,
            started_s=0.0,
            now_s=1.0,
            upstream_exp3667={"dependency_aware_beats_carnot": True},
        )
    with pytest.raises(ValueError, match="binary classes"):
        exp3668.stratified_train_test_indices(np.asarray([0, 0, 0]), random_seed=1)
    with pytest.raises(ValueError, match="test_fraction"):
        exp3668.stratified_train_test_indices(labels_arr, random_seed=1, test_fraction=1.0)
    with pytest.raises(ValueError, match="at least two rows"):
        exp3668.stratified_train_test_indices(np.asarray([0, 1]), random_seed=1)
    assert exp3668._round_metric(None) is None
    assert exp3668._round_p(None) is None


def test_exp3668_write_artifact_outputs_valid_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3668: writer persists the validated terminal JSON artifact."""

    labels, scores_by_verifier = _synthetic_fover_fixture()
    monkeypatch.setattr(
        exp3668,
        "build_artifact",
        lambda root, started_s=None, now_s=None: exp3668.build_artifact_from_scores(
            labels=labels,
            scores_by_verifier=scores_by_verifier,
            started_s=0.0,
            now_s=1.0,
            n_bootstrap=8,
            upstream_exp3667={"dependency_aware_beats_carnot": True},
        ),
    )

    output = exp3668.write_artifact(tmp_path, output_path="results/exp3668.json")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / "results/exp3668.json"
    exp3668.validate_artifact(payload)
