"""Tests for Exp 3644 Weaver peer-comparison correlation audit.

Spec: REQ-VERIFY-3644, SCENARIO-VERIFY-3644.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.verify import weaver_peer_comparison_v3 as exp3644
from carnot.verify.weaver_peer_comparison_v3 import (
    REQUIRED_ARTIFACT_FIELDS,
    build_artifact,
    build_artifact_from_scores,
    correlation_aware_weights,
    validate_artifact,
    weaver_style_weights,
)


def _redundant_fixture() -> tuple[list[int], dict[str, list[float]]]:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    return labels, {
        "fr11_session_memory": [0.10, 0.20, 0.15, 0.25, 0.80, 0.85, 0.90, 0.75],
        "tier0r_curry_howard": [0.11, 0.21, 0.16, 0.26, 0.81, 0.86, 0.91, 0.76],
        "tier0s_arithmetic_gap": [0.80, 0.10, 0.70, 0.20, 0.20, 0.90, 0.30, 0.85],
        "tier0u_logical_consistency": [0.40, 0.65, 0.35, 0.60, 0.55, 0.30, 0.70, 0.45],
    }


def test_exp3644_reports_correlation_and_weighted_aurocs() -> None:
    """SCENARIO-VERIFY-3644: redundant verifier pairs are audited explicitly."""

    labels, scores_by_verifier = _redundant_fixture()
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=1.25,
        random_seed=3644,
    )

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == (
        "complete: weaver_compared_correlation_matters_carnot_differentiates_on_correlation_awareness"
    )
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact["field_principles"])
    assert artifact["inference_substrate"].startswith("verifier_ensemble_against_cached_candidates")
    assert artifact["mean_offdiagonal_verifier_correlation"] > 0.25
    assert artifact["most_redundant_verifier_pair"]["pair"] == [
        "fr11_session_memory",
        "tier0r_curry_howard",
    ]
    assert artifact["most_redundant_verifier_pair"]["conditional_abs_correlation"] > 0.99
    assert artifact["correlation_awareness_matters"] is True
    assert artifact["n_examples"] == 8
    assert artifact["random_seed"] == 3644
    assert len(artifact["pearson_verifier_correlation_matrix"]) == 4
    assert set(artifact["conditional_verifier_correlation_by_label"]) == {"correct", "incorrect"}
    for key in (
        "ensemble_auroc_unweighted",
        "ensemble_auroc_weaver_style",
        "ensemble_auroc_carnot",
        "ensemble_auroc_correlation_aware",
    ):
        assert 0.0 <= artifact[key] <= 1.0


def test_exp3644_blocks_when_fover_or_verifiers_are_unavailable(tmp_path: Path) -> None:
    """REQ-VERIFY-3644: missing preconditions produce the terminal blocked verdict."""

    artifact = build_artifact(tmp_path, started_s=0.0, now_s=0.5)

    validate_artifact(artifact)
    assert artifact["honest_verdict"] == "complete: blocked_fover_corpus_or_verifiers_unavailable"
    assert artifact["mean_offdiagonal_verifier_correlation"] is None
    assert artifact["most_redundant_verifier_pair"] is None
    assert artifact["ensemble_auroc_unweighted"] is None
    assert artifact["ensemble_auroc_weaver_style"] is None
    assert artifact["ensemble_auroc_carnot"] is None
    assert artifact["correlation_awareness_matters"] is False
    assert artifact["n_examples"] == 0


def test_exp3644_weight_helpers_remain_normalized_and_finite() -> None:
    """REQ-VERIFY-3644: label-free and correlation-aware weights fail closed."""

    score_matrix = np.asarray(
        [
            [0.5, 0.1, 0.0],
            [0.5, 0.2, 1.0],
            [0.5, 0.3, 0.0],
            [0.5, 0.4, 1.0],
        ],
        dtype=float,
    )

    weaver = weaver_style_weights(score_matrix)
    aware = correlation_aware_weights(score_matrix)

    assert np.isfinite(weaver).all()
    assert np.isfinite(aware).all()
    assert np.all(weaver >= 0.0)
    assert np.all(aware >= 0.0)
    assert float(weaver.sum()) == 1.0
    assert float(aware.sum()) == 1.0


def test_exp3644_build_artifact_success_and_scoring_failure_are_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3644: injected scoring paths prove success and fail-closed branches."""

    labels, scores_by_verifier = _redundant_fixture()
    monkeypatch.setattr(
        exp3644,
        "probe_preconditions",
        lambda root, n_examples: [{"resource": "fixture", "available": True, "detail": "ok"}],
    )
    monkeypatch.setattr(
        exp3644,
        "score_fover_corpus",
        lambda root, n_examples, random_seed: (labels, scores_by_verifier),
    )

    artifact = build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    validate_artifact(artifact)
    assert artifact["honest_verdict"].startswith("complete: weaver_compared_")

    def _raise_score(*args: object, **kwargs: object) -> tuple[list[int], dict[str, list[float]]]:
        raise RuntimeError("scoring unavailable")

    monkeypatch.setattr(exp3644, "score_fover_corpus", _raise_score)
    blocked = build_artifact(tmp_path, started_s=1.0, now_s=2.0)
    validate_artifact(blocked)
    assert blocked["honest_verdict"] == "complete: blocked_fover_corpus_or_verifiers_unavailable"
    assert blocked["preconditions_checked"][-1]["resource"] == "fover_scoring"


def test_exp3644_score_fover_corpus_uses_exp2837_scoring_helpers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3644: FoVer scoring preserves the four Exp 2837 verifier columns."""

    from carnot.eval import fover_memory_leakage_v3 as fover

    rows = [{"label": "incorrect", "step_text": "bad"}, {"label": "correct", "step_text": "ok"}]
    monkeypatch.setattr(fover, "_read_fover_rows", lambda path: rows)
    monkeypatch.setattr(
        fover,
        "_select_balanced_subset",
        lambda source_rows, seed, n_examples: list(source_rows)[:n_examples],
    )
    monkeypatch.setattr(fover, "_label_to_int", lambda label: 1 if label == "incorrect" else 0)
    monkeypatch.setattr(
        fover,
        "_score_text_verifiers",
        lambda texts: {
            "tier0r_curry_howard": [0.8, 0.1],
            "tier0s_arithmetic_gap": [0.7, 0.2],
            "tier0u_logical_consistency": [0.6, 0.3],
        },
    )
    monkeypatch.setattr(fover, "_load_fr11_memory_index", lambda root: {"question_ids": {"x"}})
    monkeypatch.setattr(fover, "_fr11_memory_score", lambda row, memory: 0.9)

    labels, scores = exp3644.score_fover_corpus(tmp_path, n_examples=2, random_seed=7)

    assert labels == [1, 0]
    assert list(scores) == list(exp3644.VERIFIER_NAMES)
    assert scores["fr11_session_memory"] == [0.9, 0.9]
    assert scores["tier0r_curry_howard"] == [0.8, 0.1]


def test_exp3644_probe_preconditions_reports_import_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-3644: precondition details explain blocked verifier imports."""

    from carnot.eval import fover_memory_leakage_v3 as fover

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fover_corpus.jsonl").write_text(
        json.dumps({"label": "correct", "step_text": "ok"}) + "\n",
        encoding="utf-8",
    )

    def _raise_text(texts: list[str]) -> dict[str, list[float]]:
        raise RuntimeError("text verifier failed")

    def _raise_memory(root: Path) -> dict[str, object]:
        raise RuntimeError("memory failed")

    monkeypatch.setattr(fover, "_score_text_verifiers", _raise_text)
    monkeypatch.setattr(fover, "discover_fr11_state_files", lambda root: [{"path": "state"}])
    monkeypatch.setattr(fover, "_load_fr11_memory_index", _raise_memory)

    checks = exp3644.probe_preconditions(tmp_path, n_examples=1)

    assert checks[0]["available"] is True
    assert checks[1]["available"] is False
    assert "text verifier failed" in checks[1]["detail"]
    assert checks[2]["available"] is False
    assert "memory failed" in checks[2]["detail"]


def test_exp3644_helper_edges_and_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-VERIFY-3644: helper edge cases stay finite or raise clear errors."""

    labels, scores_by_verifier = _redundant_fixture()
    with pytest.raises(ValueError, match="same length"):
        build_artifact_from_scores(
            labels=labels[:-1],
            scores_by_verifier=scores_by_verifier,
            started_s=0.0,
            now_s=1.0,
        )
    with pytest.raises(ValueError, match="at least two"):
        build_artifact_from_scores(
            labels=[0, 1],
            scores_by_verifier={"one": [0.1, 0.9]},
            started_s=0.0,
            now_s=1.0,
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        exp3644.safe_pearson_matrix(np.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="two-dimensional"):
        correlation_aware_weights(np.asarray([1.0, 2.0]))
    assert exp3644.safe_pearson([1.0], [2.0]) == 0.0
    assert exp3644.safe_pearson([1.0, 1.0], [2.0, 3.0]) == 0.0
    assert exp3644.mean_offdiagonal_abs([]) == 0.0
    assert exp3644._round(None) is None
    assert np.allclose(correlation_aware_weights(np.ones((3, 2))), [0.5, 0.5])
    assert np.allclose(exp3644.normalize_weights([0.0, 0.0]), [0.5, 0.5])
    with pytest.raises(ValueError, match="column count"):
        exp3644.ensemble_scores(np.ones((2, 2)), [1.0])
    assert exp3644.tie_aware_auroc([1, 1], [0.1, 0.2]) == 0.5
    with pytest.raises(ValueError, match="at least one"):
        exp3644._score_matrix({}, [])
    with pytest.raises(ValueError, match="same length"):
        exp3644._score_matrix({"a": [1.0], "b": [1.0, 2.0]}, ["a", "b"])

    def _raise_solve(cov: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("singular")

    monkeypatch.setattr(exp3644.np.linalg, "solve", _raise_solve)
    aware = correlation_aware_weights(
        np.asarray([[0.1, 0.2], [0.2, 0.1], [0.8, 0.7], [0.9, 0.6]], dtype=float)
    )
    assert np.isfinite(aware).all()
    assert float(aware.sum()) == 1.0


def test_exp3644_validate_artifact_rejects_schema_errors() -> None:
    """REQ-VERIFY-3644: schema validation rejects malformed terminal artifacts."""

    labels, scores_by_verifier = _redundant_fixture()
    artifact = build_artifact_from_scores(
        labels=labels,
        scores_by_verifier=scores_by_verifier,
        started_s=0.0,
        now_s=1.0,
    )

    broken = dict(artifact)
    broken.pop("honest_verdict")
    with pytest.raises(ValueError, match="missing required"):
        validate_artifact(broken)

    for mutation, message in [
        (lambda item: item.update({"field_principles": None}), "field_principles"),
        (
            lambda item: item.update(
                {
                    "field_principles": {
                        key: value
                        for key, value in artifact["field_principles"].items()
                        if key != "n_examples"
                    }
                }
            ),
            "missing field principles",
        ),
        (lambda item: item.update({"honest_verdict": "bad"}), "unsupported"),
        (lambda item: item.update({"correlation_awareness_matters": {"value": True}}), "boolean"),
        (lambda item: item.update({"n_examples": -1}), "nonnegative"),
        (lambda item: item.update({"duration_s": "fast"}), "numeric"),
        (lambda item: item.update({"n_examples": 0}), "n_examples > 0"),
        (lambda item: item.update({"ensemble_auroc_carnot": None}), "must be finite"),
        (lambda item: item.update({"ensemble_auroc_carnot": 2.0}), r"\[0, 1\]"),
        (lambda item: item.update({"most_redundant_verifier_pair": None}), "must be present"),
    ]:
        candidate = dict(artifact)
        mutation(candidate)
        with pytest.raises(ValueError, match=message):
            validate_artifact(candidate)


def test_exp3644_write_artifact_persists_valid_json(tmp_path: Path) -> None:
    """REQ-VERIFY-3644: the writer persists the terminal JSON artifact."""

    output_path = exp3644.write_artifact(
        tmp_path,
        output_path="results/custom_exp3644.json",
        started_s=0.0,
        now_s=0.1,
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    validate_artifact(payload)
    assert payload["honest_verdict"] == "complete: blocked_fover_corpus_or_verifiers_unavailable"
