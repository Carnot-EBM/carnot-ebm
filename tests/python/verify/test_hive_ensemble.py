"""Tests for Exp 2398 HIVE soft-vote Tier 0 ensemble.

Spec: REQ-TIER0-012, SCENARIO-TIER0-012
"""

from __future__ import annotations

import json
import math

import numpy as np

from carnot.verify.hive_ensemble import (
    HiveEnsembleDetector,
    _soft_vote_weights_from_coefficients,
    build_experiment_artifact,
)


def _entry(case_id: str, correctness_label: str, scale: float, high_frequency: bool) -> dict:
    if high_frequency:
        alternatives = [
            {" the": -0.05 * scale, " and": -0.2 * scale, " to": -0.4 * scale},
            {" is": -0.1 * scale, " of": -0.3 * scale, " in": -0.5 * scale},
            {"</think>": -0.2 * scale, " answer": -1.4 * scale, "\n": -2.5 * scale},
        ]
        response = "Here's a thinking process <think> unresolved trace </think> answer is 9"
    else:
        alternatives = [
            {" theorem": -0.05 * scale, " proof": -0.2 * scale, " lemma": -0.4 * scale},
            {" silicon": -0.1 * scale, " lattice": -0.3 * scale, " orbital": -0.5 * scale},
            {" answer": -0.2 * scale, " result": -1.4 * scale, " value": -2.5 * scale},
        ]
        response = "7"

    return {
        "case_id": case_id,
        "correctness_label": correctness_label,
        "correct": correctness_label == "correct",
        "prompt": "Return exactly this integer and no other text: 7",
        "response_text": response,
        "token_logprobs": [-0.05 * scale, -0.1 * scale, -0.2 * scale],
        "token_texts": [next(iter(position)) for position in alternatives],
        "top_logprobs": alternatives,
    }


def _manifest_rows() -> list[dict]:
    return [
        _entry("correct-1", "correct", 0.5, False),
        _entry("correct-2", "correct", 0.6, False),
        _entry("correct-3", "correct", 0.7, False),
        _entry("wrong-1", "incorrect", 2.2, True),
        _entry("wrong-2", "incorrect", 2.5, True),
        _entry("wrong-3", "incorrect", 2.8, True),
    ]


def test_soft_vote_weights_are_positive_and_normalized() -> None:
    """REQ-TIER0-012-3: logistic coefficients become usable soft-vote weights."""
    weights = _soft_vote_weights_from_coefficients(np.array([-2.0, 0.0, 2.0]))

    assert np.all(weights > 0.0)
    assert math.isclose(float(np.sum(weights)), 1.0)
    assert weights[2] > weights[1] > weights[0]


def test_hive_artifact_fuses_available_verifiers_on_labeled_manifest(tmp_path) -> None:
    """REQ-TIER0-012-4: artifact reports AUROC, weights, and fused verifier count."""
    manifest = tmp_path / "telemetry.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in _manifest_rows()),
        encoding="utf-8",
    )

    artifact = build_experiment_artifact(manifest_path=manifest, n_eval_examples=6)

    assert artifact["status"] == "complete"
    assert artifact["n_eval_examples"] == 6
    assert artifact["n_verifiers_fused"] >= 2
    assert artifact["random_seed"] == 42
    assert artifact["hive_ensemble_auroc"] >= 0.5
    assert math.isfinite(artifact["hive_gap_closed_vs_hallscan"])
    assert set(artifact["verifier_weights"]) == set(artifact["available_verifiers"])


def test_hive_detector_verify_returns_weighted_score_after_fit() -> None:
    """SCENARIO-TIER0-012: fitted detector returns one bounded ensemble score."""
    rows = _manifest_rows()
    labels = [0, 0, 0, 1, 1, 1]
    detector = HiveEnsembleDetector(n_splits=3).fit(rows, labels)

    result = detector.verify(rows[-1])

    assert 0.0 <= result["hive_ensemble_score"] <= 1.0
    assert result["n_verifiers_fused"] >= 2
    assert set(result["verifier_scores"]) == set(detector.verifier_weights_)
