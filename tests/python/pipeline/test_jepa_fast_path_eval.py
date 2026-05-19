"""Exp 2550 balanced JEPA fast-path discrimination tests.

Spec: REQ-JEPA-006, SCENARIO-JEPA-012
"""

from __future__ import annotations

import json

from carnot.pipeline.jepa_fast_path_eval import (
    DEFAULT_THRESHOLD,
    build_balanced_jepa_corpus,
    evaluate_jepa_fast_path,
    run_experiment,
)


def test_balanced_corpus_has_required_safe_and_unsafe_rows() -> None:
    """REQ-JEPA-006: the eval corpus has 50 safe and 50 unsafe labeled rows."""
    corpus = build_balanced_jepa_corpus(seed=42)

    assert len(corpus) == 100
    assert sum(row.label for row in corpus) == 50
    assert sum(1 for row in corpus if row.label == 0) == 50
    assert all(len(row.response.split()) < 50 for row in corpus if row.label == 0)


def test_default_threshold_achieves_balanced_discrimination() -> None:
    """SCENARIO-JEPA-012: threshold 0.2 separates safe and unsafe responses."""
    metrics = evaluate_jepa_fast_path(
        build_balanced_jepa_corpus(seed=42),
        threshold=DEFAULT_THRESHOLD,
    )

    assert 0.30 <= metrics["fast_path_rate"] <= 0.80
    assert metrics["fast_path_precision"] >= 0.80
    assert metrics["jepa_discrimination_achieved"] is True


def test_run_experiment_writes_required_artifact_fields(tmp_path) -> None:
    """REQ-JEPA-006: Exp 2550 artifact includes the required terminal schema fields."""
    output_path = tmp_path / "experiment_2550_jepa_real_eval.json"

    artifact = run_experiment(output_path=output_path)
    written = json.loads(output_path.read_text())

    required = {
        "honest_verdict",
        "fast_path_rate",
        "fast_path_precision",
        "jepa_discrimination_achieved",
        "threshold_used",
        "n_corpus",
        "preconditions_checked",
        "duration_s",
        "random_seed",
    }
    assert required <= set(artifact)
    assert artifact == written
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["n_corpus"] >= 50
    assert artifact["random_seed"] == 42
