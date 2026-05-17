"""Tests for the Exp 2244 ODAR routing benchmark.

Spec: REQ-ODAR-2244, SCENARIO-ODAR-2244.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from carnot.pipeline import odar_benchmark as exp


def test_req_odar_2244_corpus_is_balanced_and_routes_by_efe() -> None:
    """REQ-ODAR-2244: the corpus has 15 fast-path and 15 deliberative cases."""

    corpus = exp.build_reasoning_corpus()
    decisions = [exp.route_case(case, threshold=0.5) for case in corpus]

    assert len(corpus) == 30
    assert sum(case.difficulty == "high_confidence" for case in corpus) == 15
    assert sum(case.difficulty == "ambiguous" for case in corpus) == 15
    assert sum(decision["route"] == "FAST_PATH" for decision in decisions) == 15
    assert sum(decision["route"] == "DELIBERATIVE" for decision in decisions) == 15
    assert all(
        decision["route"] == "FAST_PATH"
        for case, decision in zip(corpus, decisions, strict=True)
        if case.difficulty == "high_confidence"
    )
    assert all(
        decision["route"] == "DELIBERATIVE"
        for case, decision in zip(corpus, decisions, strict=True)
        if case.difficulty == "ambiguous"
    )


def test_scenario_odar_2244_benchmark_reduces_compute_without_accuracy_loss() -> None:
    """SCENARIO-ODAR-2244: ODAR clears the compute gate without accuracy loss."""

    result = exp.evaluate_benchmark(exp.build_reasoning_corpus(), threshold=0.5)

    assert result["n_corpus"] == 30
    assert result["tier_calls_A"] == 120
    assert result["tier_calls_B"] == 75
    assert result["compute_reduction_pct"] == pytest.approx(37.5)
    assert result["accuracy_delta"] == pytest.approx(0.0)
    assert result["fast_path_fraction"] == pytest.approx(0.5)
    assert result["odar_benchmark_passed"] is True


def test_req_odar_2244_run_writes_valid_terminal_artifact(tmp_path: Path) -> None:
    """REQ-ODAR-2244: the runner writes the required artifact schema."""

    output = tmp_path / "results" / "experiment_2244_odar_benchmark.json"

    artifact = exp.run_benchmark(output_path=output)

    assert output.exists()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["odar_benchmark_passed"] is True
    assert artifact["compute_reduction_pct"] >= 30.0
    assert artifact["accuracy_delta"] >= -2.0
    assert artifact["n_corpus"] == 30
    assert "odar_router_imported" in artifact["preconditions_checked"]
    for field in exp.REQUIRED_ARTIFACT_FIELDS:
        assert field in artifact
        assert field in artifact["field_principles"]
    exp.validate_artifact(artifact)
