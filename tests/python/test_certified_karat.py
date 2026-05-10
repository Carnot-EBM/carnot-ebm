"""Tests for Certified KArAt Evaluation.

Spec references: REQ-KAN-1689, SCENARIO-KAN-1689.
"""

import json
from pathlib import Path

import pytest

from carnot.models.certified_karat import (
    CertifiedKArAtBenchmark,
    build_experiment_1689_artifact,
    write_experiment_1689_artifact,
)


def test_certified_karat_benchmark() -> None:
    """REQ-KAN-1689: Benchmark runs and computes metrics correctly."""
    benchmark = CertifiedKArAtBenchmark(samples=10)
    results = benchmark.run()
    
    assert "accuracy" in results
    assert "max_error" in results
    assert "mse" in results
    assert "baseline_bounds" in results
    assert "certified_bounds" in results
    
    assert results["accuracy"] > 0.0
    assert results["max_error"] >= 0.0
    assert results["mse"] >= 0.0


def test_build_experiment_1689_artifact() -> None:
    """SCENARIO-KAN-1689: Artifact contains required keys."""
    artifact = build_experiment_1689_artifact()
    assert artifact["schema"] == "carnot.certified_karat.v1"
    assert artifact["status"] == "complete"
    assert artifact["experiment"] == 1689
    assert artifact["honest_verdict"] == "complete: certified_karat_evaluated"
    assert "results" in artifact


def test_write_experiment_1689_artifact(tmp_path: Path) -> None:
    """SCENARIO-KAN-1689: Artifact is written to disk correctly."""
    out_path = tmp_path / "test_1689.json"
    written = write_experiment_1689_artifact(out_path)
    
    assert out_path.exists()
    content = json.loads(out_path.read_text(encoding="utf-8"))
    assert content["schema"] == "carnot.certified_karat.v1"
    assert content == written
