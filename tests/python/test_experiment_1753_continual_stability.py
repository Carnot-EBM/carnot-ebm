#!/usr/bin/env python3
"""Tests for Exp 1753 Continual Stability Evaluation Loop.

Spec: REQ-LEARN-1753, SCENARIO-LEARN-1753.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from scripts.experiment_1753_continual_stability import (
    ARTIFACT_SCHEMA,
    MODEL_ID,
    build_artifact,
    run_experiment,
    validate_artifact,
)

def test_build_artifact_success() -> None:
    """Verify artifact builds successfully with cases."""
    benchmark = {
        "benchmark_id": "test_bench",
        "cases": [{"case_id": "1"}, {"case_id": "2"}]
    }
    artifact = build_artifact(benchmark=benchmark, benchmark_path="dummy.json")
    assert artifact["status"] == "complete"
    assert artifact["evaluated_case_count"] == 2
    assert artifact["model_id"] == MODEL_ID
    assert artifact["schema"] == ARTIFACT_SCHEMA
    validate_artifact(artifact)

def test_build_artifact_empty() -> None:
    """Verify artifact is blocked if no cases exist."""
    benchmark = {"benchmark_id": "empty_bench", "cases": []}
    artifact = build_artifact(benchmark=benchmark, benchmark_path="dummy.json")
    assert artifact["status"] == "blocked"
    assert artifact["evaluated_case_count"] == 0

def test_run_experiment_creates_file() -> None:
    """Verify the full script writes the output file."""
    with TemporaryDirectory() as d:
        out_path = Path(d) / "out.json"
        bench_path = Path(d) / "bench.json"
        
        # Write dummy benchmark
        bench_path.write_text('{"cases": [{"case_id": "1"}], "benchmark_id": "tmp"}')
        
        result = run_experiment(output_path=out_path, benchmark_path=bench_path)
        assert out_path.exists()
        assert result["status"] == "complete"
        assert result["evaluated_case_count"] == 1
