"""Tests for Exp 1819 KAN Latency Benchmark.

Spec: REQ-KAN-1819, SCENARIO-KAN-1819.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import experiment_1819_kan_latency as exp


def test_scenario_kan_1819_benchmark_writes_artifact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SCENARIO-KAN-1819: the runner outputs TPS and latency overhead."""
    output_path = tmp_path / "experiment_1819_kan_latency.json"
    artifact = exp.run_experiment(output_path, run_date="20260511")
    
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted == artifact
    
    exp.validate_artifact(artifact)
    assert artifact["model"] == exp.MODEL_SPECS[0]
    assert artifact["cikan_tps"] <= artifact["baseline_tps"]
    
    cli_path = tmp_path / "cli_1819.json"
    rc = exp.main(["--output", str(cli_path), "--run-date", "20260511"])
    assert rc == 0
    assert json.loads(cli_path.read_text(encoding="utf-8"))["status"] == "complete"
    assert "wrote=true" in capsys.readouterr().out


def test_validate_artifact_fails() -> None:
    with pytest.raises(AssertionError, match="missing required fields"):
        exp.validate_artifact({"status": "complete"})
