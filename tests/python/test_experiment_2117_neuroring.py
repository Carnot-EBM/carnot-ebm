"""Tests for Experiment 2117: NeuroRing Hardware Accounting for Sudoku Constraints."""

import json
import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from experiment_2117_neuroring import main

RESULT_PATH = REPO_ROOT / "results" / "experiment_2117_neuroring.json"

def test_experiment_2117(tmp_path, monkeypatch):
    """Test that the experiment script runs and generates valid JSON."""
    # Run the experiment
    exit_code = main()
    assert exit_code == 0

    assert RESULT_PATH.exists()
    
    with open(RESULT_PATH) as f:
        artifact = json.load(f)

    assert artifact["experiment"] == 2117
    assert "metrics" in artifact
    metrics = artifact["metrics"]
    assert metrics["n_spins"] == 729
    assert metrics["ring_size"] == 8
    assert metrics["total_sweeps"] == 2000
    assert metrics["latency_cycles_per_sweep"] == 4  # ring_size // 2
