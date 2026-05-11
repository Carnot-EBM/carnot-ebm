"""Tests for Experiment 1807 EDDP Metric Benchmarking.

Spec: REQ-BENCH-1807, SCENARIO-BENCH-1807
"""

import json
from pathlib import Path

from scripts.experiment_1807_eddp import (
    compute_mcmc_metrics,
    get_thrml_module,
    run_benchmark,
)


def test_mcmc_metrics_computation():
    """SCENARIO-BENCH-1807: MCMC metrics are computed correctly."""
    metrics = compute_mcmc_metrics()
    assert "energy" in metrics
    assert "delay" in metrics
    assert "deficiency" in metrics
    assert "eddp" in metrics
    assert metrics["eddp"] == metrics["energy"] * metrics["delay"] * metrics["deficiency"]


def test_run_benchmark_outputs_valid_json(tmp_path: Path):
    """REQ-BENCH-1807: Artifact is written with required fields."""
    out_file = tmp_path / "experiment_1807_eddp.json"
    run_benchmark(str(out_file))

    assert out_file.exists()
    with open(out_file) as f:
        data = json.load(f)

    assert data["metadata"]["experiment_id"] == 1807
    assert "mcmc" in data["metrics"]
    
    mcmc = data["metrics"]["mcmc"]
    assert "energy" in mcmc
    assert "delay" in mcmc
    assert "deficiency" in mcmc
    
    thrml_mod = get_thrml_module()
    if thrml_mod is not None:
        assert data["honest_verdict"] == "complete_eddp_benchmark_passed"
        assert data["thrml_import_ready"] is True
        assert "thrml" in data["metrics"]
    else:
        assert data["honest_verdict"] == "thrml_not_importable_sim_blocked"
        assert data["thrml_import_ready"] is False
