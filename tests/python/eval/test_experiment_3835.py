"""Tests for experiment 3835."""

import json
import pytest
from pathlib import Path

from carnot.eval.experiment_3835 import run_experiment_3835, _seed_t_ci95, _round_metric

def test_seed_t_ci95():
    """Test the CI95 calculation matches _seed_t_ci95 expectations."""
    # With one element, it raises ValueError
    with pytest.raises(ValueError):
        _seed_t_ci95([])
        
    # With a few elements
    values = [0.91, 0.92, 0.90, 0.93, 0.91]
    ci = _seed_t_ci95(values)
    assert ci["mean"] == _round_metric(sum(values) / 5)
    assert ci["low"] < ci["mean"]
    assert ci["high"] > ci["mean"]

def test_run_experiment_3835_success(tmp_path, monkeypatch):
    """Test the full experiment run."""
    repo_root = Path(__file__).resolve().parents[3]
    
    # We patch random to just take a small sample to speed up tests? 
    # Actually, running the whole 5 seeds on 1000 items takes 1 second, it's fast enough.
    
    # Check that it returns a valid artifact structure
    result = run_experiment_3835(repo_root)
    
    assert "honest_verdict" in result
    assert result["full_ensemble_auroc_mean"] > 0.90
    assert result["formal_only_auroc_mean"] > 0.88
    assert result["learned_only_auroc_mean"] > 0.85
    assert result["per_condition_ci95"]["full_ensemble_auroc"]["mean"] == result["full_ensemble_auroc_mean"]
    
    # Output file assertion structure
    assert result["cited_upstream_artifacts"]["exp3826"] == "results/experiment_3826_fover_ablation_faithful.json"
    
def test_preconditions_blocked(tmp_path, monkeypatch):
    """Test that blocked precondition raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="BLOCKED"):
        run_experiment_3835(tmp_path)
