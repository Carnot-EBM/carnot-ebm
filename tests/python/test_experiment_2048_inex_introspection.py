"""Tests for Exp 2048 InEx-style Continuous Introspection."""

import pytest
import jax.random as jrandom
from unittest.mock import patch
from scripts.experiment_2048_inex_introspection import _simulate_introspection, main

def test_introspection_gate():
    metrics = _simulate_introspection(n_cases=10, threshold=0.5)
    
    assert "false_accept_rate" in metrics
    assert "false_reject_rate" in metrics
    assert "total_resampled" in metrics
    assert metrics["model_used"] == "unsloth/gemma-4-31B-it-GGUF"
    assert metrics["n_cases"] == 10

def test_main():
    with patch("scripts.experiment_2048_inex_introspection.RESULTS_PATH", "results/test_experiment_2048_inex_introspection.json"):
        main()
        import os
        assert os.path.exists("results/test_experiment_2048_inex_introspection.json")
        os.remove("results/test_experiment_2048_inex_introspection.json")
