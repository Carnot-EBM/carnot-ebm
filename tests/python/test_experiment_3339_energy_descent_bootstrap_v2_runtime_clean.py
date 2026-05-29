"""Tests for Exp 3339 Energy Descent Bootstrap V2 Runtime Clean.

Spec refs: REQ-INFER-SOTA-3339
"""

import sys
import pytest
from unittest.mock import MagicMock

from carnot.reporting.energy_descent_bootstrap_v2_runtime_clean_3339 import exact_verifier, run_experiment

def test_exact_verifier():
    """REQ-INFER-SOTA-3339: Score every candidate with exact verifiers."""
    assert exact_verifier("The answer is 42.", "42") is True
    assert exact_verifier("I think it is 43.", "42") is False
    assert exact_verifier("42", "42") is True
    assert exact_verifier("", "42") is False

def test_run_experiment_mocked(monkeypatch, tmp_path):
    """REQ-INFER-SOTA-3339: Test run_experiment without live GPU requirements."""
    monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
    monkeypatch.setattr(
        "carnot.reporting.energy_descent_bootstrap_v2_runtime_clean_3339.cached_sota_pair",
        lambda gpu_indices: [
            {"hf_id": "mock/model-1", "gpu": 0, "model_path": "/mock/path"}
        ]
    )
    
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.cuda.device_count.return_value = 0
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    
    artifact = run_experiment(project_root=tmp_path)
    
    assert artifact["honest_verdict"].startswith("blocked")
    assert artifact["energy_descent_bootstrap_ready"] is False
    assert len(artifact["model_specs"]) == 1
