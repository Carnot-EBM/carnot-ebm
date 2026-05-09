"""
Tests for Experiment 1629: Validate EBRM optimizations against local SOTA models.

References:
- REQ-EBRM-1629
- SCENARIO-EBRM-1629
"""
import json
import os
import sys
import importlib.util
from unittest.mock import patch, mock_open

# Load the script dynamically since it's not a standard package module
def load_script():
    script_path = "scripts/experiment_1629_ebrm_sota.py"
    spec = importlib.util.spec_from_file_location("experiment_1629_ebrm_sota", script_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules["experiment_1629_ebrm_sota"] = module
        spec.loader.exec_module(module)
        return module
    return None

exp_1629 = load_script()

def test_run_evaluation_success():
    """Test successful SOTA model loading and evaluation."""
    mock_models = [
        {"hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "model_path": "/fake/path/1"},
        {"hf_id": "unsloth/gemma-4-31B-it-GGUF", "model_path": "/fake/path/2"},
    ]
    with patch("experiment_1629_ebrm_sota.cached_sota_pair", return_value=mock_models):
        result = exp_1629.run_evaluation()
        assert result["honest_verdict"] == "ebrm_sota_validation_complete"
        assert len(result["models_used"]) == 2
        assert "unsloth/Qwen3.6-35B-A3B-GGUF" in result["models_used"]

def test_run_evaluation_partial():
    """Test partial success when models differ."""
    mock_models = [
        {"hf_id": "unsloth/Llama-3-8B", "model_path": "/fake/path/1"},
        {"hf_id": "unsloth/Mistral-7B", "model_path": "/fake/path/2"},
    ]
    with patch("experiment_1629_ebrm_sota.cached_sota_pair", return_value=mock_models):
        result = exp_1629.run_evaluation()
        assert result["honest_verdict"] == "ebrm_sota_validation_partial"

def test_run_evaluation_unavailable():
    """Test failure when models are unavailable."""
    with patch("experiment_1629_ebrm_sota.cached_sota_pair", return_value=None):
        result = exp_1629.run_evaluation()
        assert result["honest_verdict"] == "models_unavailable"
        assert result["models_used"] == []
        
    with patch("experiment_1629_ebrm_sota.cached_sota_pair", return_value=[{"hf_id": "only_one"}]):
        result = exp_1629.run_evaluation()
        assert result["honest_verdict"] == "models_unavailable"

def test_main():
    """Test main function writes deliverable correctly."""
    with patch("experiment_1629_ebrm_sota.run_evaluation") as mock_run:
        mock_run.return_value = {"honest_verdict": "success", "models_used": []}
        
        m_open = mock_open()
        with patch("builtins.open", m_open):
            with patch("os.makedirs"):
                exp_1629.main()
                
        m_open.assert_called_with("results/experiment_1629_ebrm_sota.json", "w")
