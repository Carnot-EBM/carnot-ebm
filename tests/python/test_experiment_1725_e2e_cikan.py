"""Tests for Exp 1725.

Spec traces: REQ-LEARN-1725, SCENARIO-LEARN-1725
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import scripts.experiment_1725_e2e_cikan as exp


def test_experiment_1725_e2e_cikan(tmp_path: Path) -> None:
    mock_results_path = tmp_path / "experiment_1725_e2e_cikan.json"
    
    with patch("scripts.experiment_1725_e2e_cikan.Path") as mock_path_cls:
        mock_path_obj = MagicMock()
        mock_path_obj.parent.mkdir = MagicMock()
        mock_path_obj.open = mock_results_path.open
        mock_path_cls.return_value = mock_path_obj
        
        exp.run_experiment()
            
    assert mock_results_path.exists()
    
    with mock_results_path.open("r") as f:
        data = json.load(f)
        
    assert data["experiment_id"] == "1725"
    assert data["model_used"] in ["unsloth/Qwen3.6-35B-A3B-GGUF", "unsloth/gemma-4-31B-it-GGUF"]
    assert data["n_processed"] == 50
    assert "adaptation_rate" in data
    assert data["honest_verdict"] == "e2e_pipeline_successful"
