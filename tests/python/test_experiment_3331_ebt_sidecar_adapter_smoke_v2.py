import json
import pytest
from pathlib import Path
from unittest.mock import patch
import sys
import importlib

import scripts.experiment_3331_ebt_sidecar_adapter_smoke_v2 as exp_module

def test_experiment_3331_creates_artifact(tmp_path):
    # Mock the result path
    result_path = tmp_path / "results" / "experiment_3331_ebt_sidecar_adapter_smoke_v2.json"
    
    with patch("scripts.experiment_3331_ebt_sidecar_adapter_smoke_v2.Path") as mock_path:
        # Mock Path so it uses tmp_path for the specific file but works normally otherwise
        def side_effect(*args, **kwargs):
            p = Path(*args, **kwargs)
            if p.name == "experiment_3331_ebt_sidecar_adapter_smoke_v2.json":
                return result_path
            return p
            
        mock_path.side_effect = side_effect
        
        # Run main
        exp_module.main()
        
    assert result_path.exists()
    
    data = json.loads(result_path.read_text())
    assert data["honest_verdict"] == "sidecar_ready"
    assert data["inference_substrate"] == "local_cpu_sidecar_replay"
    assert data["ebt_source_status"] == "vendored_local"
    assert data["adapter_ready"] is True
    assert data["diagnostic_rank_metric"] == 1.0
    assert data["n_cases"] > 0
    assert data["claim_boundary"] == "sidecar_diagnostic_only"
    assert "proposal_ranking_diagnostic" in data["useful_for"]
    assert "exp3328_style_proposal_ranking" in data["useful_for"]

def test_experiment_3331_handles_import_error(tmp_path):
    result_path = tmp_path / "results" / "experiment_3331_ebt_sidecar_adapter_smoke_v2.json"
    
    with patch("scripts.experiment_3331_ebt_sidecar_adapter_smoke_v2.Path") as mock_path:
        def side_effect(*args, **kwargs):
            p = Path(*args, **kwargs)
            if p.name == "experiment_3331_ebt_sidecar_adapter_smoke_v2.json":
                return result_path
            return p
            
        mock_path.side_effect = side_effect
        
        # Simulate import error for carnot.models.ebt by temporarily removing it
        # The script does: from carnot.models.ebt import EBTConfig, EBTransformer
        # We can mock builtins.__import__
        original_import = __import__
        
        def mock_import(name, *args, **kwargs):
            if name == "carnot.models.ebt":
                raise ImportError("Mocked import error")
            return original_import(name, *args, **kwargs)
            
        with patch("builtins.__import__", side_effect=mock_import):
            exp_module.main()
            
    assert result_path.exists()
    data = json.loads(result_path.read_text())
    assert data["ebt_source_status"] == "import_failed"
    assert data["adapter_ready"] is False
    assert data["honest_verdict"] == "blocked"
    assert "Mocked import error" in data["blocked_reasons"][0]
