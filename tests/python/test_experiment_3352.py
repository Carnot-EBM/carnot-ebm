import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Pre-import torch and transformers to bypass memory leak checks inside the test
try:
    import torch
    import transformers
except ImportError:
    pass

# Mock llama_cpp BEFORE importing the loader or experiment script
sys.modules['llama_cpp'] = MagicMock()

from scripts.experiment_3352_constrained_generation import main, extract_compute_lines
from carnot.pipeline.gemma4_quantized_loader import Gemma4QuantizedLoader

def test_extract_compute_lines():
    res = extract_compute_lines("Hello\nCOMPUTE: 2 + 2 = 4\nWorld")
    assert res == ["2 + 2 = 4"]

def test_gemma_loader_generate_grammar_stub():
    loader = Gemma4QuantizedLoader(model_path="")
    loader.load()
    
    # Unconstrained
    res = loader.generate("prompt")
    assert "42" in res
    
    # Constrained
    res2 = loader.generate("prompt", grammar_string='root ::= "COMPUTE:"\n')
    assert "COMPUTE:" in res2

@patch("scripts.experiment_3352_constrained_generation.ExperimentTemplate.assert_deliverable_written")
@patch("scripts.experiment_3352_constrained_generation.cached_sota_pair")
def test_experiment_main(mock_cached, mock_assert, tmp_path):
    mock_cached.return_value = [{"name": "mock", "hf_id": "mock/hf", "gpu": 0, "model_path": None}]
    
    with patch("scripts.experiment_template._get_repo_root", return_value=tmp_path):
        main()
        
    deliverable = tmp_path / "results" / "experiment_3352_constrained_generation.json"
    assert deliverable.exists()
    data = json.loads(deliverable.read_text())
    assert "unconstrained_metrics" in data
    assert "constrained_metrics" in data
    assert data["status"] == "success"
