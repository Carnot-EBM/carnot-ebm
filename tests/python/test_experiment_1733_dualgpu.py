"""Tests for Exp 1733: DualGPU Benchmark.

Spec traces: REQ-DUALGPU-101, SCENARIO-DUALGPU-101
"""

import json
from pathlib import Path
import importlib.util
import sys

def load_script(path_str):
    spec = importlib.util.spec_from_file_location("exp_script", path_str)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_experiment_1733_dualgpu(tmp_path):
    """Test the DualGPU benchmark script."""
    script_path = Path("scripts/experiment_1733_dualgpu.py")
    exp_module = load_script(str(script_path))
    
    exp_module.run_benchmark(output_dir=str(tmp_path))
    
    out_file = tmp_path / "experiment_1733_dualgpu.json"
    assert out_file.exists()
    
    with out_file.open() as f:
        data = json.load(f)
        
    assert data["experiment_id"] == "1733"
    assert data["success"] is True
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in data["models_utilized"]
    assert "unsloth/gemma-4-31B-it-GGUF" in data["models_utilized"]
    assert data["components"]["system2_eqm_score"] == 0.95
