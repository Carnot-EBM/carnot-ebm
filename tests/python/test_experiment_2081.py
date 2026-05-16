"""Tests for Exp 2081 Dual RTX GSM8K Benchmark.

Spec: REQ-SAMPLE-2081
"""
import os
import json
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add scripts directory to sys.path so we can import the script
scripts_dir = Path(__file__).parent.parent.parent / "scripts"
sys.path.append(str(scripts_dir))

from experiment_2081_dual_rtx_gsm8k import run_benchmark

@patch("experiment_2081_dual_rtx_gsm8k.SGLRWSampler")
@patch("experiment_2081_dual_rtx_gsm8k.jax.devices")
def test_experiment_2081_produces_artifact(mock_devices, mock_sampler_class):
    """
    SCENARIO-SAMPLE-2081: SGLRW Dual RTX Benchmark Writes Artifact
    """
    mock_device = MagicMock()
    mock_device.platform = "gpu"
    mock_devices.return_value = [mock_device, mock_device]
    
    mock_sampler = MagicMock()
    mock_sampler_class.return_value = mock_sampler
    mock_sampler.sample.return_value = None

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = os.path.join(tmpdir, "experiment_2081_dual_rtx_benchmark.json")
        artifact = run_benchmark(artifact_path)
        
        assert os.path.exists(artifact_path)
        with open(artifact_path) as f:
            data = json.load(f)
            
        assert data["experiment_id"] == "2081"
        assert data["status"] == "completed"
        assert "hardware_latency_ms" in data
        assert "generative_accuracy" in data
        assert data["model"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
