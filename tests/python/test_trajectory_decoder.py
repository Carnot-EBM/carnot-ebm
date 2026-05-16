"""
Tests for trajectory decoder.
Traces to REQ-INFER-2073 and SCENARIO-INFER-2073.
"""

import json
import os
import pytest
import numpy as np

from carnot.inference.trajectory_decoder import run_experiment, MODEL_SPECS, TrajectoryDecoder

def test_model_specs():
    """Test that the required models are defined. Traces to REQ-INFER-2073."""
    assert 'unsloth/Qwen3.6-35B-A3B-GGUF' in MODEL_SPECS
    assert 'unsloth/gemma-4-31B-it-GGUF' in MODEL_SPECS

def test_trajectory_decoder_loop():
    """Test the trajectory decoding loop. Traces to SCENARIO-INFER-2073."""
    decoder = TrajectoryDecoder(vocab_size=100)
    
    # Mock continuous state
    continuous_state = np.array([[0.1, -0.2, 0.5]])
    
    probabilities = decoder.decode(continuous_state)
    
    # Probabilities should be normalized
    np.testing.assert_allclose(np.sum(probabilities), 1.0, rtol=1e-5)
    assert probabilities.shape == (1, 100)

def test_experiment_artifact(tmp_path):
    """Test that the experiment saves the artifact. Traces to REQ-INFER-2073."""
    output_file = tmp_path / "experiment_2073_trajectory_decoder.json"
    
    # Run experiment
    run_experiment(str(output_file))
    
    # Check artifact
    assert output_file.exists()
    
    with open(output_file, 'r') as f:
        data = json.load(f)
    
    assert data["status"] == "complete"
    assert "trajectory_decoder_tested" in data
    assert data["trajectory_decoder_tested"] is True
