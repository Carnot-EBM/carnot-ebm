"""Tests for E2E Cascade Experiment 1980.

References: REQ-1980-E2E-CASCADE, SCENARIO-1980-E2E-CASCADE
"""

import os
import json
import pytest
from pathlib import Path
from unittest.mock import patch
import jax.numpy as jnp

from carnot.ebt_decoding import EBTDecodingLoop
from carnot.pipeline.z3_validator import Z3Validator
from carnot.pipeline.continuous_self_learner import ContinuousSelfLearner

def test_experiment_1980_components():
    """Test that all three components can be instantiated and used properly."""
    model_id = "unsloth/Qwen3.6-35B-A3B-GGUF"
    
    # 1. EBT Decoding Loop
    loop = EBTDecodingLoop(model_hf_id=model_id)
    assert loop.model_hf_id == model_id
    res = loop.decode("Test query", max_steps=1)
    assert res["prompt"] == "Test query"
    
    # 2. Z3Validator
    validator = Z3Validator()
    constraints = [{"type": "lower_bound", "target": "X", "value": 0.0}]
    assignment = {"X": 5.0}
    assert validator.validate(constraints, assignment) is True
    
    # 3. ContinuousSelfLearner
    learner = ContinuousSelfLearner(model_name=model_id)
    assert learner.model_name == model_id
    scenarios = [jnp.array([2.0, 2.0, 2.0])]
    deltas = learner.process_scenarios(scenarios)
    assert len(deltas) == 1
    assert isinstance(deltas[0], float)

@patch("scripts.experiment_template.ExperimentTemplate.assert_deliverable_written")
def test_experiment_1980_script(mock_assert, tmp_path, monkeypatch):
    """Test the full script logic without writing to the real results dir."""
    import scripts.experiment_1980 as exp1980
    
    deliverable_path = tmp_path / "experiment_1980_e2e_cascade.json"
    
    with patch("scripts.experiment_1980.ExperimentTemplate") as mock_tmpl_class:
        mock_tmpl = mock_tmpl_class.return_value
        
        class MockPhase:
            def __init__(self, name):
                self.name = name
            def __enter__(self):
                pass
            def __exit__(self, *args):
                pass
                
        mock_tmpl.phase = MockPhase
        mock_tmpl._output_path = deliverable_path
        mock_tmpl.build_result.return_value = {"mock": "artifact"}
        mock_tmpl.assert_deliverable_written = mock_assert
        
        # run
        exp1980.main()
        
        mock_tmpl.build_result.assert_called_once()
        args, kwargs = mock_tmpl.build_result.call_args
        assert kwargs["status"] == "success"
        data = args[0]
        assert data["model_used"] == "unsloth/Qwen3.6-35B-A3B-GGUF"
        assert data["queries_processed"] == 5
        assert len(data["ebt_results"]) == 5
        assert len(data["formal_validation_results"]) == 5
        assert len(data["continuous_learning_deltas"]) == 5
