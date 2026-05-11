import os
import json
import pytest
from carnot.training.moe_distill import OnlineReplayBuffer, run_distillation_experiment, MODEL_SPECS

def test_req_learn_1820_model_specs():
    """Test REQ-LEARN-1820: MODEL_SPECS MUST include unsloth/Qwen3.6-35B-A3B-GGUF."""
    assert "unsloth/Qwen3.6-35B-A3B-GGUF" in MODEL_SPECS

def test_req_learn_1820_1_replay_buffer():
    """Test REQ-LEARN-1820-1: moe_distill.py SHALL implement the replay buffer and fine-tune router logits."""
    buffer = OnlineReplayBuffer(capacity=2)
    buffer.add({"state": "A", "reward": 1.0})
    buffer.add({"state": "B", "reward": 0.5})
    buffer.add({"state": "C", "reward": 0.0})
    
    # Capacity is 2, so A should be popped
    assert len(buffer.buffer) == 2
    assert buffer.buffer[0]["state"] == "B"
    
    # Sample test
    samples = buffer.sample(1)
    assert len(samples) == 1
    
    samples_all = buffer.sample(3)
    assert len(samples_all) == 2  # Max capacity is 2
    
    # Fine-tune test
    loss = buffer.fine_tune_router({"state": "test", "reward": 0.6})
    assert isinstance(loss, float)

def test_req_learn_1820_2_experiment_logging(tmp_path):
    """Test REQ-LEARN-1820-2: The artifact SHALL include required schema fields."""
    output_path = tmp_path / "experiment_1820_moe_distill.json"
    results = run_distillation_experiment(str(output_path))
    
    assert os.path.exists(output_path)
    with open(output_path, "r") as f:
        saved_results = json.load(f)
        
    assert "distillation_loss" in saved_results
    assert "honest_verdict" in saved_results
    assert saved_results["honest_verdict"] == "distillation_logged"
    assert saved_results["model_specs"] == MODEL_SPECS
    assert results == saved_results
