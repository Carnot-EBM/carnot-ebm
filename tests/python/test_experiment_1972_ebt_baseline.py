"""Tests for Draft EBT Decoding Loop.

Spec: REQ-EBT-1972, SCENARIO-EBT-1972
"""
from carnot.ebt_decoding import EBTDecodingLoop


def test_ebt_decoding_loop_initialization():
    """Verify the EBT decoding loop initializes correctly.
    
    Traces: REQ-EBT-1972
    """
    loop = EBTDecodingLoop(model_hf_id="unsloth/Qwen3.6-35B-A3B-GGUF")
    assert loop.model_hf_id == "unsloth/Qwen3.6-35B-A3B-GGUF"


def test_ebt_decoding_multi_step_energy_minimization():
    """Verify the EBT decoding loop tracks multi-step energy minimization.
    
    Traces: SCENARIO-EBT-1972
    """
    loop = EBTDecodingLoop()
    prompt = "Test prompt"
    result = loop.decode(prompt, max_steps=3)
    
    assert result["prompt"] == prompt
    assert result["model_used"] == loop.model_hf_id
    assert len(result["optimization_history"]) == 3
    
    # Verify energy decreases
    history = result["optimization_history"]
    assert history[0]["best_energy"] > history[1]["best_energy"]
    assert history[1]["best_energy"] > history[2]["best_energy"]


def test_ebt_decoding_batch():
    """Verify batch decoding for multiple prompt cases.
    
    Traces: SCENARIO-EBT-1972
    """
    loop = EBTDecodingLoop()
    prompts = ["Case 1", "Case 2", "Case 3"]
    results = loop.decode_batch(prompts, max_steps=2)
    
    assert len(results) == 3
    assert results[0]["prompt"] == "Case 1"
    assert results[1]["prompt"] == "Case 2"
    assert results[2]["prompt"] == "Case 3"
