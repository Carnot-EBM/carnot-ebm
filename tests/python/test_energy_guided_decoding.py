"""
Tests for Energy-Guided Decoding (EGD) Wrapper.

Spec traces: REQ-PIPELINE-1670, SCENARIO-PIPELINE-1670
"""

import json
import os
import pytest
from carnot.pipeline.energy_guided_decoding import EGDWrapper, run_experiment_1670

def test_egd_wrapper_selection():
    """
    Test that EGDWrapper correctly applies EGD selection across inference calls.
    Spec traces: REQ-PIPELINE-1670
    """
    def mock_inference(prompt):
        # Return mock candidates and their base probabilities or raw text
        return ["Candidate A", "Candidate B"]
    
    def mock_energy_fn(candidate):
        # Candidate A has higher energy (worse), Candidate B has lower energy (better)
        energies = {"Candidate A": 10.0, "Candidate B": 2.0}
        return energies.get(candidate, 100.0)

    wrapper = EGDWrapper(
        model_name="unsloth/gemma-4-31B-it-GGUF",
        inference_fn=mock_inference,
        energy_fn=mock_energy_fn
    )

    result = wrapper.generate_with_egd("Test prompt")
    # EGD should select the candidate with minimal energy
    assert result == "Candidate B"

def test_egd_wrapper_empty_candidates():
    """
    Test that EGDWrapper handles empty candidates gracefully.
    Spec traces: REQ-PIPELINE-1670
    """
    wrapper = EGDWrapper(
        model_name="unsloth/gemma-4-31B-it-GGUF",
        inference_fn=lambda p: [],
        energy_fn=lambda c: 0.0
    )
    result = wrapper.generate_with_egd("Test prompt")
    assert result == ""

def test_experiment_1670_artifact(tmp_path):
    """
    Test that the experiment writes the correct artifact and evaluates Yes-ratio bias.
    Spec traces: SCENARIO-PIPELINE-1670
    """
    output_file = tmp_path / "experiment_1670_egd.json"
    
    # Run the experiment pointing to our temporary output file
    run_experiment_1670(output_path=str(output_file))
    
    assert os.path.exists(output_file)
    with open(output_file, "r") as f:
        data = json.load(f)
        
    assert "yes_ratio_bias" in data
    assert "status" in data
    assert data["status"] == "complete"
    assert "model_specs" in data
    assert data["model_specs"] == "unsloth/gemma-4-31B-it-GGUF"
    assert "hallucination_evaluated" in data
    assert data["hallucination_evaluated"] is True
    assert data["yes_ratio_bias"] == 0.25  # 1 yes out of 4 cases
