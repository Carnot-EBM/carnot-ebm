import json
import pytest
from pathlib import Path
from carnot.pipeline.ets_decoder import ETSDecoder

def test_ets_decoder_spec_scenario():
    """
    Tests SCENARIO-PIPELINE-1615: ETS Decoder Reduces Energy
    Given a set of candidate tokens with base LLM probabilities
    When ETSDecoder.decode() is run with a mock energy function and Monte Carlo sampling
    Then the selected token maximizes the combined probability p_llm * exp(-beta * E_mc).
    """
    # Equal base probabilities
    base_probs = {"A": 0.5, "B": 0.5}
    
    # Mock energy function: 'A' has high energy (bad), 'B' has low energy (good)
    def mock_energy_fn(seq: str) -> float:
        if "A" in seq:
            return 10.0
        return 1.0

    decoder = ETSDecoder(base_policy=base_probs, energy_fn=mock_energy_fn, mc_samples=5, beta=1.0)
    best_token = decoder.decode(prefix="Context ")
    
    # Despite equal probabilities, B should be chosen due to lower energy
    assert best_token == "B", f"Expected B, got {best_token}"
    assert decoder.last_stats["selected_token"] == "B"

def test_ets_decoder_artifact_generation():
    """
    Tests REQ-PIPELINE-1615 artifact requirement.
    Returns a JSON artifact to results/experiment_1615_ets_decoding.json containing metrics.
    """
    # Unequal probabilities, but energy overrides
    base_probs = {"token1": 0.8, "token2": 0.2}
    
    def mock_energy_fn(seq: str) -> float:
        # token1 violates constraints
        if "token1" in seq:
            return 50.0
        return 0.0

    decoder = ETSDecoder(base_policy=base_probs, energy_fn=mock_energy_fn, mc_samples=10, beta=2.0)
    decoder.decode(prefix="Q: 2+2=? A: ")
    
    artifact_path = "results/experiment_1615_ets_decoding.json"
    decoder.save_artifact(artifact_path)
    
    assert Path(artifact_path).exists()
    
    with open(artifact_path, "r") as f:
        data = json.load(f)
        
    assert data["status"] == "complete"
    assert data["honest_verdict"] == "ets_decoding_successful"
    assert data["mc_samples_per_token"] == 10
    assert data["selected_token"] == "token2"
    assert "token_stats" in data
    assert "token1" in data["token_stats"]
    assert "token2" in data["token_stats"]
