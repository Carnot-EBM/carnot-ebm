import os
import json
import sys
import tempfile
from pathlib import Path
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import scripts.experiment_2487 as mod

class MockLlama:
    def __init__(self, mode="normal"):
        self.mode = mode
        self.call_count = 0

    def __call__(self, prompt, max_tokens=200, logprobs=5):
        self.call_count += 1
        # Generate fake logprobs.
        # If in PRC prompt (call_count <= 20) and we want elevated energy (more variance/uncertainty), 
        # we give lower probabilities for top tokens, which makes them closer together.
        # But wait, the energy is computed from raw logits. We just pass `top_logprobs`.
        # `top_logprobs_to_logit_vector` flattens them. 
        # Let's just create some valid top_logprobs so it doesn't crash.
        
        # We need a list of position dictionaries.
        # For simplicity, 2 positions, each with 5 logprobs.
        if self.mode == "elevated" and self.call_count <= 20:
            # PRC topic -> higher energy. Energy is -temp * log(sum(exp(logits))). 
            # Wait, `SemanticEnergyDetector` computes mean energy magnitude. 
            # We want energy to be higher.
            logprob_dict = {"a": -0.1, "b": -0.2, "c": -0.3, "d": -0.4, "e": -0.5}
        else:
            # Neutral topic -> lower energy.
            logprob_dict = {"a": -1.0, "b": -2.0, "c": -3.0, "d": -4.0, "e": -5.0}

        return {
            "choices": [
                {
                    "logprobs": {
                        "top_logprobs": [logprob_dict, logprob_dict]
                    }
                }
            ]
        }


def test_deliverable_has_required_schema_fields(tmp_path, monkeypatch):
    """End-to-end run writes deliverable JSON with all required ARTIFACT FIELDS."""
    
    # Redirect deliverable to a temp location so we don't clobber results/.
    original_deliverable = mod.DELIVERABLE
    temp_deliverable = str(tmp_path / "experiment_2487.json")
    monkeypatch.setattr(mod, "DELIVERABLE", temp_deliverable)

    mock_llm = MockLlama(mode="elevated")
    mod.main(mock_llm=mock_llm, mock_time=100.0)

    assert Path(temp_deliverable).exists(), "Deliverable JSON not written"

    with open(temp_deliverable) as f:
        artifact = json.load(f)

    required_fields = [
        "energy_prc_mean",
        "energy_neutral_mean",
        "prc_energy_elevated",
        "phase4_validated_via_prc",
        "duration_s",
        "model_used",
        "honest_verdict",
    ]
    for field in required_fields:
        assert field in artifact, f"Missing required field: {field}"

    assert isinstance(artifact["energy_prc_mean"], float)
    assert isinstance(artifact["energy_neutral_mean"], float)
    assert isinstance(artifact["prc_energy_elevated"], bool)
    assert isinstance(artifact["phase4_validated_via_prc"], bool)
    assert artifact["duration_s"] >= 60.0
    assert "complete:" in artifact["honest_verdict"]
