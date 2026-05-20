import pytest
torch = pytest.importorskip("torch")
import json
import os
from carnot.inference.hw_dab import HWDABLogitsProcessor, run_hw_dab_experiment

def test_hw_dab_logits_processor():
    """SCENARIO-INFER-2133-001: Hardware-Assisted DAB updates logits correctly."""
    processor = HWDABLogitsProcessor(lut_size=256)
    input_ids = torch.tensor([[1, 2, 3]])
    scores = torch.tensor([[0.5, 0.2, 0.1, 0.8]])
    
    new_scores = processor(input_ids, scores)
    
    assert new_scores.shape == scores.shape
    # Energy should be subtracted, so new_scores <= scores
    assert torch.all(new_scores <= scores)

def test_hw_dab_experiment(tmp_path):
    """SCENARIO-INFER-2133-001: experiment run outputs JSON correctly."""
    res_path = tmp_path / "experiment_2133_hw_dab.json"
    run_hw_dab_experiment(str(res_path))
    
    assert os.path.exists(res_path)
    with open(res_path, "r") as f:
        data = json.load(f)
        
    assert data["status"] == "success"
    assert data["hw_dab_ready"] is True
    assert "lut_size" in data
