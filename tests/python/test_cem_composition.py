import torch
import pytest
from carnot.pipeline.dab_adapter import ComposedDABLogitsProcessor

class DummyEBM:
    def __init__(self, energy_val):
        self.energy_val = energy_val
    def __call__(self, input_ids, scores):
        return torch.full_like(scores, self.energy_val)

def test_composed_dab_logits_processor_sum():
    # REQ-CEM-COMP-01: Support summing energies
    ebm1 = DummyEBM(1.0)
    ebm2 = DummyEBM(2.0)
    processor = ComposedDABLogitsProcessor([ebm1, ebm2], strategy="sum")
    
    scores = torch.zeros((1, 5))
    new_scores = processor(None, scores)
    assert torch.allclose(new_scores, torch.tensor(-3.0))

def test_composed_dab_logits_processor_max():
    # REQ-CEM-COMP-02: Support max energy
    ebm1 = DummyEBM(1.0)
    ebm2 = DummyEBM(2.5)
    processor = ComposedDABLogitsProcessor([ebm1, ebm2], strategy="max")
    
    scores = torch.zeros((1, 5))
    new_scores = processor(None, scores)
    assert torch.allclose(new_scores, torch.tensor(-2.5))

def test_composed_dab_logits_processor_learned():
    # REQ-CEM-COMP-03: Support learned temperature weights
    ebm1 = DummyEBM(1.0)
    ebm2 = DummyEBM(2.0)
    processor = ComposedDABLogitsProcessor([ebm1, ebm2], strategy="learned", weights=[0.5, 0.5])
    
    scores = torch.zeros((1, 5))
    new_scores = processor(None, scores)
    assert torch.allclose(new_scores, torch.tensor(-1.5))

def test_composed_dab_logits_processor_invalid_strategy():
    ebm1 = DummyEBM(1.0)
    processor = ComposedDABLogitsProcessor([ebm1], strategy="invalid")
    scores = torch.zeros((1, 5))
    with pytest.raises(ValueError):
        processor(None, scores)

def test_composed_dab_logits_processor_missing_weights():
    ebm1 = DummyEBM(1.0)
    with pytest.raises(ValueError):
        ComposedDABLogitsProcessor([ebm1], strategy="learned")
