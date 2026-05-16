import torch
from carnot.pipeline.dab_adapter import CarnotDABLogitsProcessor

class DummyEBM:
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Return a constant energy of 0.5 for all logits
        return torch.ones_like(scores) * 0.5

def test_dab_adapter():
    ebm = DummyEBM()
    adapter = CarnotDABLogitsProcessor(ebm)
    
    input_ids = torch.tensor([[1, 2, 3]])
    scores = torch.tensor([[0.5, 0.8, -0.2]])
    
    adjusted_scores = adapter(input_ids, scores)
    
    expected = torch.tensor([[0.0, 0.3, -0.7]])
    
    torch.testing.assert_close(adjusted_scores, expected)
