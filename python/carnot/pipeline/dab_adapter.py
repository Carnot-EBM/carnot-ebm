import torch
from transformers import LogitsProcessor

class CarnotDABLogitsProcessor(LogitsProcessor):
    """
    DAB adapter for HuggingFace LogitsProcessor.
    Constrains LLMs with EBMs at generation time by subtracting state energy from logits.
    """
    def __init__(self, ebm):
        """
        Args:
            ebm: A callable EBM that takes (input_ids, scores) and returns energy values
                 matching the shape of scores.
        """
        self.ebm = ebm
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        energy = self.ebm(input_ids, scores)
        return scores - energy
