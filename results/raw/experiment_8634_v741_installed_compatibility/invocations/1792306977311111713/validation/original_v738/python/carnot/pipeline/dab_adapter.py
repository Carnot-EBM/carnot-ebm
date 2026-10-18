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

class ComposedDABLogitsProcessor(LogitsProcessor):
    """
    Composes multiple EBMs for Dual-Process Alignment.
    Combines energies using sum, max, or learned temperature weights.
    """
    def __init__(self, ebms, strategy="sum", weights=None):
        """
        Args:
            ebms: A list of callable EBMs.
            strategy: String indicating how to combine energies ('sum', 'max', 'learned').
            weights: List of weights corresponding to each EBM, required if strategy is 'learned'.
        """
        self.ebms = ebms
        self.strategy = strategy
        self.weights = weights
        
        if strategy == "learned" and weights is None:
            raise ValueError("weights must be provided for learned strategy")
            
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.ebms:
            return scores
            
        energies = [ebm(input_ids, scores) for ebm in self.ebms]
        
        if self.strategy == "sum":
            total_energy = sum(energies)
        elif self.strategy == "max":
            total_energy = torch.stack(energies).max(dim=0)[0]
        elif self.strategy == "learned":
            total_energy = sum(w * e for w, e in zip(self.weights, energies))
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
        return scores - total_energy

