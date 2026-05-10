"""Energy-Based Transformer (EBT) abstraction layer.

Spec: REQ-INFER-018
"""

from typing import Any, Protocol


class LogProbModel(Protocol):
    def get_sequence_logprob(self, text: str) -> float:
        ...


class EBTBridge:
    """Bridges autoregressive models to an EBT formulation."""
    
    def __init__(self, model: Any) -> None:
        """Initialize the EBTBridge.
        
        Args:
            model: An autoregressive model that can provide sequence log probabilities.
                   It should expose a `get_sequence_logprob(text: str) -> float` method.
        """
        self.model = model

    def sequence_energy(self, text: str) -> float:
        """Calculate sequence energy.
        
        Higher probability sequences are assigned lower energy.
        Energy is calculated as the negative log probability.
        
        Args:
            text: The sequence to calculate energy for.
            
        Returns:
            The calculated energy value.
        """
        logprob = self.model.get_sequence_logprob(text)
        return float(-logprob)
