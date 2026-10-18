"""Latent Spills Sensing EBM.

Implements REQ-INFER-3406 for training-free hallucination detection via measuring 'energy spills' in latent representations.
"""

from typing import List

class LatentSpillsDetector:
    """EBM pipeline to read latent activation energies and detect spills."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        
    def calculate_energy_spills(self, latents: List[float]) -> List[float]:
        """Calculates energy spills dynamically per token.
        
        A 'spill' represents unexpected high energy in latent representations,
        signaling guessing or hallucination.
        """
        spills = []
        for latent in latents:
            # Simulated EBM spill calculation: e.g. distance from expected mean
            energy = abs(latent) 
            spills.append(energy)
        return spills
        
    def detect_hallucination(self, latents: List[float]) -> bool:
        """Returns True if a spill exceeds the threshold."""
        spills = self.calculate_energy_spills(latents)
        return any(s > self.threshold for s in spills)
        
    def score_sequence(self, latents: List[float]) -> float:
        """Returns the mean energy spill of the sequence."""
        if not latents:
            return 0.0
        spills = self.calculate_energy_spills(latents)
        return sum(spills) / len(spills)
