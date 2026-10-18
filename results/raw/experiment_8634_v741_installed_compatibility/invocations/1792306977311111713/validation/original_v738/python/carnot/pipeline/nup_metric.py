import numpy as np
from typing import List, Dict, Any

class NUPMetric:
    """
    NUP Metric to detect symmetry breaking in the gradient landscape
    as an early hallucination indicator.
    """
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        
    def gradients_to_ising_energy(self, gradients: List[np.ndarray]) -> List[float]:
        """
        Maps gradients to an Ising energy configuration.
        Treats the sign of the gradient as an Ising spin (-1 or +1),
        and computes the energy as the negative dot product (normalized)
        between consecutive spins.
        """
        if not gradients or len(gradients) < 2:
            return []
            
        energies = []
        for i in range(len(gradients) - 1):
            spin1 = np.sign(gradients[i])
            spin1[spin1 == 0] = 1.0
            spin2 = np.sign(gradients[i+1])
            spin2[spin2 == 0] = 1.0
            
            energy = -np.dot(spin1.flatten(), spin2.flatten()) / spin1.size
            energies.append(float(energy))
            
        return energies

    def detect_phase_transition(self, energies: List[float]) -> bool:
        """
        Measures sudden energy shifts (phase transitions).
        """
        if len(energies) < 2:
            return False
            
        for i in range(len(energies) - 1):
            shift = abs(energies[i+1] - energies[i])
            if shift >= self.threshold:
                return True
        return False

    def evaluate(self, gradients: List[np.ndarray]) -> Dict[str, Any]:
        """
        Evaluates the gradients and returns a dictionary with the results.
        """
        energies = self.gradients_to_ising_energy(gradients)
        is_hallucination = self.detect_phase_transition(energies)
        return {
            "energies": energies,
            "hallucination_detected": is_hallucination
        }
