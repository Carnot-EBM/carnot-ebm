from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from carnot.phase3.continuous_ebm import ContinuousEBM

@dataclass
class EnergyVector:
    """Energy vector representing continuous latent constraints in CLaRa-V."""
    components: np.ndarray

    @property
    def total_energy(self) -> float:
        """Returns the sum of all energy components."""
        return float(np.sum(self.components))

@dataclass
class ContinuousLatentState:
    """Continuous latent state for CLaRa-V continuous reasoning."""
    z: np.ndarray
    energy: EnergyVector

    @classmethod
    def from_dimensions(cls, dim: int) -> ContinuousLatentState:
        """Initialize a zero-energy state with zeros."""
        z = np.zeros(dim, dtype=np.float64)
        energy_components = np.zeros(dim, dtype=np.float64)
        return cls(z=z, energy=EnergyVector(components=energy_components))

    def evaluate_ebm_energy(self, ebm: ContinuousEBM) -> float:
        """Evaluate the energy of the latent state using a ContinuousEBM."""
        if self.z.shape[0] != ebm.variables:
            raise ValueError(f"State dimension {self.z.shape[0]} does not match EBM variables {ebm.variables}")
        
        # E(x) = -0.5 * x^T * J * x - h^T * x
        j_term = -0.5 * self.z @ ebm.coupling @ self.z
        h_term = -ebm.bias @ self.z
        return float(j_term + h_term)
