"""Energy Based Model implementations: Ising, Gibbs, Boltzmann tiers."""

from carnot.models.boltzmann import BoltzmannConfig, BoltzmannModel
from carnot.models.ising import IsingConfig, IsingModel

__all__ = [
    "BoltzmannConfig",
    "BoltzmannModel",
    "IsingConfig",
    "IsingModel",
]
