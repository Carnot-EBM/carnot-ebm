"""TSUSampler interface that mocks the thrml SDK API.

Spec: REQ-SAMPLE-2059
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from carnot.samplers.backend import SamplerBackend

logger = logging.getLogger(__name__)


@dataclass
class TSUSampler(SamplerBackend):
    """Integration stub for the thrml SDK that mocks the API.

    Spec: REQ-SAMPLE-2059
    """

    seed: int = 42

    @property
    def backend_name(self) -> str:
        return "thrml_tsu"

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """Run mock minimize_energy as if calling thrml SDK."""
        logger.info("TSUSampler.minimize_energy called (mocking thrml API).")
        rng = np.random.default_rng(self.seed)
        n_spins = biases.shape[0]
        return rng.integers(0, 2, size=(n_samples, n_spins)).astype(bool)

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Run mock sample as if calling thrml SDK."""
        logger.info("TSUSampler.sample called (mocking thrml API).")
        rng = np.random.default_rng(self.seed)
        n_spins = biases.shape[0]
        return rng.integers(0, 2, size=(n_samples, n_spins)).astype(bool)
