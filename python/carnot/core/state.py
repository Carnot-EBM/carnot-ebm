"""Model state and configuration.

Spec: REQ-CORE-003, REQ-CORE-004
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from safetensors.numpy import load_file, save_file


@dataclass
class ModelConfig:
    """Model hyperparameters.

    Spec: REQ-CORE-003
    """

    input_dim: int
    hidden_dims: list[int] = field(default_factory=list)
    precision: str = "f32"  # "f32" or "f64"


@dataclass
class ModelMetadata:
    """Training metadata.

    Spec: REQ-CORE-003
    """

    step: int = 0
    loss_history: list[float] = field(default_factory=list)


@dataclass
class ModelState:
    """Complete model state for serialization.

    Spec: REQ-CORE-003, REQ-CORE-004
    """

    parameters: dict[str, jnp.ndarray]
    config: ModelConfig
    metadata: ModelMetadata = field(default_factory=ModelMetadata)

    def save(self, path: Path) -> None:
        """Save model state to disk.

        Parameters saved as safetensors, metadata as JSON sidecar.
        Spec: REQ-CORE-004, SCENARIO-CORE-004
        """
        path = Path(path)
        # Save parameters as safetensors
        np_params = {k: np.asarray(v) for k, v in self.parameters.items()}
        save_file(np_params, str(path / "model.safetensors"))

        # Save config + metadata as JSON
        meta = {
            "config": {
                "input_dim": self.config.input_dim,
                "hidden_dims": self.config.hidden_dims,
                "precision": self.config.precision,
            },
            "metadata": {
                "step": self.metadata.step,
                "loss_history": self.metadata.loss_history,
            },
        }
        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> ModelState:
        """Load model state from disk.

        Spec: REQ-CORE-004, SCENARIO-CORE-004
        """
        path = Path(path)
        np_params = load_file(str(path / "model.safetensors"))
        parameters = {k: jnp.array(v) for k, v in np_params.items()}

        with open(path / "metadata.json") as f:
            meta = json.load(f)

        config = ModelConfig(**meta["config"])
        metadata = ModelMetadata(**meta["metadata"])

        return cls(parameters=parameters, config=config, metadata=metadata)
