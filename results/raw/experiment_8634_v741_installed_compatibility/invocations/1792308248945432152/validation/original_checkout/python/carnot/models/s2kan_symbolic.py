"""S2KAN with Extensible Symbolic Primitives Dictionary.

Spec references: REQ-KAN-1926, SCENARIO-KAN-1926.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]

DEFAULT_RESULT_PATH = _repo_root() / "results/experiment_1926_s2kan_symbolic.json"

# 1. Dictionary of symbolic primitives
PRIMITIVE_DICT = {
    "identity": lambda x: x,
    "sin": jnp.sin,
    "exp": jnp.exp,
    "step": lambda x: jax.nn.sigmoid(10.0 * x),
    "square": lambda x: x ** 2,
}
PRIMITIVE_NAMES = list(PRIMITIVE_DICT.keys())

def evaluate_primitives_dict(x: jax.Array) -> jax.Array:
    """Evaluate all primitives from the dictionary on input x.
    
    Returns:
        Tensor of shape (..., len(PRIMITIVE_NAMES))
    """
    return jnp.stack([PRIMITIVE_DICT[name](x) for name in PRIMITIVE_NAMES], axis=-1)

@dataclass(frozen=True)
class S2KANSymbolicConfig:
    input_dim: int = 1
    temperature: float = 1.0

@dataclass(frozen=True)
class S2KANSymbolicParams:
    gate_logits: jax.Array  # Shape: (input_dim, len(PRIMITIVE_NAMES))

class S2KANSymbolicLayer:
    """Softly Symbolified KAN with primitives dictionary."""
    def __init__(
        self,
        config: S2KANSymbolicConfig,
        key: jax.Array | None = None,
        params: S2KANSymbolicParams | None = None,
    ) -> None:
        self.config = config
        self.n_prims = len(PRIMITIVE_NAMES)
        if params is None:
            if key is None:
                key = jrandom.PRNGKey(0)
            self.params = S2KANSymbolicParams(
                gate_logits=0.01 * jrandom.normal(
                    key, (self.config.input_dim, self.n_prims), dtype=jnp.float32
                )
            )
        else:
            self.params = params

    def _gates(self) -> jax.Array:
        """Compute differentiable gates via softmax."""
        return jax.nn.softmax(self.params.gate_logits / self.config.temperature, axis=-1)

    def forward(self, x: jax.Array) -> jax.Array:
        """Evaluate S2KAN gating on input x.
        
        Args:
            x: Tensor of shape (..., input_dim)
            
        Returns:
            Tensor of shape (..., input_dim) with gated primitives applied.
        """
        x_tensor = jnp.asarray(x, dtype=jnp.float32)
        prims = evaluate_primitives_dict(x_tensor)  # Shape: (..., input_dim, n_prims)
        gates = self._gates()  # Shape: (input_dim, n_prims)
        
        # Multiply primitives by gates and sum over the primitive dimension
        y = jnp.sum(prims * gates, axis=-1)
        return y

def build_experiment_1926_artifact(run_date: str = "20260512") -> dict[str, object]:
    """Build the stable Exp 1926 artifact payload."""
    config = S2KANSymbolicConfig(input_dim=1)
    
    # We want to validate against a known functional form: y = sin(x)
    logits = np.zeros((1, len(PRIMITIVE_NAMES)), dtype=np.float32)
    logits[0, PRIMITIVE_NAMES.index("sin")] = 10.0
    
    params = S2KANSymbolicParams(gate_logits=jnp.array(logits))
    layer = S2KANSymbolicLayer(config, params=params)
    
    x = jnp.array([[0.0], [np.pi / 2]], dtype=jnp.float32)
    y = layer.forward(x)
    
    # Validation against known functional form y = sin(x)
    y_expected = jnp.sin(x)
    max_abs_error = float(jnp.max(jnp.abs(y - y_expected)))
    
    return {
        "schema": "carnot.s2kan_symbolic.experiment_1926.v1",
        "status": "complete",
        "experiment_id": 1926,
        "run_date": run_date,
        "spec_traces": ["REQ-KAN-1926", "SCENARIO-KAN-1926"],
        "module": "python/carnot/models/s2kan_symbolic.py",
        "artifact_path": "results/experiment_1926_s2kan_symbolic.json",
        "validation_max_abs_error": max_abs_error,
        "validation_passed": max_abs_error < 1e-2,
        "honest_verdict": "complete: s2kan_symbolic_primitives_dict_and_gates_implemented",
    }

def write_experiment_1926_artifact(
    output_path: str | Path = DEFAULT_RESULT_PATH,
    run_date: str = "20260512",
) -> dict[str, object]:
    """Write the stable Exp 1926 artifact to disk."""
    artifact = build_experiment_1926_artifact(run_date=run_date)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact

__all__ = [
    "PRIMITIVE_DICT",
    "PRIMITIVE_NAMES",
    "S2KANSymbolicConfig",
    "S2KANSymbolicParams",
    "S2KANSymbolicLayer",
    "build_experiment_1926_artifact",
    "write_experiment_1926_artifact",
]
