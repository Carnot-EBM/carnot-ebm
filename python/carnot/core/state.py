"""Model state and configuration — persistence layer for EBM parameters.

**Researcher summary:**
    Dataclasses for model hyperparameters (``ModelConfig``), training metadata
    (``ModelMetadata``), and serializable state (``ModelState``). Uses safetensors
    for cross-language-compatible parameter storage (Rust <-> Python).

**Detailed explanation for engineers:**
    An Energy-Based Model has learned parameters (e.g., coupling matrices, biases,
    neural network weights) that define the energy landscape. To save and reload
    models — or to share them between the Rust and Python implementations — we
    need a standard serialization format.

    This module provides three dataclasses:

    1. ``ModelConfig``: Hyperparameters that define the model architecture (input
       dimension, hidden layer sizes, numeric precision). These don't change
       during training.

    2. ``ModelMetadata``: Training progress info (current step, loss history).
       Useful for resuming training or inspecting convergence.

    3. ``ModelState``: The complete package — parameters + config + metadata.
       Has ``save()`` and ``load()`` methods that write/read from disk.

    **Why safetensors?**
    The safetensors format (by Hugging Face) stores tensors in a simple, safe,
    memory-mappable binary format. Unlike pickle, it cannot execute arbitrary
    code on load. Crucially, it is language-agnostic — the Rust crate
    ``safetensors`` can read files written by the Python library and vice versa.
    This is how Carnot achieves cross-language model interoperability.

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
    """Model hyperparameters that define the architecture.

    **Researcher summary:**
        Stores input_dim, hidden_dims, and precision. Immutable after creation.

    **Detailed explanation for engineers:**
        These are the architectural choices that must be known before creating
        the model. They are saved alongside the parameters so that a loaded
        model can be reconstructed with the exact same architecture.

    Attributes:
        input_dim: Number of input features / visible units. For example, 784
            for a 28x28 image flattened to a vector, or 81 for a Sudoku grid.
        hidden_dims: List of hidden layer sizes for multi-layer models (Gibbs,
            Boltzmann tiers). Empty list for the Ising tier, which has no
            hidden layers.
        precision: Floating-point precision — "f32" for 32-bit (default) or
            "f64" for 64-bit. Controls memory usage and numerical accuracy.

    For example::

        config = ModelConfig(input_dim=784, hidden_dims=[256, 128], precision="f32")

    Spec: REQ-CORE-003
    """

    input_dim: int
    hidden_dims: list[int] = field(default_factory=list)
    precision: str = "f32"  # "f32" or "f64"


@dataclass
class ModelMetadata:
    """Training metadata — tracks progress and history.

    **Researcher summary:**
        Stores training step count and loss history for checkpoint/resume.

    **Detailed explanation for engineers:**
        When training an EBM (e.g., via score matching), we track:
        - ``step``: How many gradient updates have been applied.
        - ``loss_history``: A list of loss values recorded during training,
          useful for plotting convergence curves.

        This metadata is saved as a JSON sidecar file alongside the safetensors
        parameters, making it easy to inspect without loading the full model.

    Spec: REQ-CORE-003
    """

    step: int = 0
    loss_history: list[float] = field(default_factory=list)


@dataclass
class ModelState:
    """Complete model state for serialization — parameters + config + metadata.

    **Researcher summary:**
        Bundles parameters (as a dict of named JAX arrays), model config, and
        training metadata. Serializes to safetensors + JSON sidecar for
        cross-language compatibility.

    **Detailed explanation for engineers:**
        This is the "save file" for a trained EBM. It contains everything
        needed to reconstruct the model:

        - ``parameters``: A dictionary mapping parameter names to JAX arrays.
          For an Ising model, this might be {"coupling": <784x784 array>,
          "bias": <784 array>}. For a neural-net-based model, it would
          contain weight matrices and bias vectors for each layer.
        - ``config``: The ModelConfig that defines the architecture.
        - ``metadata``: Training progress (step count, loss history).

        **Disk format:**
        A directory containing two files:
        - ``model.safetensors`` — binary parameter data (cross-language safe)
        - ``metadata.json`` — human-readable config and training metadata

    For example::

        import jax.numpy as jnp
        from pathlib import Path

        state = ModelState(
            parameters={"coupling": jnp.eye(10), "bias": jnp.zeros(10)},
            config=ModelConfig(input_dim=10),
        )
        state.save(Path("/tmp/my_model"))

        loaded = ModelState.load(Path("/tmp/my_model"))
        assert loaded.config.input_dim == 10

    Spec: REQ-CORE-003, REQ-CORE-004
    """

    parameters: dict[str, jnp.ndarray]
    config: ModelConfig
    metadata: ModelMetadata = field(default_factory=ModelMetadata)

    def save(self, path: Path) -> None:
        """Save model state to disk as safetensors + JSON sidecar.

        **Researcher summary:**
            Writes parameters to safetensors and config/metadata to JSON.

        **Detailed explanation for engineers:**
            This method writes two files into the directory at ``path``:

            1. ``model.safetensors``: The parameter tensors, converted from
               JAX arrays to NumPy arrays (safetensors requires NumPy). The
               safetensors format is a simple header + raw binary layout that
               can be read by both Python and Rust without any pickle or
               serialization risk.

            2. ``metadata.json``: A human-readable JSON file containing the
               model config (input_dim, hidden_dims, precision) and training
               metadata (step, loss_history).

        Args:
            path: Directory path where files will be written. The directory
                must already exist.

        Spec: REQ-CORE-004, SCENARIO-CORE-004
        """
        path = Path(path)
        # Convert JAX arrays -> NumPy arrays (safetensors requires numpy)
        np_params = {k: np.asarray(v) for k, v in self.parameters.items()}
        # Write binary parameter data in the cross-language safetensors format
        save_file(np_params, str(path / "model.safetensors"))

        # Write config + metadata as a human-readable JSON sidecar
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

        **Researcher summary:**
            Reads safetensors parameters and JSON metadata, returns ModelState.

        **Detailed explanation for engineers:**
            The inverse of ``save()``. Reads the safetensors file (which gives
            us NumPy arrays), converts them to JAX arrays (for GPU
            acceleration), and reconstructs the ModelConfig and ModelMetadata
            from the JSON sidecar.

            Because safetensors is language-agnostic, a model saved by the
            Rust implementation can be loaded here, and vice versa.

        Args:
            path: Directory path containing ``model.safetensors`` and
                ``metadata.json``.

        Returns:
            A fully reconstructed ModelState.

        Spec: REQ-CORE-004, SCENARIO-CORE-004
        """
        path = Path(path)
        # Load parameters: safetensors -> NumPy -> JAX arrays
        np_params = load_file(str(path / "model.safetensors"))
        parameters = {k: jnp.array(v) for k, v in np_params.items()}

        # Load config and metadata from JSON sidecar
        with open(path / "metadata.json") as f:
            meta = json.load(f)

        config = ModelConfig(**meta["config"])
        metadata = ModelMetadata(**meta["metadata"])

        return cls(parameters=parameters, config=config, metadata=metadata)
