"""Carnot: Open-source Energy Based Model framework (Rust + JAX).

Carnot provides three model tiers for energy-based modeling at different scales:

- **Boltzmann** (large): Research-scale deep EBMs with full JAX autodiff support.
  Suitable for academic research and large-scale experiments.

- **Gibbs** (medium): Applied ML and domain adaptation. Balanced between
  expressiveness and efficiency for production workloads.

- **Ising** (small): Edge deployment, teaching, and rapid prototyping.
  Minimal footprint with binary spin variables.

Basic usage::

    from carnot import IsingModel, IsingConfig, LangevinSampler

    config = IsingConfig(grid_size=16)
    model = IsingModel(config)
    sampler = LangevinSampler(step_size=0.01, n_steps=100)

All models share common core abstractions (EnergyFunction, ModelState, ModelConfig)
and can be trained with the same set of loss functions (NCE, DSM, SNL).
"""

from importlib import import_module
from pathlib import Path as Path
import sys
from typing import Any

from ._version import __version__

# Raw checkout imports use the same package identity without changing sys.path.
sys.modules.setdefault("carnot", sys.modules[__name__])
_EXPORTS = {
    **dict.fromkeys(["AutoGradMixin", "EnergyFunction", "ModelConfig", "ModelState"], "core"),
    **dict.fromkeys(
        [
            "BoltzmannConfig",
            "BoltzmannModel",
            "EBTConfig",
            "EBTransformer",
            "GibbsConfig",
            "GibbsModel",
            "IsingConfig",
            "IsingModel",
        ],
        "models",
    ),
    **dict.fromkeys(["HMCSampler", "LangevinSampler"], "samplers"),
    **dict.fromkeys(
        [
            "ReplayBuffer",
            "dsm_loss",
            "dsm_loss_stochastic",
            "nce_loss",
            "nce_loss_stochastic",
            "nce_loss_with_replay",
            "optimization_training_loss",
            "snl_loss",
            "snl_loss_stochastic",
        ],
        "training",
    ),
}


def _optional_jax() -> bool:
    """Check optional JAX only when its public API is requested.

    The resident worker needs NumPy arithmetic but does not need training.
    Delaying this check keeps that explicit entry free of the training graph.
    """
    if "JAX_AVAILABLE" not in globals():
        try:
            jax_module = import_module("jax")
            numpy_module = import_module("jax.numpy")
        except ImportError:
            globals().update(jax=None, JAX_AVAILABLE=False)
        else:
            globals().update(jax=jax_module, jnp=numpy_module, JAX_AVAILABLE=True)
    return bool(globals()["JAX_AVAILABLE"])


def __getattr__(name: str) -> Any:
    """Resolve original exports on demand so inference does not start training imports."""
    if name in {"JAX_AVAILABLE", "jax", "jnp", "__all__", "RUST_AVAILABLE"}:
        available = _optional_jax()
        if name == "__all__":
            value = ["__version__", "RUST_AVAILABLE", "JAX_AVAILABLE"]
            if available:
                value.extend(_EXPORTS)
        elif name == "RUST_AVAILABLE":
            value = import_module("._rust_compat", "carnot").RUST_AVAILABLE if available else False
        elif name in globals():
            return globals()[name]
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        globals()[name] = value
        return value
    if name in _EXPORTS and _optional_jax():
        value = getattr(import_module("." + _EXPORTS[name], "carnot"), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
