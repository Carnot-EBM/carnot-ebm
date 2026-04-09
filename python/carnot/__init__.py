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

from carnot._version import __version__

# Core abstractions
from carnot.core import AutoGradMixin, EnergyFunction, ModelConfig, ModelState

# Models — three tiers
from carnot.models import (
    BoltzmannConfig,
    BoltzmannModel,
    EBTConfig,
    EBTransformer,
    GibbsConfig,
    GibbsModel,
    IsingConfig,
    IsingModel,
)

# Samplers
from carnot.samplers import HMCSampler, LangevinSampler

# Training losses
from carnot.training import (
    ReplayBuffer,
    dsm_loss,
    dsm_loss_stochastic,
    nce_loss,
    nce_loss_stochastic,
    nce_loss_with_replay,
    optimization_training_loss,
    snl_loss,
    snl_loss_stochastic,
)

# Rust binding availability (optional — no Rust toolchain needed for pure-Python)
from carnot._rust_compat import RUST_AVAILABLE

__all__ = [
    # Version
    "__version__",
    # Rust bindings
    "RUST_AVAILABLE",
    # Core
    "AutoGradMixin",
    "EnergyFunction",
    "ModelConfig",
    "ModelState",
    # Models
    "BoltzmannConfig",
    "BoltzmannModel",
    "EBTConfig",
    "EBTransformer",
    "GibbsConfig",
    "GibbsModel",
    "IsingConfig",
    "IsingModel",
    # Samplers
    "HMCSampler",
    "LangevinSampler",
    # Training
    "ReplayBuffer",
    "dsm_loss",
    "dsm_loss_stochastic",
    "nce_loss",
    "nce_loss_stochastic",
    "nce_loss_with_replay",
    "optimization_training_loss",
    "snl_loss",
    "snl_loss_stochastic",
]
