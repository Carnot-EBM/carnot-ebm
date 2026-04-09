"""Sampler backend abstraction layer.

**Researcher summary:**
    Defines a ``SamplerBackend`` protocol so experiments can swap between
    CPU-based parallel Gibbs sampling (via ``ParallelIsingSampler``) and
    Extropic's Thermodynamic Sampling Unit (TSU) hardware — or any future
    backend — by changing a single config string or environment variable.

**Detailed explanation for engineers:**
    Today, Carnot has exactly one sampling backend: the JAX-based
    ``ParallelIsingSampler`` that runs on CPU/GPU. When Extropic ships their
    TSU hardware, we'll need a second backend that speaks to the TSU driver
    over its native interface. Other backends (FPGA, analog, cloud API) may
    follow.

    This module provides:

    1. ``SamplerBackend`` — a Python Protocol (structural interface) with two
       methods (``minimize_energy`` and ``sample``) and one property
       (``backend_name``). Any object matching this shape counts as a backend,
       no inheritance required.

    2. ``CpuBackend`` — wraps ``ParallelIsingSampler`` behind the protocol.
       This is the default backend and the one used in all current experiments.

    3. ``TsuBackend`` — a stub that logs calls and returns random binary
       samples. It exists so that config files can reference "tsu" today
       without crashing, and so integration tests can verify the backend
       switching logic end-to-end. When the real TSU driver lands, this stub
       gets replaced with actual hardware calls.

    4. ``get_backend(name)`` — factory function that maps a string name to a
       backend instance. Reads ``CARNOT_BACKEND`` env var as default.

Spec: REQ-SAMPLE-003
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler

logger = logging.getLogger(__name__)


@runtime_checkable
class SamplerBackend(Protocol):
    """Protocol that every sampling backend must satisfy.

    **Detailed explanation for engineers:**
        This is a structural (duck-typed) interface. Any class that implements
        ``minimize_energy``, ``sample``, and the ``backend_name`` property with
        matching signatures is a valid ``SamplerBackend`` — no explicit
        inheritance or registration needed.

        - ``minimize_energy`` runs a full annealing + sampling pipeline and
          returns low-energy configurations. This is the high-level API most
          experiments use.

        - ``sample`` draws samples at a fixed temperature (no annealing). This
          is the lower-level API used for Boltzmann distribution estimation
          and KL gradient computation.

        - ``backend_name`` is a human-readable string like "cpu" or "tsu" used
          for logging and config validation.

    Spec: REQ-SAMPLE-003
    """

    @property
    def backend_name(self) -> str:
        """Human-readable name for this backend (e.g. "cpu", "tsu")."""
        ...

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """Run annealing to find low-energy spin configurations.

        **Detailed explanation for engineers:**
            Performs simulated annealing (or hardware-native equivalent) to
            produce spin configurations biased toward low energy. The returned
            array contains boolean spins in {0, 1}.

        Args:
            biases: Bias vector, shape ``(n_spins,)``.
            couplings: Symmetric coupling matrix, shape ``(n_spins, n_spins)``.
            n_samples: Number of independent samples to return.
            n_steps: Number of annealing / sweep steps.
            beta: Inverse temperature (final, if annealing).

        Returns:
            Boolean array of shape ``(n_samples, n_spins)``.

        Spec: REQ-SAMPLE-003
        """
        ...

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Draw samples at fixed temperature (no annealing).

        **Detailed explanation for engineers:**
            Unlike ``minimize_energy``, this method samples from the Boltzmann
            distribution at a fixed beta without an annealing schedule. The
            ``config`` dict allows backend-specific parameters (e.g.
            ``steps_per_sample``, ``use_checkerboard`` for CPU; hardware
            register settings for TSU).

        Args:
            biases: Bias vector, shape ``(n_spins,)``.
            couplings: Symmetric coupling matrix, shape ``(n_spins, n_spins)``.
            n_samples: Number of samples to draw.
            config: Backend-specific configuration dict. Must include ``"beta"``
                (float). CPU backend also reads ``"steps_per_sample"`` (int,
                default 20) and ``"use_checkerboard"`` (bool, default True).

        Returns:
            Boolean array of shape ``(n_samples, n_spins)``.

        Spec: REQ-SAMPLE-003
        """
        ...


@dataclass
class CpuBackend:
    """CPU backend wrapping ``ParallelIsingSampler``.

    **Detailed explanation for engineers:**
        This is a thin adapter that translates the ``SamplerBackend`` interface
        into calls to the existing ``ParallelIsingSampler``. All heavy lifting
        — parallel Gibbs sweeps, checkerboard decomposition, JAX JIT
        compilation — happens inside ``ParallelIsingSampler``. This class just
        manages the JAX PRNG key and parameter mapping.

    Attributes:
        seed: Random seed for JAX PRNG key generation. Each call to
            ``minimize_energy`` or ``sample`` consumes and advances the key,
            so results are reproducible given the same seed and call sequence.

    Spec: REQ-SAMPLE-003
    """

    seed: int = 42
    _key: jax.Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._key = jrandom.PRNGKey(self.seed)

    @property
    def backend_name(self) -> str:
        return "cpu"

    def _next_key(self) -> jax.Array:
        """Split and advance the internal PRNG key."""
        self._key, subkey = jrandom.split(self._key)
        return subkey

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """Run annealing via ``ParallelIsingSampler`` to find low-energy states.

        **Detailed explanation for engineers:**
            Creates a ``ParallelIsingSampler`` with the annealing warmup phase
            set to ``n_steps`` and a linear schedule from low beta to the
            requested beta. Collects ``n_samples`` after annealing completes.

        Spec: REQ-SAMPLE-003
        """
        sampler = ParallelIsingSampler(
            n_warmup=n_steps,
            n_samples=n_samples,
            steps_per_sample=20,
            schedule=AnnealingSchedule(beta_init=0.1, beta_final=beta),
            use_checkerboard=True,
        )
        b = jnp.asarray(biases, dtype=jnp.float32)
        J = jnp.asarray(couplings, dtype=jnp.float32)
        samples = sampler.sample(self._next_key(), b, J, beta=beta)
        return np.asarray(samples)

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Draw samples at fixed temperature via ``ParallelIsingSampler``.

        **Detailed explanation for engineers:**
            Sets both ``beta_init`` and ``beta_final`` to the same value so
            there is no annealing — every sweep runs at the requested
            temperature. The warmup phase still runs to let the chain mix
            before collecting samples.

        Spec: REQ-SAMPLE-003
        """
        beta = float(config.get("beta", 10.0))
        steps_per_sample = int(config.get("steps_per_sample", 20))
        use_checkerboard = bool(config.get("use_checkerboard", True))
        n_warmup = int(config.get("n_warmup", 500))

        sampler = ParallelIsingSampler(
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=steps_per_sample,
            schedule=AnnealingSchedule(beta_init=beta, beta_final=beta),
            use_checkerboard=use_checkerboard,
        )
        b = jnp.asarray(biases, dtype=jnp.float32)
        J = jnp.asarray(couplings, dtype=jnp.float32)
        samples = sampler.sample(self._next_key(), b, J, beta=beta)
        return np.asarray(samples)


@dataclass
class TsuBackend:
    """Stub backend for Extropic's Thermodynamic Sampling Unit (TSU).

    **Detailed explanation for engineers:**
        The TSU is Extropic's custom ASIC that performs native thermodynamic
        sampling — instead of simulating Boltzmann dynamics on a CPU/GPU, it
        uses physical thermal noise in analog circuits to sample directly from
        the Ising distribution. This gives orders-of-magnitude speedup and
        energy efficiency for sampling-dominated workloads.

        This stub exists so that:
        1. Config files and experiments can reference ``"tsu"`` as a backend
           today without crashing.
        2. Integration tests can verify the backend-switching logic.
        3. The call signature and return shapes are documented and tested,
           making the real TSU driver a drop-in replacement.

        Every call is logged to ``self.call_log`` (a list of dicts) so tests
        can verify that the right methods were called with the right arguments.
        The stub returns random binary arrays of the correct shape — not
        physically meaningful, but structurally correct.

    Attributes:
        call_log: List of dicts recording every method call. Each dict has
            keys ``"method"`` (str), ``"biases_shape"`` (tuple),
            ``"couplings_shape"`` (tuple), ``"n_samples"`` (int), and
            method-specific keys.
        seed: Random seed for reproducible stub output.

    Spec: REQ-SAMPLE-003
    """

    seed: int = 42
    call_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def backend_name(self) -> str:
        return "tsu"

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """Stub: logs call and returns random binary samples.

        **Detailed explanation for engineers:**
            When the real TSU driver is available, this method will program the
            TSU's bias and coupling registers, run the annealing protocol in
            hardware, and read back the resulting spin states. For now it logs
            the call parameters and returns random {0, 1} arrays.

        Spec: REQ-SAMPLE-003
        """
        logger.warning(
            "TsuBackend.minimize_energy called — this is a stub. "
            "Install the Extropic TSU driver for real hardware sampling."
        )
        self.call_log.append(
            {
                "method": "minimize_energy",
                "biases_shape": biases.shape,
                "couplings_shape": couplings.shape,
                "n_samples": n_samples,
                "n_steps": n_steps,
                "beta": beta,
            }
        )
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
        """Stub: logs call and returns random binary samples.

        **Detailed explanation for engineers:**
            When the real TSU driver is available, this method will program the
            TSU with the given Ising parameters and draw samples at fixed
            temperature. For now it logs and returns random bits.

        Spec: REQ-SAMPLE-003
        """
        logger.warning(
            "TsuBackend.sample called — this is a stub. "
            "Install the Extropic TSU driver for real hardware sampling."
        )
        self.call_log.append(
            {
                "method": "sample",
                "biases_shape": biases.shape,
                "couplings_shape": couplings.shape,
                "n_samples": n_samples,
                "config": config,
            }
        )
        rng = np.random.default_rng(self.seed)
        n_spins = biases.shape[0]
        return rng.integers(0, 2, size=(n_samples, n_spins)).astype(bool)


_BACKENDS: dict[str, type] = {
    "cpu": CpuBackend,
    "tsu": TsuBackend,
}


def get_backend(name: str | None = None) -> SamplerBackend:
    """Factory function: return a ``SamplerBackend`` by name.

    **Detailed explanation for engineers:**
        Looks up the backend class in a registry and instantiates it with
        default parameters. The name defaults to the ``CARNOT_BACKEND``
        environment variable, falling back to ``"cpu"`` if unset.

        This is the primary entry point for experiments that want to be
        backend-agnostic::

            from carnot.samplers.backend import get_backend

            backend = get_backend()  # reads CARNOT_BACKEND or defaults to cpu
            samples = backend.minimize_energy(biases, couplings, 100, 1000, 10.0)

    Args:
        name: Backend name (``"cpu"`` or ``"tsu"``). If None, reads
            ``CARNOT_BACKEND`` env var, defaulting to ``"cpu"``.

    Returns:
        An instance of the requested backend.

    Raises:
        ValueError: If the name doesn't match any registered backend.

    Spec: REQ-SAMPLE-003
    """
    if name is None:
        name = os.environ.get("CARNOT_BACKEND", "cpu")

    if name not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise ValueError(
            f"Unknown sampler backend {name!r}. Available backends: {available}"
        )

    return _BACKENDS[name]()
