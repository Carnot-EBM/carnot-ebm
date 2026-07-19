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

    4. ``CASALBackend`` — adapts the CASAL primal-dual equality sampler to the
       backend boundary without making ``CASALSampler`` pretend to be an Ising
       backend directly.

    5. ``FPGAIsingSampler`` — optional KV260/PYNQ-oriented backend with a
       software-model control plane and CPU fallback.

    6. ``ClutCpuBackend`` — optional CPU adapter for the cLUT logistic
       Bernoulli random-variate path. It is opt-in and does not alter defaults.

    7. ``OneAxisRustBackend`` — optional production adapter for the promoted
       one-axis corrected-cDLS Rust/PyO3 kernel. It is opt-in and preserves the
       default CPU backend.

    8. ``get_backend(name)`` — factory function that maps a string name to a
       backend instance. Reads ``CARNOT_BACKEND`` env var as default.

Spec: REQ-SAMPLE-003, REQ-SAMPLE-2250, REQ-SAMPLE-3118, REQ-SAMPLE-5723
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from carnot.samplers.clut_backend import ClutCpuBackend
from carnot.samplers.one_axis_rust_backend import OneAxisRustBackend
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

        - ``set_constraints`` and ``dual_update_step`` are optional
          primal-dual hooks. Non-CASAL backends implement them as no-ops so the
          shared protocol can carry CASAL without breaking existing Ising
          backend call sites.

    Spec: REQ-SAMPLE-003, REQ-SAMPLE-2250
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

    def set_constraints(self, constraints: Any) -> None:
        """Set equality constraints for primal-dual samplers.

        **Detailed explanation for engineers:**
            CASAL uses equality residuals as first-class sampler state, while
            Ising backends do not. The protocol carries this hook so CASAL can
            be configured through the same backend boundary. Non-CASAL
            implementations deliberately treat it as a no-op.

        Spec: REQ-SAMPLE-2250
        """
        return None

    def dual_update_step(self, dual_lr: float) -> None:
        """Set or apply the primal-dual update learning rate.

        **Detailed explanation for engineers:**
            CASAL maps this to its Lagrange-multiplier step size. Backends that
            do not maintain dual variables keep the default no-op behavior.

        Spec: REQ-SAMPLE-2250
        """
        return None


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
        return cast("jax.Array", subkey)

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
        couplings_jax = jnp.asarray(couplings, dtype=jnp.float32)
        samples = sampler.sample(self._next_key(), b, couplings_jax, beta=beta)
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
        h_schedule = int(config.get("h_schedule", 0))

        sampler = ParallelIsingSampler(
            n_warmup=n_warmup,
            n_samples=n_samples,
            steps_per_sample=steps_per_sample,
            schedule=AnnealingSchedule(beta_init=beta, beta_final=beta),
            use_checkerboard=use_checkerboard,
        )
        b = jnp.asarray(biases, dtype=jnp.float32)
        couplings_jax = jnp.asarray(couplings, dtype=jnp.float32)
        samples = sampler.sample(
            self._next_key(), b, couplings_jax, beta=beta, h_schedule=h_schedule
        )
        return np.asarray(samples)

    def set_constraints(self, constraints: Any) -> None:
        """No-op primal-dual hook for the CPU Ising backend.

        Spec: REQ-SAMPLE-2250
        """
        return None

    def dual_update_step(self, dual_lr: float) -> None:
        """No-op dual-update hook for the CPU Ising backend.

        Spec: REQ-SAMPLE-2250
        """
        return None


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

    def set_constraints(self, constraints: Any) -> None:
        """No-op primal-dual hook for the TSU stub backend.

        Spec: REQ-SAMPLE-2250
        """
        return None

    def dual_update_step(self, dual_lr: float) -> None:
        """No-op dual-update hook for the TSU stub backend.

        Spec: REQ-SAMPLE-2250
        """
        return None


@dataclass
class CASALBackend:
    """SamplerBackend adapter for CASAL primal-dual equality sampling.

    **Detailed explanation for engineers:**
        ``CASALSampler`` has a native continuous interface:
        ``sample(x_init, energy_fn)`` plus equality constraints and dual-step
        configuration. ``SamplerBackend`` is historically Ising-shaped:
        ``sample(biases, couplings, n_samples, config)``. This adapter keeps
        those surfaces separated. Backend users can pass native CASAL inputs in
        ``config`` (``x_init``, ``energy_fn``, and optional ``constraints``), or
        call the Ising-shaped methods for a simulator-only quadratic surrogate.

    Spec: REQ-SAMPLE-2250
    """

    constraints: Any | None = None
    step_size: float = 1e-2
    dual_step_size: float = 1.0
    n_steps: int = 100
    seed: int = 0
    noise_scale: float = 1.0
    projection_steps: int = 4
    projection_damping: float = 1e-8
    penalty_weight: float = 1.0
    dual_convergence_tol: float = 1e-6
    last_violation_mean: float | None = field(default=None, init=False)
    last_dual_update_norm: float | None = field(default=None, init=False)
    _last_sampler: Any | None = field(default=None, init=False, repr=False)

    @property
    def backend_name(self) -> str:
        return "casal"

    def set_constraints(self, constraints: Any) -> None:
        """Install equality residuals for the next CASAL sampler run."""
        self.constraints = constraints
        self._last_sampler = None

    def dual_update_step(self, dual_lr: float) -> None:
        """Update CASAL's Lagrange-multiplier learning rate."""
        if dual_lr <= 0:
            raise ValueError("dual_lr must be positive for CASALBackend")
        self.dual_step_size = float(dual_lr)
        if self._last_sampler is not None:
            self._last_sampler.dual_step_size = self.dual_step_size

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """Run the simulator-only CASAL quadratic surrogate and threshold output.

        Spec: REQ-SAMPLE-2250
        """
        continuous = self.sample(
            biases,
            couplings,
            n_samples,
            {
                "energy_fn": self._quadratic_energy_fn(biases, couplings),
                "n_steps": n_steps,
                "noise_scale": 1.0 / max(float(beta), 1e-6),
            },
        )
        return np.asarray(continuous >= 0.5, dtype=bool)

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Run CASAL through the backend call shape.

        Spec: REQ-SAMPLE-2250
        """
        runtime_config = dict(config)
        if "constraints" in runtime_config:
            self.set_constraints(runtime_config["constraints"])
        if "dual_lr" in runtime_config:
            self.dual_update_step(float(runtime_config["dual_lr"]))
        if "dual_step_size" in runtime_config:
            self.dual_update_step(float(runtime_config["dual_step_size"]))

        energy_fn = runtime_config.get("energy_fn")
        if energy_fn is None:
            energy_fn = self._quadratic_energy_fn(biases, couplings)

        x_init = jnp.asarray(runtime_config.get("x_init", biases), dtype=jnp.float32)
        initial_states = self._initial_states(x_init, n_samples)
        samples = []
        for index, initial_state in enumerate(initial_states):
            sampler = self._build_sampler(runtime_config, seed_offset=index)
            sample = sampler.sample(initial_state, energy_fn)
            samples.append(sample)
            self._last_sampler = sampler

        if self._last_sampler is not None:
            self.last_violation_mean = self._last_sampler.last_violation_mean
            self.last_dual_update_norm = self._last_sampler.last_dual_update_norm

        return np.asarray(jnp.stack(samples, axis=0))

    def dual_update_converged(self, tolerance: float | None = None) -> bool:
        """Expose CASAL's latest dual convergence diagnostic."""
        if self._last_sampler is None:
            return False
        return bool(self._last_sampler.dual_update_converged(tolerance))

    def _build_sampler(self, config: dict[str, Any], seed_offset: int) -> Any:
        from carnot.samplers.casal import CASALSampler

        return CASALSampler(
            constraints=(
                self.constraints if self.constraints is not None else self._unconstrained_residual
            ),
            step_size=float(config.get("step_size", self.step_size)),
            dual_step_size=float(config.get("dual_step_size", self.dual_step_size)),
            n_steps=int(config.get("n_steps", config.get("steps", self.n_steps))),
            seed=int(config.get("seed", self.seed)) + seed_offset,
            noise_scale=float(config.get("noise_scale", self.noise_scale)),
            projection_steps=int(config.get("projection_steps", self.projection_steps)),
            projection_damping=float(config.get("projection_damping", self.projection_damping)),
            penalty_weight=float(config.get("penalty_weight", self.penalty_weight)),
            dual_convergence_tol=float(
                config.get("dual_convergence_tol", self.dual_convergence_tol)
            ),
        )

    @staticmethod
    def _initial_states(x_init: jax.Array, n_samples: int) -> list[jax.Array]:
        if x_init.ndim > 1 and x_init.shape[0] == n_samples:
            return [x_init[index] for index in range(n_samples)]
        return [x_init for _ in range(n_samples)]

    @staticmethod
    def _quadratic_energy_fn(
        biases: np.ndarray, couplings: np.ndarray
    ) -> Callable[[jax.Array], jax.Array]:
        b = jnp.asarray(biases, dtype=jnp.float32)
        coupling_matrix = jnp.asarray(couplings, dtype=jnp.float32)

        def energy_fn(x: jax.Array) -> jax.Array:
            x_flat = jnp.ravel(jnp.asarray(x, dtype=jnp.float32))
            return -jnp.dot(b, x_flat) - 0.5 * (x_flat @ coupling_matrix @ x_flat)

        return energy_fn

    @staticmethod
    def _unconstrained_residual(x: jax.Array) -> jax.Array:
        return jnp.zeros((), dtype=x.dtype)


BackendFactory = Callable[[], SamplerBackend]


_BACKENDS: dict[str, BackendFactory] = {
    "casal": CASALBackend,
    "clut_cpu": ClutCpuBackend,
    "cpu": CpuBackend,
    "one_axis_rust": OneAxisRustBackend,
    "tsu": TsuBackend,
}


# Registry for CARNOT_SAMPLER env-var-selectable backends.
# Maps the short name used in CARNOT_SAMPLER to the backend class.
# DWaveNealBackend uses a different call interface than SamplerBackend (J/h
# convention) but is included here so experiments can instantiate it by name.
# WHY a separate registry from _BACKENDS: _BACKENDS drives get_backend() which
# requires full SamplerBackend protocol; backend_registry drives
# get_sampler_backend() which is a looser factory for experiment scripts.
def _build_backend_registry() -> dict[str, type]:
    """Build the sampler backend registry, importing lazily to avoid hard deps."""
    from carnot.samplers.dwave_backend import DWaveNealBackend  # noqa: F401
    from carnot.samplers.parallel_ising import ParallelIsingSampler  # noqa: F401 (used as value)
    from carnot.samplers.tsu_sampler import TSUSampler

    return {
        "casal": CASALBackend,
        "clut_cpu": ClutCpuBackend,
        "cpu": CpuBackend,
        "dwave": DWaveNealBackend,
        "one_axis_rust": OneAxisRustBackend,
        "thrml_tsu": TSUSampler,
    }


# Public registry — populated on first access via get_sampler_backend().
# We use a module-level dict that is filled lazily so that importing this
# module at top level does not trigger heavy imports (dwave, jax, etc.) before
# the user has set JAX_PLATFORMS or other env vars.
backend_registry: dict[str, type] = {}


def _ensure_registry() -> None:
    """Populate backend_registry if it is empty."""
    if not backend_registry:
        backend_registry.update(_build_backend_registry())


def get_sampler_backend(name: str | None = None) -> object:
    """Return a sampler backend instance by name, respecting CARNOT_SAMPLER env var.

    **Detailed explanation for engineers:**
        This is the preferred entry point for experiments and production code
        that want to select a sampling backend at runtime without hard-coding
        a class name.  It reads ``CARNOT_SAMPLER`` from the environment when
        ``name`` is not supplied, defaulting to ``"cpu"`` if the variable is
        absent.

        Supported backends:
        - ``"cpu"``: ``CpuBackend`` (wraps ``ParallelIsingSampler``, JAX-based)
        - ``"clut_cpu"``: ``ClutCpuBackend`` (CPU-only cLUT random-variate path)
        - ``"casal"``: ``CASALBackend`` (wraps ``CASALSampler`` for continuous
          primal-dual simulation)
        - ``"dwave"``: ``DWaveNealBackend`` (D-Wave Ocean SDK or CPU fallback)

        Unlike ``get_backend()``, this function does NOT require the returned
        object to satisfy the full ``SamplerBackend`` protocol.  ``DWaveNealBackend``
        uses a J/h call convention rather than biases/couplings, so it is not
        interchangeable at the protocol level, but it is still a valid backend
        for experiments that call ``sample()`` and ``latency_ms()`` directly.

    Args:
        name: Backend name (``"cpu"`` or ``"dwave"``).  If None, reads
            ``CARNOT_SAMPLER`` env var, defaulting to ``"cpu"``.

    Returns:
        An instance of the requested backend class.

    Raises:
        ValueError: If *name* is not present in ``backend_registry``.

    Spec: REQ-SAMPLE-035
    """
    _ensure_registry()
    if name is None:
        name = os.environ.get("CARNOT_SAMPLER", "cpu")
    if name not in backend_registry:
        available = ", ".join(sorted(backend_registry))
        raise ValueError(f"Unknown CARNOT_SAMPLER backend {name!r}. Available: {available}")
    return backend_registry[name]()


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
        name: Backend name (``"cpu"``, ``"tsu"``, ``"casal"``, or ``"fpga"``).
            If None, reads ``CARNOT_BACKEND`` env var, defaulting to ``"cpu"``.

    Returns:
        An instance of the requested backend.

    Raises:
        ValueError: If the name doesn't match any registered backend.

    Spec: REQ-SAMPLE-003, REQ-SAMPLE-3118
    """
    if name is None:
        name = os.environ.get("CARNOT_BACKEND", "cpu")

    if name == "fpga":
        from carnot.samplers.fpga_backend import FpgaBackend

        fpga_backend: SamplerBackend = FpgaBackend()
        return fpga_backend

    if name in {"dwave_neal", "dwave_tabu", "dwave_qpu"}:
        from carnot.samplers.dwave_sampler import DWaveSampler

        mode = name[len("dwave_") :]  # strips "dwave_" prefix → "neal", "tabu", "qpu"
        dwave_backend: SamplerBackend = DWaveSampler(mode=mode)
        return dwave_backend

    if name == "thrml_tsu":
        from carnot.samplers.tsu_sampler import TSUSampler

        return TSUSampler()

    if name not in _BACKENDS:
        available = ", ".join(
            sorted(
                [*_BACKENDS.keys(), "fpga", "dwave_neal", "dwave_tabu", "dwave_qpu", "thrml_tsu"]
            )
        )
        raise ValueError(f"Unknown sampler backend {name!r}. Available backends: {available}")

    return _BACKENDS[name]()
