"""DWaveNealBackend — thin adapter exposing DWaveSampler(neal) as a simple J/h interface.

**Researcher summary:**
    The existing ``DWaveSampler`` implements the full ``SamplerBackend`` protocol
    with biases+couplings arrays.  ``DWaveNealBackend`` is a higher-level wrapper
    that additionally:

    1. Tries to import ``dwave.samplers`` (preferred) or ``neal`` (fallback) and
       exposes ``self.available`` so callers can branch on D-Wave availability
       without catching exceptions.
    2. Accepts ``J`` (coupling matrix) and ``h`` (bias vector) in the same
       convention as ``ParallelIsingSampler`` and returns ``jnp.ndarray`` samples.
    3. Provides ``latency_ms(n_spins)`` — runs 10 warm-up calls and returns the
       mean wall-clock milliseconds per call for a random n_spins problem.

**Why a separate wrapper:**
    Experiments need a trivial probe:
        backend = DWaveNealBackend()
        if backend.available:
            lat = backend.latency_ms(100)

    The full ``DWaveSampler`` is production-grade but requires ``dimod`` and
    handles three modes.  This wrapper is experiment-script-friendly and also
    falls back to ``ParallelIsingSampler`` when D-Wave is not installed.

Spec: REQ-SAMPLE-034, SCENARIO-SAMPLE-058, SCENARIO-SAMPLE-059
"""

from __future__ import annotations

import logging
import time
from typing import Any

import jax.numpy as jnp
import numpy as np

logger = logging.getLogger(__name__)


class DWaveNealBackend:
    """Experiment-friendly wrapper around D-Wave's SimulatedAnnealingSampler.

    **Detailed explanation for engineers:**
        On construction this class attempts to import ``dwave.samplers`` (the
        newer unified Ocean SDK package) and then falls back to ``neal`` (the
        legacy standalone package).  If neither is importable, ``self.available``
        is False and all sampling calls fall back to ``ParallelIsingSampler``.

        The J/h interface matches the convention used in ``ParallelIsingSampler``:
        - ``h``: bias vector, shape (n_spins,).  Positive h_i encourages spin=1.
        - ``J``: coupling matrix, shape (n_spins, n_spins).  J[i,j] > 0 means
          spins i and j prefer to align.  Symmetric, zero diagonal.

        Samples are returned as ``jnp.ndarray`` of shape (n_samples, n_spins)
        with bool dtype so they are compatible with existing JAX pipeline code.

    Attributes:
        available: True when the D-Wave Ocean SDK (neal or dwave.samplers)
            was successfully imported at construction time.
        _sampler: The underlying SimulatedAnnealingSampler instance, or None.

    Spec: REQ-SAMPLE-034
    """

    available: bool
    _sampler: Any

    def __init__(self) -> None:
        """Try to import D-Wave Ocean SDK; set available=True on success."""
        self._sampler = None
        self.available = False

        # Try dwave.samplers first (Ocean SDK >= 6.0 unified package).
        try:
            from dwave.samplers import SimulatedAnnealingSampler  # type: ignore[import-untyped]

            self._sampler = SimulatedAnnealingSampler()
            self.available = True
            logger.debug("DWaveNealBackend: using dwave.samplers.SimulatedAnnealingSampler")
            return
        except ImportError:
            pass

        # Fall back to the legacy standalone neal package.
        try:
            from neal import SimulatedAnnealingSampler  # type: ignore[import-untyped]

            self._sampler = SimulatedAnnealingSampler()
            self.available = True
            logger.debug("DWaveNealBackend: using neal.SimulatedAnnealingSampler")
            return
        except ImportError:
            pass

        logger.warning(
            "DWaveNealBackend: dwave-ocean-sdk not installed. "
            "Install with: pip install dwave-ocean-sdk. "
            "Falling back to ParallelIsingSampler for all sample() calls."
        )

    def sample(
        self,
        J: jnp.ndarray,
        h: jnp.ndarray,
        n_samples: int = 100,
    ) -> jnp.ndarray:
        """Draw samples using SimulatedAnnealingSampler or CPU fallback.

        **Detailed explanation for engineers:**
            When D-Wave is available, this method:
            1. Converts J and h to a dimod BinaryQuadraticModel (see
               ``dwave_sampler._ising_to_bqm`` for the conversion math).
            2. Submits to ``SimulatedAnnealingSampler.sample_ising`` with a
               fixed beta range (0.1 → 10.0) and 1000 sweeps per read.
            3. Collects the sample array from dimod and converts it to
               ``jnp.ndarray``.

            When D-Wave is NOT available, falls back to ``ParallelIsingSampler``
            which uses JAX-based parallel Gibbs sampling on CPU.

        Args:
            J: Coupling matrix, shape (n_spins, n_spins).  Symmetric, zero diagonal.
            h: Bias vector, shape (n_spins,).
            n_samples: Number of independent reads to return.

        Returns:
            Boolean JAX array of shape (n_samples, n_spins).

        Spec: REQ-SAMPLE-034
        """
        J_np = np.asarray(J, dtype=np.float64)
        h_np = np.asarray(h, dtype=np.float64)
        n_spins = int(h_np.shape[0])

        if not self.available:
            return self._cpu_fallback(J_np, h_np, n_samples)

        # Build dimod-compatible h dict and J dict.
        h_dict = {i: float(h_np[i]) for i in range(n_spins)}
        J_dict: dict[tuple[int, int], float] = {}
        for i in range(n_spins):
            for j in range(i + 1, n_spins):
                w = float(J_np[i, j])
                if w != 0.0:
                    J_dict[(i, j)] = w

        response = self._sampler.sample_ising(
            h_dict,
            J_dict,
            num_reads=n_samples,
            num_sweeps=1000,
            beta_range=[0.1, 10.0],
        )

        # Convert dimod SampleSet to boolean numpy array.
        rows: list[np.ndarray] = []
        for sample in response.samples():
            row = np.array([bool(sample.get(i, 0)) for i in range(n_spins)], dtype=bool)
            rows.append(row)
            if len(rows) >= n_samples:
                break

        if not rows:
            return jnp.zeros((n_samples, n_spins), dtype=bool)

        while len(rows) < n_samples:
            rows.append(rows[-1])

        return jnp.asarray(np.stack(rows[:n_samples], axis=0))

    def _cpu_fallback(
        self,
        J_np: np.ndarray,
        h_np: np.ndarray,
        n_samples: int,
    ) -> jnp.ndarray:
        """Fall back to ParallelIsingSampler when D-Wave is unavailable.

        **Detailed explanation for engineers:**
            This path is hit in CI or on machines without the Ocean SDK.
            ParallelIsingSampler runs JAX-based parallel Gibbs sampling on
            CPU — not Boltzmann-correct but useful for shape and contract testing.

        Spec: REQ-SAMPLE-034
        """
        import jax.random as jrandom

        from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler

        sampler = ParallelIsingSampler(
            n_warmup=200,
            n_samples=n_samples,
            steps_per_sample=10,
            schedule=AnnealingSchedule(beta_init=0.1, beta_final=10.0),
            use_checkerboard=True,
        )
        key = jrandom.PRNGKey(42)
        h_jax = jnp.asarray(h_np, dtype=jnp.float32)
        J_jax = jnp.asarray(J_np, dtype=jnp.float32)
        samples = sampler.sample(key, h_jax, J_jax, beta=10.0)
        return samples

    def latency_ms(self, n_spins: int) -> float:
        """Benchmark this backend: mean wall-clock ms per call for n_spins.

        **Detailed explanation for engineers:**
            Runs 10 calls with a random n_spins problem (small n_samples=10 to
            keep it fast) and returns the arithmetic mean in milliseconds.
            The first call may be slower due to JIT compilation (JAX path) or
            dimod warm-up.  All 10 calls are included in the mean to give a
            realistic per-call figure in a short loop.

        Args:
            n_spins: Number of Ising spins.  Larger values stress the BQM
                conversion and sampler inner loop.

        Returns:
            Mean call latency in milliseconds (float).

        Spec: REQ-SAMPLE-034
        """
        rng = np.random.default_rng(0)
        J_rand = rng.standard_normal((n_spins, n_spins)).astype(np.float32)
        J_rand = (J_rand + J_rand.T) / 2.0
        np.fill_diagonal(J_rand, 0.0)
        h_rand = rng.standard_normal(n_spins).astype(np.float32)

        J_jax = jnp.asarray(J_rand)
        h_jax = jnp.asarray(h_rand)

        n_calls = 10
        elapsed_total = 0.0
        for _ in range(n_calls):
            t0 = time.perf_counter()
            self.sample(J_jax, h_jax, n_samples=10)
            elapsed_total += time.perf_counter() - t0

        return (elapsed_total / n_calls) * 1000.0
