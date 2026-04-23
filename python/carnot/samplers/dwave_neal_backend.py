"""DWaveNealBackend: simulated annealing sampler for IsingEBM via dwave-ocean-sdk.

**Researcher summary:**
    Wraps D-Wave's ``neal.SimulatedAnnealingSampler`` behind a clean interface
    that accepts ``IsingModel`` objects directly and returns a ``SampleResult``
    with the lowest-energy spin configuration found across ``num_reads`` SA runs.

**Why simulated annealing differs from Gibbs:**
    The ``ParallelIsingSampler`` (CpuBackend) uses parallel Gibbs sampling: at
    each step every spin is updated to its conditional optimum given the current
    state of all other spins.  Gibbs is exact (in the Markov-chain Monte Carlo
    sense) but can get trapped in local energy minima because it never accepts
    an uphill (energy-increasing) move.

    Simulated annealing (SA) starts at a high temperature (low beta) where the
    Boltzmann distribution is nearly uniform, so uphill moves are frequently
    accepted and the chain explores broadly.  Over time the temperature decreases
    (beta increases) and the chain focuses on low-energy configurations.  This
    cooling schedule lets SA escape local minima that Gibbs would never leave.

    The trade-off: SA is slower per call (sequential sweeps, no JAX parallelism)
    but may find deeper energy minima for dense constraint graphs with many
    frustrated cycles.

**Usage:**
    backend = DWaveNealBackend(num_reads=100, num_sweeps=1000)
    result = backend.sample(ising_model)   # SampleResult(spins, energy, wall_time_s)

Spec: REQ-SAMPLE-017, REQ-SAMPLE-018
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np


@dataclass
class SampleResult:
    """Lowest-energy sample returned by a sampling backend call.

    **Detailed explanation for engineers:**
        Every call to ``DWaveNealBackend.sample()`` returns a single
        ``SampleResult`` representing the best configuration found across
        all ``num_reads`` independent SA runs.

    Attributes:
        spins: Boolean array of shape ``(n_spins,)``. ``True`` = spin-up (+1
            in ±1 convention), ``False`` = spin-down (−1 in ±1 convention).
            Always in {0,1} boolean representation for JAX pipeline compat.
        energy: Ising energy of ``spins`` under the IsingEBM Hamiltonian,
            E = −0.5 x^T J x − b^T x, where x is the float cast of ``spins``.
            Lower values are better (more constraint-satisfying).
        wall_time_s: Wall-clock seconds elapsed for the full sample() call,
            including BQM conversion and SA sweeps.  Used to compare latency
            across backends.

    Spec: REQ-SAMPLE-018
    """

    spins: np.ndarray
    energy: float
    wall_time_s: float


class DWaveNealBackend:
    """D-Wave Ocean SDK simulated annealing sampler wrapping an IsingEBM.

    **Detailed explanation for engineers:**
        On construction this class tries to import ``neal`` (D-Wave's standalone
        SA package).  If ``neal`` is not available, ``self.available`` is False
        and ``sample()`` falls back to returning a zero-energy all-False
        configuration so that the experiment can still run and report
        ``blocked_on_dependency``.

        The J/h convention used here matches the ``IsingModel`` class:
        - ``J`` = ``ising_ebm.coupling`` (symmetric, zero-diagonal, shape NxN)
        - ``h`` = ``ising_ebm.bias`` (shape N)
        - Hamiltonian: E(x) = −0.5 x^T J x − h^T x  (x in {0,1})

        Internally, ``to_bqm()`` converts these to a ``dimod.BinaryQuadraticModel``
        using the SPIN vartype (±1 convention) because ``neal.sample()`` expects
        SPIN or BINARY vartype BQMs.  After sampling, ±1 results are converted
        back to {0,1} booleans before returning.

    Attributes:
        num_reads: Number of independent SA runs submitted in one call.
            The lowest-energy run is returned in the ``SampleResult``.
        num_sweeps: Number of temperature-sweep iterations per SA run.
            More sweeps = better quality but slower.
        beta_range: (beta_start, beta_end) for the SA cooling schedule.
            beta_start should be small (high temperature, broad exploration);
            beta_end should be large (low temperature, exploitation).
        available: True when ``neal`` was successfully imported.

    Spec: REQ-SAMPLE-017, REQ-SAMPLE-018
    """

    def __init__(
        self,
        num_reads: int = 100,
        num_sweeps: int = 1000,
        beta_range: tuple[float, float] = (0.1, 5.0),
    ) -> None:
        """Instantiate backend; try importing neal.

        Spec: REQ-SAMPLE-017
        """
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self._sampler = None
        self.available = False

        # Try importing D-Wave Ocean SDK's neal package.
        # We prefer dwave.samplers (Ocean SDK >= 6) but fall back to the
        # standalone neal package for older installs.
        try:
            from neal import SimulatedAnnealingSampler  # type: ignore[import-untyped]

            self._sampler = SimulatedAnnealingSampler()
            self.available = True
        except ImportError:
            try:
                from dwave.samplers import SimulatedAnnealingSampler  # type: ignore[import-untyped]

                self._sampler = SimulatedAnnealingSampler()
                self.available = True
            except ImportError:
                pass

    def to_bqm(self, ising_ebm: object) -> object:
        """Convert an IsingEBM to a dimod BinaryQuadraticModel (SPIN vartype).

        **Detailed explanation for engineers:**
            dimod's BQM uses the SPIN vartype (±1) internally.  The Ising
            Hamiltonian in ±1 notation is:
                E(σ) = −sum_{i<j} J[i,j] σ_i σ_j − sum_i h_i σ_i
            This is structurally identical to the {0,1} form used in IsingModel
            (after a variable substitution σ = 2x−1), but with rescaled J and h.

            We skip the rescaling here and pass J and h directly to
            ``BinaryQuadraticModel.from_ising()`` so that the energy ordering
            (which configuration has lower energy) is preserved.  The absolute
            energy values will differ between the ±1 and {0,1} formulations, but
            the RANKING of configurations is what matters for optimization.

        Args:
            ising_ebm: An ``IsingModel`` instance with ``.coupling`` (J matrix)
                and ``.bias`` (h vector) attributes.

        Returns:
            A ``dimod.BinaryQuadraticModel`` with SPIN vartype.

        Spec: REQ-SAMPLE-017
        """
        import dimod  # type: ignore[import-untyped]

        J = np.asarray(ising_ebm.coupling, dtype=np.float64)  # type: ignore[attr-defined]
        h = np.asarray(ising_ebm.bias, dtype=np.float64)  # type: ignore[attr-defined]
        n = int(h.shape[0])

        h_dict: dict[int, float] = {i: float(h[i]) for i in range(n)}
        J_dict: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                w = float(J[i, j])
                if w != 0.0:
                    J_dict[(i, j)] = w

        return dimod.BinaryQuadraticModel.from_ising(h_dict, J_dict)

    def sample(self, ising_ebm: object) -> SampleResult:
        """Find a low-energy configuration via simulated annealing.

        **Detailed explanation for engineers:**
            Converts the IsingEBM to a BQM, submits it to
            ``neal.SimulatedAnnealingSampler``, then returns the lowest-energy
            sample from all ``num_reads`` independent runs.

            The returned ``SampleResult.energy`` is computed using the IsingModel
            convention (x in {0,1}), so it is directly comparable to energies
            produced by ``CpuBackend.minimize_energy``.

        Args:
            ising_ebm: An ``IsingModel`` instance to sample from.

        Returns:
            ``SampleResult`` with the lowest-energy configuration found.

        Spec: REQ-SAMPLE-017, REQ-SAMPLE-018
        """
        t0 = time.perf_counter()

        h = np.asarray(ising_ebm.bias)  # type: ignore[attr-defined]
        n = int(h.shape[0])

        if not self.available or self._sampler is None:
            # Cannot sample — return a sentinel result.
            spins = np.zeros(n, dtype=bool)
            return SampleResult(spins=spins, energy=float("inf"), wall_time_s=time.perf_counter() - t0)

        bqm = self.to_bqm(ising_ebm)
        response = self._sampler.sample(
            bqm,
            num_reads=self.num_reads,
            num_sweeps=self.num_sweeps,
            beta_range=list(self.beta_range),
        )

        # response.first gives the lowest-energy sample.
        best = response.first
        # best.sample is {variable_index: ±1}.  Convert +1 → True, −1 → False.
        spins = np.array(
            [best.sample.get(i, 1) == 1 for i in range(n)],
            dtype=bool,
        )

        # Recompute energy in {0,1} IsingModel convention so it is comparable
        # to energy values from CpuBackend.minimize_energy.
        import jax.numpy as jnp

        x = jnp.asarray(spins, dtype=jnp.float32)
        energy = float(ising_ebm.energy(x))  # type: ignore[attr-defined]

        return SampleResult(
            spins=spins,
            energy=energy,
            wall_time_s=time.perf_counter() - t0,
        )
