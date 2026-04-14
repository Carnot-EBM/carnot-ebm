"""D-Wave sampler backend — local simulated/tabu annealing and cloud QPU.

**Researcher summary:**
    Bridges Carnot's SamplerBackend protocol to D-Wave's Ocean SDK (Apache 2.0).
    Supports three modes: local SimulatedAnnealingSampler (neal), local
    TabuSampler, and real D-Wave QPU via Leap cloud (requires API token). All
    three share the same conversion logic: Carnot biases/couplings →
    dimod BinaryQuadraticModel → D-Wave sampler → Carnot boolean sample array.

**Detailed explanation for engineers:**
    D-Wave quantum annealers natively solve Ising/QUBO problems. A QUBO
    (Quadratic Unconstrained Binary Optimization) problem asks:

        minimize  sum_i h_i * x_i + sum_{i<j} Q_ij * x_i * x_j

    where x_i ∈ {0, 1}. Carnot's Ising energy (dropping the constant beta
    factor from the problem encoding) is:

        E(s) = -sum_i b_i * s_i - sum_{i,j} J_ij * s_i * s_j

    with s ∈ {0,1} and symmetric zero-diagonal J. To get dimod to *minimize*
    E(s) we negate and identify coefficients:

        h_i = -b_i
        Q_ij = -2 * J_ij   (factor-of-2 because the full symmetric sum
                             counts each pair twice; dimod counts each pair once)

    **Why beta is not in the BQM:**
    The beta (inverse temperature) encodes how sharply the energy landscape
    is weighted. For the neal and tabu backends, beta controls the annealing
    schedule directly via ``beta_range`` and ``beta_schedule``. For the QPU,
    the annealing schedule is hardware-controlled; we expose ``annealing_time``
    (in microseconds) as a QPU-specific knob. Scaling the BQM coefficients by
    beta would shift the effective temperature for neal, but that is handled
    by the ``beta_range`` parameter instead to avoid confusion.

    **QPU embedding:**
    D-Wave QPUs have a fixed sparse topology (Pegasus or Chimera). EmbeddingComposite
    automatically finds a minor embedding that maps the logical problem graph
    onto the physical qubit graph. For dense problems (many non-zero J entries),
    embedding may fail or chain breaks may be high. Use ``health_check()`` to
    query the QPU's native problem size limits before submitting.

    **Chain breaks (QPU only):**
    A chain break occurs when qubits representing the same logical variable
    disagree after annealing. EmbeddingComposite resolves breaks by majority
    vote (default). The fraction of broken chains is stored in
    ``last_chain_break_fraction`` after each QPU solve for diagnostic use.

    **Ocean SDK (Apache 2.0):**
    - dwave-neal: SimulatedAnnealingSampler (local, no hardware needed)
    - dwave-samplers: TabuSampler (local, no hardware needed)
    - dwave-system: DWaveSampler + EmbeddingComposite (requires Leap API token)

    Install: ``pip install carnot[dwave]``

Spec: REQ-SAMPLE-003, REQ-SAMPLE-007
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _build_neal_sampler() -> Any:
    """Instantiate the local SimulatedAnnealingSampler from dwave-neal.

    This sampler runs entirely in Python/NumPy — no hardware, no cloud token.
    It is the default and the safe fallback for CI environments.
    """
    from neal import SimulatedAnnealingSampler  # type: ignore[import-untyped]

    return SimulatedAnnealingSampler()


def _build_tabu_sampler() -> Any:
    """Instantiate the local TabuSampler from dwave-samplers.

    Tabu search is a metaheuristic that tracks recently visited configurations
    (the 'tabu list') to escape local minima without accepting worsening moves.
    Faster than neal for dense problems but may miss global optima.
    """
    from tabu import TabuSampler  # type: ignore[import-untyped]

    return TabuSampler()


def _build_qpu_sampler(token: str | None) -> Any:
    """Instantiate the D-Wave QPU sampler wrapped in EmbeddingComposite.

    EmbeddingComposite transparently handles minor-embedding: it maps the
    logical problem graph onto the QPU's sparse physical topology (Pegasus
    or Chimera) before submitting and unembeds the results afterward.

    Args:
        token: D-Wave Leap API token. If None, the Ocean SDK reads the token
            from the DWAVE_API_TOKEN environment variable or ~/.config/dwave/dwave.conf.
    """
    from dwave.system import DWaveSampler as _DWaveSampler  # type: ignore[import-untyped]
    from dwave.system import EmbeddingComposite  # type: ignore[import-untyped]

    raw = _DWaveSampler(token=token)
    return EmbeddingComposite(raw)


def _ising_to_bqm(biases: np.ndarray, couplings: np.ndarray) -> Any:
    """Convert Carnot Ising parameters to a dimod BinaryQuadraticModel.

    **Detailed explanation for engineers:**
        Carnot uses {0,1} spins (BINARY vartype in dimod). The energy is:

            E(s) = -sum_i b_i * s_i  -  sum_{i,j} J_ij * s_i * s_j

        where J is symmetric and zero-diagonal. Expanding the double sum:

            sum_{i,j} J_ij * s_i * s_j = sum_i J_ii * s_i^2
                                        + 2 * sum_{i<j} J_ij * s_i * s_j
                                       = 2 * sum_{i<j} J_ij * s_i * s_j
                                         (because diagonal is zero)

        dimod's BQM minimizes: sum_i h_i * x_i + sum_{i<j} Q_ij * x_i * x_j

        Matching coefficients:
            h_i   = -b_i           (negate to convert maximization to minimization)
            Q_ij  = -2 * J_ij      (for each unique pair i < j)

    Args:
        biases: Bias vector, shape (n_spins,). Positive value encourages spin=1.
        couplings: Symmetric coupling matrix, shape (n_spins, n_spins),
            zero diagonal. J[i,j] > 0 means spins i and j prefer to align.

    Returns:
        dimod.BinaryQuadraticModel with BINARY vartype.
    """
    import dimod  # type: ignore[import-untyped]

    n = int(biases.shape[0])
    b = np.asarray(biases, dtype=np.float64)
    J = np.asarray(couplings, dtype=np.float64)

    linear = {i: -float(b[i]) for i in range(n)}

    quadratic: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            w = float(J[i, j])
            if w != 0.0:
                quadratic[(i, j)] = -2.0 * w

    return dimod.BinaryQuadraticModel(linear, quadratic, vartype=dimod.BINARY)


def _sample_set_to_array(sample_set: Any, n_spins: int, n_samples: int) -> np.ndarray:
    """Convert a dimod SampleSet to a boolean NumPy array of shape (n_samples, n_spins).

    **Detailed explanation for engineers:**
        dimod SampleSets can return fewer than ``n_samples`` distinct
        configurations (some samplers aggregate repeated reads via
        ``num_occurrences``). This function expands occurrences first,
        then pads with the last sample or trims to match exactly
        ``n_samples`` rows.

        The variable ordering in dimod samples is by variable label (integers
        0..n_spins-1 here). We read them in that order to build the row vector.

    Args:
        sample_set: A dimod.SampleSet returned by any D-Wave sampler.
        n_spins: Number of spins (expected columns).
        n_samples: Desired number of rows in the output array.

    Returns:
        Boolean array of shape (n_samples, n_spins).
    """
    rows: list[np.ndarray] = []

    for sample, _energy, num_occurrences in sample_set.data(
        ["sample", "energy", "num_occurrences"]
    ):
        row = np.array([bool(sample[i]) for i in range(n_spins)], dtype=bool)
        count = int(num_occurrences)
        for _ in range(count):
            rows.append(row)
            if len(rows) >= n_samples:
                break
        if len(rows) >= n_samples:
            break

    if not rows:
        # Sampler returned nothing — return all-zero samples.
        return np.zeros((n_samples, n_spins), dtype=bool)

    # Pad with the last row if we got fewer samples than requested.
    while len(rows) < n_samples:
        rows.append(rows[-1])

    return np.stack(rows[:n_samples], axis=0)


@dataclass
class DWaveSampler:
    """D-Wave sampler backend implementing the Carnot SamplerBackend protocol.

    **Researcher summary:**
        Pluggable backend that routes Carnot Ising problems to D-Wave's Ocean
        SDK. The three modes cover the full range from CI-safe local simulation
        (neal) to real quantum hardware (qpu) with no API changes.

    **Detailed explanation for engineers:**
        This class is a thin adapter. All problem encoding happens in
        ``_ising_to_bqm``. All result decoding happens in ``_sample_set_to_array``.
        The mode-specific samplers are built lazily in ``__post_init__`` so
        import errors (e.g. Ocean SDK not installed) surface at instantiation
        time, not at module import time. This keeps the module importable even
        when the optional dwave extra is not installed.

    Attributes:
        mode: Which D-Wave backend to use:
            - ``"neal"``: local SimulatedAnnealingSampler (default, CI-safe)
            - ``"tabu"``: local TabuSampler (faster heuristic, no hardware)
            - ``"qpu"``: real D-Wave QPU via EmbeddingComposite (requires
              Leap API token)
        leap_token: D-Wave Leap API token for QPU mode. If None, the Ocean
            SDK reads it from DWAVE_API_TOKEN env var or ~/.config/dwave/.
        annealing_time: QPU annealing time in microseconds (default 20).
            Longer annealing gives the system more time to find the ground
            state but consumes more QPU time. Ignored for neal/tabu.
        last_chain_break_fraction: After a QPU solve, the mean chain-break
            fraction across all returned samples. Zero for neal/tabu. Use
            this to diagnose embedding quality — high fractions (>0.1) suggest
            the problem is too dense for the available QPU topology.

    Spec: REQ-SAMPLE-003, REQ-SAMPLE-007
    """

    mode: str = "neal"
    leap_token: str | None = None
    annealing_time: int = 20
    last_chain_break_fraction: float = field(init=False, default=0.0)
    _sampler: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the underlying Ocean SDK sampler for the requested mode."""
        if self.mode == "neal":
            self._sampler = _build_neal_sampler()
        elif self.mode == "tabu":
            self._sampler = _build_tabu_sampler()
        elif self.mode == "qpu":
            self._sampler = _build_qpu_sampler(self.leap_token)
        else:
            raise ValueError(
                f"Unknown DWaveSampler mode {self.mode!r}. "
                "Valid modes: 'neal', 'tabu', 'qpu'."
            )

    @property
    def backend_name(self) -> str:
        """Human-readable backend name, e.g. 'dwave_neal'."""
        return f"dwave_{self.mode}"

    def _submit(
        self,
        bqm: Any,
        n_samples: int,
        n_steps: int,
        beta: float,
        fixed_temp: bool,
    ) -> Any:
        """Submit a BQM to the underlying sampler and return the SampleSet.

        **Detailed explanation for engineers:**
            This method centralizes all sampler-specific keyword arguments so
            that minimize_energy and sample share a single submission path.

            For neal:
            - ``beta_range``: [beta_start, beta_end]. For minimize_energy we
              anneal from low beta (high temp, broad exploration) to the
              requested beta. For fixed-temperature sampling we use
              [beta, beta] so every sweep runs at the same temperature.
            - ``num_sweeps``: total Gibbs sweeps per read. Maps to n_steps.
            - ``num_reads``: number of independent reads (= n_samples).

            For tabu:
            - ``num_reads``: number of independent tabu restarts.
            - ``tenure``: tabu list length (proportional to n_steps).
            - beta and fixed_temp are not applicable (tabu is not Bayesian).

            For qpu:
            - ``num_reads``: number of annealing cycles.
            - ``annealing_time``: microseconds per cycle.
            - After sampling, chain_break_fraction is extracted from the
              SampleSet record array.

        Args:
            bqm: dimod.BinaryQuadraticModel to submit.
            n_samples: Number of independent samples / reads.
            n_steps: Sweep count (neal) or tabu tenure hint.
            beta: Target inverse temperature (neal fixed-temp or end-point).
            fixed_temp: If True, run at exactly beta (no annealing schedule).

        Returns:
            dimod.SampleSet from the underlying sampler.
        """
        if self.mode == "neal":
            beta_start = float(beta) if fixed_temp else 0.1
            beta_end = float(beta)
            sample_set = self._sampler.sample(
                bqm,
                num_reads=n_samples,
                num_sweeps=max(1, n_steps),
                beta_range=[beta_start, beta_end],
            )
        elif self.mode == "tabu":
            # Tabu search is deterministic per restart; n_steps maps to tenure.
            tenure = max(1, n_steps // 10)
            sample_set = self._sampler.sample(
                bqm,
                num_reads=n_samples,
                tenure=tenure,
            )
        else:
            # QPU mode.
            sample_set = self._sampler.sample(
                bqm,
                num_reads=n_samples,
                annealing_time=self.annealing_time,
            )
            # Extract mean chain-break fraction for diagnostics.
            cbf = getattr(sample_set.record, "chain_break_fraction", None)
            if cbf is not None:
                self.last_chain_break_fraction = float(np.mean(cbf))

        return sample_set

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
            Converts the Carnot Ising problem to a dimod BQM, submits to
            the underlying Ocean sampler with an annealing schedule that
            runs from high temperature (low beta) to the requested beta, and
            returns boolean spin samples sorted by energy (lowest first, as
            dimod guarantees this ordering in the returned SampleSet).

        Args:
            biases: Bias vector, shape (n_spins,).
            couplings: Symmetric coupling matrix, shape (n_spins, n_spins).
            n_samples: Number of independent reads.
            n_steps: Sweep count per read (neal) or tabu tenure hint.
            beta: Target inverse temperature (end-point for annealing).

        Returns:
            Boolean array of shape (n_samples, n_spins).

        Spec: REQ-SAMPLE-003
        """
        b = np.asarray(biases, dtype=np.float64)
        n_spins = int(b.shape[0])
        bqm = _ising_to_bqm(b, np.asarray(couplings, dtype=np.float64))

        t0 = time.perf_counter()
        sample_set = self._submit(bqm, n_samples, n_steps, beta, fixed_temp=False)
        elapsed = time.perf_counter() - t0

        logger.debug(
            "DWaveSampler(%s).minimize_energy: n_spins=%d n_samples=%d elapsed=%.3fs",
            self.mode,
            n_spins,
            n_samples,
            elapsed,
        )

        return _sample_set_to_array(sample_set, n_spins, n_samples)

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Draw samples at fixed temperature (no annealing schedule).

        **Detailed explanation for engineers:**
            For the neal backend, this sets beta_range=[beta, beta] so every
            sweep runs at the same temperature rather than cooling. This is
            the correct way to sample from the Boltzmann distribution at a
            specific temperature rather than finding the ground state.

            The ``config`` dict reads:
            - ``"beta"`` (float, default 10.0): inverse temperature.
            - ``"n_warmup"`` (int, default 1000): sweep count per read.

        Args:
            biases: Bias vector, shape (n_spins,).
            couplings: Symmetric coupling matrix, shape (n_spins, n_spins).
            n_samples: Number of independent reads.
            config: Backend-specific configuration. Reads ``"beta"`` and
                ``"n_warmup"``.

        Returns:
            Boolean array of shape (n_samples, n_spins).

        Spec: REQ-SAMPLE-003
        """
        beta = float(config.get("beta", 10.0))
        n_steps = int(config.get("n_warmup", 1000))

        b = np.asarray(biases, dtype=np.float64)
        n_spins = int(b.shape[0])
        bqm = _ising_to_bqm(b, np.asarray(couplings, dtype=np.float64))

        t0 = time.perf_counter()
        sample_set = self._submit(bqm, n_samples, n_steps, beta, fixed_temp=True)
        elapsed = time.perf_counter() - t0

        logger.debug(
            "DWaveSampler(%s).sample: n_spins=%d n_samples=%d beta=%.2f elapsed=%.3fs",
            self.mode,
            n_spins,
            n_samples,
            beta,
            elapsed,
        )

        return _sample_set_to_array(sample_set, n_spins, n_samples)

    def health_check(self) -> dict[str, Any]:
        """Report backend type, connectivity, and problem size limits.

        **Detailed explanation for engineers:**
            For local backends (neal, tabu) this returns static information
            about what they support. For QPU, it queries the D-Wave system
            to get the number of active qubits and couplers, which determines
            the largest problem that can be embedded. Dense problems may need
            fewer logical variables than the qubit count suggests because
            embedding chains consume multiple physical qubits per logical variable.

        Returns:
            Dict with keys:
            - ``"backend"`` (str): backend name.
            - ``"mode"`` (str): "neal", "tabu", or "qpu".
            - ``"online"`` (bool): True if the backend is reachable.
            - ``"max_variables"`` (int | None): upper bound on problem size.
            - ``"max_couplers"`` (int | None): upper bound on edges (QPU only).
            - ``"qpu_name"`` (str | None): QPU model name (QPU only).
            - ``"chain_break_fraction_last"`` (float): last observed chain break
              fraction (QPU only, 0.0 otherwise).

        Spec: REQ-SAMPLE-007
        """
        if self.mode == "neal":
            return {
                "backend": self.backend_name,
                "mode": self.mode,
                "online": True,
                "max_variables": None,  # neal is unbounded (limited by RAM)
                "max_couplers": None,
                "qpu_name": None,
                "chain_break_fraction_last": 0.0,
            }

        if self.mode == "tabu":
            return {
                "backend": self.backend_name,
                "mode": self.mode,
                "online": True,
                "max_variables": None,  # tabu is unbounded (limited by RAM)
                "max_couplers": None,
                "qpu_name": None,
                "chain_break_fraction_last": 0.0,
            }

        # QPU mode: query the hardware.
        try:
            # EmbeddingComposite wraps the raw DWaveSampler; reach through to it.
            raw = self._sampler.child
            properties = raw.properties
            n_qubits = len(properties.get("qubits", []))
            n_couplers = len(properties.get("couplers", []))
            qpu_name = properties.get("chip_id", "unknown")
            return {
                "backend": self.backend_name,
                "mode": self.mode,
                "online": True,
                "max_variables": n_qubits,
                "max_couplers": n_couplers,
                "qpu_name": qpu_name,
                "chain_break_fraction_last": self.last_chain_break_fraction,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("DWaveSampler QPU health_check failed: %s", exc)
            return {
                "backend": self.backend_name,
                "mode": self.mode,
                "online": False,
                "max_variables": None,
                "max_couplers": None,
                "qpu_name": None,
                "chain_break_fraction_last": self.last_chain_break_fraction,
            }


def benchmark_dwave_vs_cpu(
    biases: np.ndarray,
    couplings: np.ndarray,
    n_samples: int = 50,
    n_steps: int = 1000,
    beta: float = 10.0,
) -> dict[str, Any]:
    """Benchmark DWaveSampler(neal) vs CpuBackend on the same Ising problem.

    **Detailed explanation for engineers:**
        Runs both samplers on identical inputs and reports wall-clock time and
        sample shapes. Does NOT compare solution quality (different stochastic
        methods will produce different energies). Useful for measuring the
        overhead of the BQM conversion and dimod call path vs the JAX path.

    Args:
        biases: Bias vector, shape (n_spins,).
        couplings: Symmetric coupling matrix, shape (n_spins, n_spins).
        n_samples: Number of independent samples for each backend.
        n_steps: Sweep / annealing steps per sample.
        beta: Final inverse temperature.

    Returns:
        Dict with ``"dwave_neal_seconds"``, ``"cpu_seconds"``, ``"n_spins"``,
        and ``"sample_shape"``.

    Spec: REQ-SAMPLE-007
    """
    from carnot.samplers.backend import CpuBackend

    dwave = DWaveSampler(mode="neal")
    cpu = CpuBackend()

    t0 = time.perf_counter()
    dwave_samples = dwave.minimize_energy(biases, couplings, n_samples, n_steps, beta)
    dwave_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    cpu_samples = cpu.minimize_energy(biases, couplings, n_samples, n_steps, beta)
    cpu_elapsed = time.perf_counter() - t0

    return {
        "dwave_neal_seconds": dwave_elapsed,
        "cpu_seconds": cpu_elapsed,
        "n_spins": int(np.asarray(biases).shape[0]),
        "sample_shape": list(dwave_samples.shape),
        "cpu_sample_shape": list(cpu_samples.shape),
    }
