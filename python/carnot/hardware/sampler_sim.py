"""Bit-accurate Python simulators of the KV260 Ising-sampler RTL.

**Why this module exists (researcher summary):**
    exp1094 (Phase 2a Sampler Correctness Audit) measured KL(FPGA Glauber
    || CPU Gibbs) = 3.07 on a frustrated antiferromagnetic ring — roughly
    50x the Phase-2a acceptance threshold (KL < 0.05). The root cause is
    **not** an implementation bug in the v1/v2 RTL; it is a fundamental
    incompatibility between the previous *fully parallel synchronous*
    update order and the underlying physics:

      * Synchronous parallel Glauber resamples every spin on the same clock
        edge using stale (pre-cycle) neighbour values. On a frustrated
        antiferromagnetic ring of even length, this induces period-2
        oscillation (configuration ping-pongs every cycle), which violates
        detailed balance and converges to a non-Boltzmann fixed point.

    The literature fix (arXiv 2603.25910 / 2604.01564) is **sequential
    single-site updates**: each clock cycle, update EXACTLY ONE spin using
    the *current* values of all other spins. This is provably ergodic by
    the standard Metropolis-Hastings detailed-balance argument and converges
    to the Boltzmann distribution.

    This module exposes Python references for both update orders so that
    we can validate the new ising_sampler_v3_sequential.v RTL **before**
    the bitstream burns to silicon. The v3 simulator must reproduce the
    exact spin_select round-robin order and the exact neighbour-read
    semantics of the Verilog so a passing simulation predicts a passing
    bitstream.

**Public API:**
    - ``SynchronousIsingSamplerV1`` — reference for the **broken** parallel
      design (matches v1/v2 RTL semantics). Used only as a comparison
      baseline; should NOT be used for production sampling.
    - ``SynchronousIsingSamplerV3`` — reference for the **fixed**
      sequential single-site design (matches v3_sequential RTL). This is
      the one we expect to ship.
    - ``true_gibbs_distribution`` — closed-form Boltzmann distribution over
      all 2**N spin configurations, for use as ground truth in KL
      validation. Tractable only for small N (<= ~16 spins).

**What the simulators do NOT model:**
    - Q8.8 fixed-point quantisation of biases / couplings / beta. Tests
      run with float64 since the open-question here is *update order*,
      not arithmetic precision. A separate quantisation-error audit
      (Phase 2b) will compare float64 vs Q8.8 simulators bit-for-bit.
    - LFSR randomness. We use numpy's Mersenne Twister; LFSR vs MT only
      changes the per-step random sequence, not the stationary
      distribution.

Spec: REQ-HARDWARE-016, SCENARIO-HARDWARE-016, REQ-SAMPLE-012.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IsingProblem:
    """Container for an Ising model definition.

    Convention: spins take values in {-1, +1}. Energy is

        E(s) = - sum_{(i,j)} J[i,j] * s[i] * s[j]  -  sum_i h[i] * s[i]

    so that *negative* J encodes an antiferromagnet (neighbours want
    opposite signs) and positive J encodes a ferromagnet. This is the
    same convention used by the v1 RTL bias/coupling registers.

    Args:
        n_spins: Number of spins.
        J: Symmetric coupling matrix, shape (n_spins, n_spins).
        h: External field vector, shape (n_spins,). Default zeros.
        beta: Inverse temperature.
    """

    n_spins: int
    J: np.ndarray
    h: np.ndarray
    beta: float = 1.0

    def energy(self, s: np.ndarray) -> float:
        """Compute the Ising energy of a single spin configuration.

        ``s`` is a length-n_spins array of +/-1. We use ``-0.5 * s @ J @ s``
        instead of summing only over (i, j) with i < j because the J
        matrix is symmetric and the 1/2 corrects for double-counting.
        """
        s = np.asarray(s, dtype=np.float64).ravel()
        return float(-0.5 * s @ self.J @ s - self.h @ s)


def antiferromagnetic_ring(n_spins: int = 8, beta: float = 2.0) -> IsingProblem:
    """Build the canonical frustrated antiferromagnetic ring used by exp1094.

    Why this topology: it is the SIMPLEST graph on which fully-parallel
    synchronous Glauber dynamics fail. All adjacent pairs prefer opposite
    signs (J = -1), so the configuration wants to alternate +-+-+-...
    The synchronous update flips every spin simultaneously each cycle
    (because each spin sees its old neighbours and switches), producing
    the period-2 oscillation that exp1094 observed.

    Args:
        n_spins: Number of spins in the ring (default 8).
        beta: Inverse temperature (default 2.0, matching exp1094 setup).

    Returns:
        IsingProblem with antiferromagnetic ring couplings and zero field.
    """
    J = np.zeros((n_spins, n_spins), dtype=np.float64)
    for i in range(n_spins):
        j = (i + 1) % n_spins  # right neighbour, periodic boundary
        J[i, j] = -1.0
        J[j, i] = -1.0
    h = np.zeros(n_spins, dtype=np.float64)
    return IsingProblem(n_spins=n_spins, J=J, h=h, beta=beta)


def true_gibbs_distribution(problem: IsingProblem) -> np.ndarray:
    """Enumerate all 2**N Boltzmann probabilities exactly.

    Returns a length-(2**N) array indexed by the integer whose binary
    representation is the spin configuration with bit i = 1 meaning
    spin i = +1 (matching the RTL packing convention).

    Tractable only for N <= ~20; we use it for N=8 in the validation
    test, which is 256 configurations.
    """
    n = problem.n_spins
    if n > 20:
        raise ValueError(
            f"true_gibbs_distribution requires enumerating 2**N states; "
            f"refusing for N={n} > 20 (would need {2**n:.2g} configurations)."
        )
    energies = np.empty(2**n, dtype=np.float64)
    for idx in range(2**n):
        s = np.array([1.0 if (idx >> i) & 1 else -1.0 for i in range(n)])
        energies[idx] = problem.energy(s)
    log_p = -problem.beta * energies
    log_p -= log_p.max()  # numerical-stability shift; doesn't change result
    p = np.exp(log_p)
    p /= p.sum()
    return p


def _config_to_index(s: np.ndarray) -> int:
    """Pack a {-1,+1} spin vector into an integer (bit i = 1 iff s[i]=+1).

    Matches the RTL convention used by ``true_gibbs_distribution`` so that
    histograms produced by the simulators line up bin-for-bin with the
    closed-form distribution.
    """
    idx = 0
    for i, v in enumerate(s):
        if v > 0:
            idx |= 1 << i
    return idx


class SynchronousIsingSamplerV1:
    """Bit-accurate reference for the BROKEN parallel-synchronous design.

    This mirrors what the v1/v2 RTL actually does: every spin computes its
    new value from a snapshot of the *previous* sweep's spin values, and
    all spins are committed simultaneously. We keep this here as the
    failure baseline against which v3 sequential is measured — the KL
    delta between the two is the empirical proof that sequential is the
    right fix.

    Do NOT use this sampler in production — it does not converge to the
    Boltzmann distribution on frustrated graphs.
    """

    def __init__(self, problem: IsingProblem, seed: int = 0) -> None:
        self.problem = problem
        self._rng = np.random.default_rng(seed)
        # Hot start: all spins +1 (matches v1 reset).
        self.s = np.ones(problem.n_spins, dtype=np.int8)

    def step(self) -> None:
        """Run one parallel synchronous sweep — flip every spin at once."""
        # Snapshot current state so each spin's new value reads OLD neighbours.
        s_old = self.s.copy()
        h_eff = self.problem.h + self.problem.J @ s_old.astype(np.float64)
        # Glauber probability of being +1 after the update.
        p_plus = 1.0 / (1.0 + np.exp(-2.0 * self.problem.beta * h_eff))
        u = self._rng.random(self.problem.n_spins)
        new_s = np.where(u < p_plus, 1, -1).astype(np.int8)
        self.s = new_s

    def sample(self, n_steps: int, burn_in: int = 0) -> np.ndarray:
        """Run ``burn_in + n_steps`` sweeps; return the last n_steps configs."""
        for _ in range(burn_in):
            self.step()
        out = np.empty((n_steps, self.problem.n_spins), dtype=np.int8)
        for t in range(n_steps):
            self.step()
            out[t] = self.s
        return out


class SynchronousIsingSamplerV3:
    """Bit-accurate reference for the v3 sequential single-site design.

    Models the new ising_sampler_v3_sequential.v RTL one-for-one:
      * One spin updated per "clock cycle" (round-robin selector).
      * The energy delta is computed against the LIVE spin vector — the
        update sees every prior single-site change in this sweep.
      * Per-step Glauber acceptance: p(s_i = +1) = sigmoid(2 * beta * h_eff).

    This sampler IS provably ergodic (single-site Metropolis-Hastings)
    and is the one whose KL we expect to fall below the 0.05 acceptance
    gate.

    Naming note: we keep the "Synchronous" prefix because the underlying
    HARDWARE clock is still synchronous — the *update order* is what
    changes. The cycle-by-cycle behaviour is fully deterministic clocked
    logic; what is no longer parallel is the spin-selection across the
    array.
    """

    def __init__(self, problem: IsingProblem, seed: int = 0) -> None:
        self.problem = problem
        self._rng = np.random.default_rng(seed)
        self.s = np.ones(problem.n_spins, dtype=np.int8)
        self.spin_select = 0  # round-robin counter
        self.steps_taken = 0

    def step(self) -> None:
        """Update exactly one spin — the one at index ``spin_select``."""
        i = self.spin_select
        # h_eff[i] uses CURRENT s (post any earlier single-site updates).
        h_eff_i = self.problem.h[i] + self.problem.J[i] @ self.s.astype(np.float64)
        p_plus = 1.0 / (1.0 + np.exp(-2.0 * self.problem.beta * h_eff_i))
        u = self._rng.random()
        self.s[i] = 1 if u < p_plus else -1
        # Round-robin advance — matches Verilog spin_select counter.
        self.spin_select = (self.spin_select + 1) % self.problem.n_spins
        self.steps_taken += 1

    def sweep(self) -> None:
        """Run ``n_spins`` consecutive single-site steps (one full sweep)."""
        for _ in range(self.problem.n_spins):
            self.step()

    def sample(
        self,
        n_steps: int,
        burn_in_sweeps: int = 0,
        record_every: str = "sweep",
    ) -> np.ndarray:
        """Run sequential single-site updates and return recorded configs.

        Args:
            n_steps: Number of recorded configurations to return.
            burn_in_sweeps: Number of full sweeps discarded before
                recording, to escape the all-+1 hot-start bias.
            record_every: Either ``"sweep"`` (default) or ``"step"``.
                * ``"sweep"`` records one configuration per N_SPINS
                  consecutive single-site updates. This is the standard
                  MCMC practice and gives nearly-independent samples,
                  matching v1's "one config per parallel sweep" cadence
                  for a fair head-to-head comparison.
                * ``"step"`` records after every single-site update,
                  which gives ``N_SPINS``x more samples but with strong
                  autocorrelation between consecutive entries.

        Returns:
            (n_steps, n_spins) int8 array of recorded configurations.

        Why the default is ``"sweep"``: the Phase-2a KL acceptance gate
        (KL < 0.05) needs roughly i.i.d. samples to suppress sampling
        noise into the noise floor. Step-recorded samples have
        autocorrelation time ~N_SPINS, so the effective sample size is
        ~n_steps / N_SPINS — sweep recording gets there directly with
        the same n_steps and converts the threshold from "tight" to
        "comfortable". Empirically (validated in exp1109): on the N=8
        antiferromagnetic ring at beta=2.0, sweep-recorded 50k samples
        produce KL ~ 0.03; step-recorded 50k samples produce KL ~ 0.15.
        """
        for _ in range(burn_in_sweeps):
            self.sweep()
        out = np.empty((n_steps, self.problem.n_spins), dtype=np.int8)
        if record_every == "sweep":
            for t in range(n_steps):
                self.sweep()
                out[t] = self.s
        elif record_every == "step":
            for t in range(n_steps):
                self.step()
                out[t] = self.s
        else:
            raise ValueError(f"record_every must be 'sweep' or 'step', got {record_every!r}")
        return out


def configurations_to_indices(configs: np.ndarray) -> np.ndarray:
    """Pack a (T, N) array of {-1,+1} configs into a length-T integer index array.

    Indexing matches ``true_gibbs_distribution`` and ``_config_to_index``.
    """
    n = configs.shape[1]
    weights = np.array([1 << i for i in range(n)], dtype=np.int64)
    bits = (configs > 0).astype(np.int64)
    return bits @ weights


def kl_against_true_gibbs(
    sampler_configs: np.ndarray,
    problem: IsingProblem,
    n_bins: int | None = None,
) -> float:
    """KL(sampler_empirical || true_Gibbs) over the full configuration space.

    Args:
        sampler_configs: (T, N) array of {-1,+1} sample configurations.
        problem: The IsingProblem whose Boltzmann distribution we treat
            as ground truth.
        n_bins: Ignored — kept for API symmetry with
            ``KLDivergenceEstimator.estimate``. The configuration space
            already forms natural discrete bins.

    Returns:
        KL(P_empirical || Q_true) in nats. Laplace smoothing (+1) is
        applied to the empirical histogram to avoid log(0) on
        configurations the sampler never visited in finite samples.
    """
    del n_bins  # configuration-space KL has natural bins, no rebinning needed
    n = problem.n_spins
    n_states = 2**n
    indices = configurations_to_indices(sampler_configs)
    counts = np.bincount(indices, minlength=n_states)
    p_smooth = (counts + 1.0) / (counts.sum() + n_states)
    q = true_gibbs_distribution(problem)
    # Avoid q==0 in case of underflow (shouldn't happen with the shift in
    # true_gibbs_distribution but defensive against future edits).
    q_safe = np.clip(q, 1e-300, None)
    return float(np.sum(p_smooth * np.log(p_smooth / q_safe)))


class SparseInertiaIsingSamplerV4:
    """Bit-accurate reference for the v4 sparse-coupling, EMA-inertia design.

    **What v4 changes vs v3:**
        * Coupling is SPARSE: each spin reads only K neighbours instead
          of all N-1. This is what makes N=128 fit in the XCK26 LUT
          budget (~36K LUTs vs the ~290K dense estimate).
        * Per-spin EMA inertia: ``h_ema[i] <- alpha * h_ema[i] +
          (1 - alpha) * h_inst[i]`` where h_inst is the sparse local
          field. Inertia is the load-bearing trick from arXiv 2604.17109
          ("p-bit inertia enables correct parallel updates"): it
          smooths out the period-2 limit cycle that fully-parallel
          synchronous Glauber falls into on frustrated graphs.
        * Update rule is ALL-AT-ONCE PARALLEL: every spin reads the same
          snapshot of the previous step's spin vector, and all spins
          commit on the same clock edge. This is the speed advantage —
          one sweep per clock cycle, not N — and is the design v3
          sequential gave up to fix detailed balance.

    **Two update modes (matching the spec + LFSR option):**
        * ``mode="stochastic"`` (default for sampling experiments):
          ``p(s_i = +1) = sigmoid(2 * beta * h_ema[i])``. This is the
          p-bit interpretation of the LFSR-driven "simulated annealing
          mode" the v4 spec mentions. KL against true Gibbs is meaningful
          only for this mode at finite temperature.
        * ``mode="deterministic"``: ``s_i = sign(h_ema[i])`` (with the
          tie convention that h_ema >= 0 maps to +1, matching the RTL
          MSB rule). This is pure E-MVL — the constraint-satisfaction
          quench the spec intends. It does NOT sample any Boltzmann
          distribution at finite beta; it converges to a deterministic
          fixed point. KL on deterministic samples is therefore not a
          fair correctness statement, only a fixed-point characterisation.

    **The KL-vs-Gibbs hypothesis under test:**
        With alpha > 0 (genuine inertia), parallel synchronous updates
        on the EMA-smoothed field should converge to a distribution
        close to the true Boltzmann. Sweeping alpha at fixed beta
        empirically locates the "best inertia" knob that v3 sequential
        avoids needing because it updates one spin at a time.
    """

    def __init__(
        self,
        n_spins: int,
        k_neighbors: int,
        alpha_ema: float,
        beta_temperature: float,
        seed: int = 0,
        mode: str = "stochastic",
    ) -> None:
        if not 0.0 <= alpha_ema < 1.0:
            raise ValueError(f"alpha_ema must be in [0, 1), got {alpha_ema}")
        if k_neighbors > n_spins:
            raise ValueError(f"k_neighbors ({k_neighbors}) must be <= n_spins ({n_spins})")
        if mode not in {"stochastic", "deterministic"}:
            raise ValueError(f"mode must be 'stochastic' or 'deterministic', got {mode!r}")
        self.n_spins = n_spins
        self.k_neighbors = k_neighbors
        self.alpha_ema = float(alpha_ema)
        self.beta = float(beta_temperature)
        self.mode = mode
        self._rng = np.random.default_rng(seed)
        # Hot start: all spins +1 (matches v4 RTL reset behaviour).
        self.s = np.ones(n_spins, dtype=np.int8)
        # Per-spin EMA field register (matches RTL h_ema[0:N-1]).
        self.h_ema = np.zeros(n_spins, dtype=np.float64)

    @staticmethod
    def build_ring_topology(
        n_spins: int, k: int, j_value: float = -1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (nbr_idx, J_sparse) for a K-nearest-neighbour ring.

        For each spin i the K neighbours are split symmetrically:
            * the K/2 spins to the right (i+1, ..., i+K/2)
            * the K/2 spins to the left  (i-1, ..., i-K/2)
        all wrapped modulo N. J_sparse is filled with ``j_value`` on
        every (i, k) entry.

        ``j_value = -1.0`` recreates the antiferromagnetic ring used by
        exp1094 (with K=2 reproducing the immediate-neighbour structure
        of that test). Larger K thickens the ring into a banded
        topology, which is what v4 actually targets in hardware.

        Args:
            n_spins: Number of spins in the ring.
            k: Sparse fan-out (must be even, <= n_spins, with k/2 left + k/2 right).
            j_value: Coupling coefficient applied to every neighbour edge.

        Returns:
            Tuple ``(nbr_idx, J_sparse)`` of shape ``(n_spins, k)`` each.
        """
        if k % 2 != 0:
            raise ValueError(f"k must be even (k/2 left + k/2 right), got {k}")
        if k > n_spins:
            raise ValueError(f"k ({k}) must be <= n_spins ({n_spins})")
        nbr_idx = np.zeros((n_spins, k), dtype=np.int32)
        j_sparse = np.full((n_spins, k), float(j_value), dtype=np.float64)
        half = k // 2
        for i in range(n_spins):
            for kk in range(k):
                # Match the v4 RTL pattern: first half = right offsets +1..+K/2,
                # second half = left offsets -K/2..-1.
                offset = (kk + 1) if kk < half else (kk - k)
                nbr_idx[i, kk] = (i + offset) % n_spins
        return nbr_idx, j_sparse

    def sweep(
        self,
        spins: np.ndarray,
        nbr_idx: np.ndarray,
        j_sparse: np.ndarray,
    ) -> np.ndarray:
        """Run one parallel synchronous sweep of the v4 dynamics.

        The instance state ``self.h_ema`` is mutated in place (matches
        the RTL register update). The returned array is the new spin
        configuration after the parallel commit.

        Sequence of operations (per the v4 spec / Verilog ``comb_field``
        block followed by ``seq_update``):
            1. Snapshot ``spins`` so that all N field sums read the
               PRE-update neighbour values (synchronous discipline).
            2. ``h_inst[i] = sum_k J_sparse[i, k] * spins[nbr_idx[i, k]]``.
            3. ``h_ema[i] <- alpha * h_ema[i] + (1 - alpha) * h_inst[i]``.
            4. Map h_ema[i] to a new spin via the configured mode.

        Args:
            spins: ``(n_spins,)`` array of {-1,+1} values, the previous
                step's configuration.
            nbr_idx: ``(n_spins, k)`` neighbour index table.
            j_sparse: ``(n_spins, k)`` coupling coefficients.

        Returns:
            ``(n_spins,)`` int8 array of the updated configuration.
        """
        s_old = np.asarray(spins, dtype=np.float64)
        # Vectorised sparse field accumulation:
        #   neighbour_spins[i, k] = s_old[nbr_idx[i, k]]
        #   h_inst[i] = sum_k J_sparse[i, k] * neighbour_spins[i, k]
        neighbour_spins = s_old[nbr_idx]
        h_inst = np.sum(j_sparse * neighbour_spins, axis=1)
        # EMA update — same arithmetic as the RTL h_ema register.
        self.h_ema = self.alpha_ema * self.h_ema + (1.0 - self.alpha_ema) * h_inst
        if self.mode == "deterministic":
            # E-MVL majority vote: sign of EMA field. Treat h_ema >= 0 as
            # +1 (matches the RTL MSB convention, where MSB=0 -> +1).
            new_s = np.where(self.h_ema >= 0.0, 1, -1).astype(np.int8)
        else:
            # p-bit / Glauber on the EMA field. p_plus[i] = sigmoid(2*beta*h_ema[i]).
            p_plus = 1.0 / (1.0 + np.exp(-2.0 * self.beta * self.h_ema))
            u = self._rng.random(self.n_spins)
            new_s = np.where(u < p_plus, 1, -1).astype(np.int8)
        self.s = new_s
        return new_s

    def sample(
        self,
        nbr_idx: np.ndarray,
        j_sparse: np.ndarray,
        n_steps: int,
        burn_in_sweeps: int = 0,
    ) -> np.ndarray:
        """Run ``burn_in_sweeps`` warm-up + ``n_steps`` recorded sweeps.

        Recording cadence is one sample per parallel sweep (the natural
        unit for v4 — one clock cycle per sweep). The burn-in is applied
        before recording so the EMA register and spin state escape their
        all-+1 hot-start bias.

        Returns ``(n_steps, n_spins)`` int8 array of recorded configs.
        """
        for _ in range(burn_in_sweeps):
            self.sweep(self.s, nbr_idx, j_sparse)
        out = np.empty((n_steps, self.n_spins), dtype=np.int8)
        for t in range(n_steps):
            self.sweep(self.s, nbr_idx, j_sparse)
            out[t] = self.s
        return out
