"""SynchronousPIMISampler — truly parallel PIMI spin updates (arXiv 2604.17109).

**What PIMI actually requires (and why Exp 860/876 missed the target):**
    Exp 860 and Exp 876 achieved only ~2–4x sweep reduction instead of the
    paper's 15–25x because both used *checkerboard* (sequential even/odd)
    updates with EMA — NOT the PIMI algorithm as described.

    The actual PIMI algorithm requires FULLY SYNCHRONOUS updates:
      - All spins compute h_local[i] using s_CURRENT (the spin vector from
        the START of the cycle, not any partial update)
      - All spins update their h_ema using those h_local values
      - All flip decisions are made simultaneously using s_CURRENT
      - s_new is assembled and the entire vector replaces s_current at once

    In checkerboard updates, even-spin flips change the local field seen by
    odd spins in the SAME sweep.  This breaks the parallel independence that
    gives PIMI its speedup — odd spins are already informed by even-spin
    results, so the EMA has less novel information per cycle.

    True parallel updates preserve full independence: every spin's decision
    is made based on the same snapshot of the system.  The EMA accumulates
    that snapshot-history, and when the system is near a minimum, consecutive
    snapshots agree strongly — the EMA reinforces the signal and convergence
    accelerates dramatically.

Spec: REQ-HW-036
"""

from __future__ import annotations

import numpy as np


class SynchronousPIMISampler:
    """Ising sampler with fully synchronous (parallel) PIMI updates.

    **Detailed explanation for engineers:**
        Standard Gibbs sampling visits spins one at a time.  Each spin's
        flip decision sees the *already-updated* neighbors from the same
        sweep.  This is like asking each person in a committee to vote
        after hearing what the previous person said — the later voters
        are influenced by the earlier ones.

        The PIMI (Parallel Ising with Momentum Inertia) algorithm from
        arXiv 2604.17109 instead has ALL spins vote simultaneously based
        on a snapshot of the system from the PREVIOUS cycle.  This is
        the secret-ballot version: everyone writes their answer at the
        same time without seeing each other's answers.

        The momentum (EMA) layer compounds the benefit: when the system
        is near a minimum, consecutive snapshots produce consistent local
        fields.  The EMA builds up a strong signal, and the flip probability
        p_flip = sigmoid(-2 * beta * h_ema[i] * s[i]) becomes very small
        for aligned spins.  This "freezes in" good spin configurations and
        prevents the random-walk flipping that slows naive samplers.

        The 15–25x speedup in the paper comes from this combination:
          1. Parallel updates give higher-quality, independent information
             per cycle.
          2. EMA momentum amplifies consistent signals and dampens noise.
          3. Together they let the sampler lock onto low-energy states in
             far fewer cycles than sequential methods.

    Per-cycle algorithm (all steps operate on ALL N spins simultaneously):
        STEP 1 (h_local): h_local[i] = sum_j J[i,j] * s_current[j] + h[i]
                          Uses s_current from the START of this cycle.
        STEP 2 (EMA):     h_ema = alpha * h_ema + (1-alpha) * h_local
                          Updates the momentum estimate for each spin.
        STEP 3 (flip):    p_flip[i] = sigmoid(-2 * beta * h_ema[i] * s_current[i])
                          s_new[i] = -s_current[i] if rand() < p_flip[i]
                                     else s_current[i]
                          CRITICAL: uses s_current[j] for ALL j, not s_new[j].

    Args:
        n_spins: Number of spin variables N.
        J: Coupling matrix of shape (N, N).  J[i,j] = coupling strength.
           For ferromagnetic Ising, J[i,j] > 0 favors aligned spins.
        h: External field vector of shape (N,).  Usually zeros for pure
           constraint satisfaction problems.
        alpha: EMA decay factor in [0, 1).  0 = no memory (standard Gibbs);
               values approaching 1 give stronger momentum / inertia.
               Higher alpha means MORE history weight (unlike some conventions).
               Formula: h_ema = alpha * h_ema + (1-alpha) * h_local
               So alpha=0.5 means 50% old history, 50% new observation.
        beta: Inverse temperature.  Higher beta = sharper flip probabilities
              = more deterministic behavior = lower effective temperature.

    Spec: REQ-HW-036
    """

    def __init__(
        self,
        n_spins: int,
        J: np.ndarray,
        h: np.ndarray,
        alpha: float = 0.125,
        beta: float = 1.0,
    ) -> None:
        self.n_spins = n_spins
        self.J = np.asarray(J, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # EMA state: running average of per-spin local fields.
        # Initialized to zero — no prior history.
        self.h_ema = np.zeros(n_spins, dtype=np.float64)

    def reset(self) -> None:
        """Reset EMA history for a fresh independent run.

        **Why this matters:**
            Between independent trials we must clear h_ema.  If we reuse
            a sampler object across trials, the EMA carries over history
            from the previous run and biases the new one.

        Spec: REQ-HW-036
        """
        self.h_ema = np.zeros(self.n_spins, dtype=np.float64)

    def sample_once(self, s_current: np.ndarray) -> np.ndarray:
        """Execute one fully-parallel PIMI cycle.

        **CRITICAL IMPLEMENTATION NOTE:**
            This function MUST use s_current (the input argument) for ALL
            local field computations in STEP 1.  It must NOT use any
            intermediate s_new values.  This is the defining property of
            synchronous / parallel updates.

            If you modify this function, verify that:
              - STEP 1 reads only self.J, self.h, and s_current
              - STEP 3 reads only self.h_ema (updated in STEP 2) and s_current
              - s_new is assembled only after all decisions are made

        Args:
            s_current: Current spin configuration, shape (N,), values in {-1, +1}.
                       This is the SNAPSHOT used for all computations this cycle.
                       It is NOT modified in-place.

        Returns:
            s_new: New spin configuration after one parallel PIMI cycle,
                   shape (N,), values in {-1, +1}.

        Spec: REQ-HW-036
        """
        s_current = np.asarray(s_current, dtype=np.float64)

        # STEP 1: Compute local field for ALL spins using s_current snapshot.
        # h_local[i] = sum_j J[i,j] * s_current[j] + h[i]
        # Every spin sees the SAME s_current — no spin gets updated-neighbor info.
        h_local = self.J @ s_current + self.h

        # STEP 2: EMA update — blend new local field into running average.
        # h_ema[i] = alpha * h_ema[i] + (1 - alpha) * h_local[i]
        # This is done for ALL spins before ANY flip decisions.
        self.h_ema = self.alpha * self.h_ema + (1.0 - self.alpha) * h_local

        # STEP 3: Compute flip probabilities and sample new spins in parallel.
        # p_flip[i] = sigmoid(-2 * beta * h_ema[i] * s_current[i])
        # Uses h_ema from STEP 2 and s_current from STEP 0.
        # ALL decisions made with the same s_current — true parallel update.
        argument = 2.0 * self.beta * self.h_ema * s_current
        argument = np.clip(argument, -500.0, 500.0)
        p_flip = 1.0 / (1.0 + np.exp(argument))

        # Generate one random number per spin; flip if rand < p_flip.
        rands = np.random.default_rng().random(self.n_spins)
        flip_mask = rands < p_flip

        # Assemble s_new from s_current with flips applied.
        s_new = s_current.copy()
        s_new[flip_mask] = -s_current[flip_mask]

        return s_new

    def sample_once_seeded(self, s_current: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One PIMI cycle with a caller-supplied RNG (for reproducible benchmarks).

        Same algorithm as sample_once() but accepts an external RNG so the
        caller controls the random seed sequence.  Used internally by run()
        and measure_convergence() for reproducible multi-trial benchmarks.

        Args:
            s_current: Spin snapshot, shape (N,), values in {-1, +1}.
            rng: Numpy random Generator (e.g., np.random.default_rng(seed)).

        Returns:
            s_new: Updated spin configuration, shape (N,).

        Spec: REQ-HW-036
        """
        s_current = np.asarray(s_current, dtype=np.float64)

        # STEP 1: local fields — uses s_current ONLY
        h_local = self.J @ s_current + self.h

        # STEP 2: EMA
        self.h_ema = self.alpha * self.h_ema + (1.0 - self.alpha) * h_local

        # STEP 3: parallel flip decisions — uses s_current ONLY
        argument = np.clip(2.0 * self.beta * self.h_ema * s_current, -500.0, 500.0)
        p_flip = 1.0 / (1.0 + np.exp(argument))
        flip_mask = rng.random(self.n_spins) < p_flip

        s_new = s_current.copy()
        s_new[flip_mask] = -s_current[flip_mask]
        return s_new

    def energy(self, s: np.ndarray) -> float:
        """Compute Ising energy E(s) = -0.5 * s^T J s - h^T s.

        **Detailed explanation:**
            The Ising energy has two contributions:
            1. Pairwise coupling: -0.5 * sum_{i,j} J[i,j] * s[i] * s[j]
               For ferromagnetic J > 0, aligned spins (same sign) give
               negative (low) energy — the system prefers alignment.
            2. External field: -sum_i h[i] * s[i]
               Spins aligned with h also lower the energy.

            The factor 0.5 prevents double-counting (J is symmetric so
            each pair (i,j) appears in both J[i,j]*s[i]*s[j] and
            J[j,i]*s[j]*s[i]).

        Args:
            s: Spin configuration, shape (N,), values in {-1, +1}.

        Returns:
            Scalar energy (lower = more constrained / ground-state-like).

        Spec: REQ-HW-036
        """
        s = np.asarray(s, dtype=np.float64)
        return -0.5 * float(s @ self.J @ s) - float(self.h @ s)

    def run(
        self,
        n_sweeps: int,
        init_state: np.ndarray,
        seed: int = 0,
    ) -> tuple[np.ndarray, list[float]]:
        """Run the sampler for n_sweeps cycles from init_state.

        Args:
            n_sweeps: Number of parallel PIMI cycles to execute.
            init_state: Starting spin configuration, shape (N,).
            seed: RNG seed for reproducibility.

        Returns:
            Tuple of (final_state, energy_trajectory):
              - final_state: Spin configuration after n_sweeps cycles.
              - energy_trajectory: List of energy values, one per cycle.

        Spec: REQ-HW-036
        """
        rng = np.random.default_rng(seed)
        self.reset()

        s = np.asarray(init_state, dtype=np.float64).copy()
        energies: list[float] = []

        for _ in range(n_sweeps):
            s = self.sample_once_seeded(s, rng)
            energies.append(self.energy(s))

        return s, energies

    def measure_convergence(
        self,
        n_trials: int,
        target_energy: float,
        max_sweeps: int,
        base_seed: int = 0,
    ) -> int:
        """Measure mean sweeps-to-converge over n_trials independent runs.

        **Detailed explanation:**
            For each trial, initializes a fresh random spin configuration,
            resets the EMA, then runs PIMI cycles until energy < target_energy
            or max_sweeps is reached.

            Each trial uses seed = base_seed + trial_index for reproducibility
            while keeping trials independent.

            Returns the integer mean across all trials.  Trials that hit
            max_sweeps without converging count as max_sweeps in the average.

        Args:
            n_trials: Number of independent convergence trials to average.
            target_energy: Energy threshold for "converged" (lower = stricter).
            max_sweeps: Cap on sweeps per trial to bound runtime.
            base_seed: Seed offset; trial k uses seed base_seed + k.

        Returns:
            Mean sweeps-to-converge as an integer (rounded).

        Spec: REQ-HW-036
        """
        N = self.n_spins
        trial_sweeps: list[int] = []

        for trial in range(n_trials):
            rng = np.random.default_rng(base_seed + trial)
            self.reset()

            # Random ±1 initialization
            s = (2 * rng.integers(0, 2, size=N) - 1).astype(np.float64)
            converged_at = max_sweeps

            for sweep in range(1, max_sweeps + 1):
                s = self.sample_once_seeded(s, rng)
                if self.energy(s) < target_energy:
                    converged_at = sweep
                    break

            trial_sweeps.append(converged_at)

        return int(round(float(np.mean(trial_sweeps))))


def make_n8_coupling_matrix() -> np.ndarray:
    """Build the N=8 constraint graph as a dense coupling matrix for benchmarking.

    **Detailed explanation:**
        Same topology as make_n8_constraint_adjacency() in sparse_inertia_ising.py:
        ring edges (0-1, 1-2, ..., 7-0) plus chord edges (0-4, 1-5, 2-6, 3-7).
        K=12 non-zero pairs, all J_ij = +1.0 (ferromagnetic).

        We return the DENSE symmetric matrix here because SynchronousPIMISampler
        uses matrix-vector multiply (self.J @ s) for simplicity.  The sparse
        version is in SparseInertiaIsingSampler; this class trades storage for
        clean O(N²) numpy multiply.

    Returns:
        J: np.ndarray of shape (8, 8), symmetric, values in {0.0, 1.0}.

    Spec: REQ-HW-036
    """
    N = 8
    J = np.zeros((N, N), dtype=np.float64)

    # Ring edges
    for i in range(N):
        j = (i + 1) % N
        J[i, j] = 1.0
        J[j, i] = 1.0

    # Chord edges (opposite pairs)
    for i in range(N // 2):
        j = i + N // 2
        J[i, j] = 1.0
        J[j, i] = 1.0

    return J


def pimi_alpha_sweep(
    alphas: list[float],
    n_trials: int = 100,
    energy_threshold: float = -3.0,
    max_sweeps: int = 400,
) -> dict[str, float]:
    """Sweep alpha values and measure mean convergence for SynchronousPIMISampler.

    Args:
        alphas: Alpha values to test (EMA decay factor).
        n_trials: Independent trials per alpha.
        energy_threshold: Convergence criterion.
        max_sweeps: Maximum sweeps per trial.

    Returns:
        Dict mapping str(alpha) → mean sweeps_to_converge.

    Spec: REQ-HW-036
    """
    J = make_n8_coupling_matrix()
    h = np.zeros(8, dtype=np.float64)
    results: dict[str, float] = {}

    for alpha in alphas:
        sampler = SynchronousPIMISampler(
            n_spins=8, J=J, h=h, alpha=alpha, beta=1.0
        )
        mean_sweeps = sampler.measure_convergence(
            n_trials=n_trials,
            target_energy=energy_threshold,
            max_sweeps=max_sweeps,
        )
        results[str(alpha)] = float(mean_sweeps)

    return results
