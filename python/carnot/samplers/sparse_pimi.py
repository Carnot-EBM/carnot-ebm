"""SparsePIMISampler — copy-node sparsified PIMI for iCE40 (arXiv 2503.01177).

**Research context (Exp 901 — FINAL PIMI attempt):**
    Three prior strategies failed to reach the 5x sweep-reduction target from
    arXiv 2604.17109 on the N=8 ring+chord ferromagnetic graph:

    - Exp 860: EMA alone (checkerboard) — 2x
    - Exp 876: EMA + alpha sweep (checkerboard) — 2x–4x
    - Exp 889: Fully synchronous PIMI updates — 4.33x (best_alpha=0.5)

    This experiment tests a different hypothesis: dense all-to-all coupling
    creates spurious local minima that increase convergence time INDEPENDENT
    of the update strategy.  arXiv 2503.01177 proposes "copy-node sparsification":
    each spin keeps only its top-k strongest couplings (by |J[i,j]| magnitude);
    weak/redundant couplings are zeroed out.  This converts O(N^2) dense edges
    to O(N*k) sparse edges, reducing the coupling graph's density.

**Copy-node sparsification (what it does and why it might help):**
    In a dense Ising graph, every spin is influenced by all N-1 others.  Many
    of those couplings may be weak (near-zero J), adding noise without useful
    constraint signal.  These weak couplings create local minima where small
    energy barriers trap the sampler for many cycles.

    By keeping only the k strongest couplings per spin, we:
    1. Remove weak-coupling noise sources that create shallow traps.
    2. Reduce the local-field computation from O(N) to O(k) per spin.
    3. Make the hardware implementation cheaper (fewer multipliers/adders).

    The "copy-node" name in arXiv 2503.01177 refers to a specific topology where
    long-range couplings route through relay nodes.  Here we implement a simpler
    variant: k-nearest-neighbor sparsification by coupling magnitude.

**Key finding (Exp 901):**
    For the N=8 ring+chord graph used in prior experiments, every spin already
    has degree exactly 3 (2 ring edges + 1 chord edge).  The dense J matrix has
    only 12 non-zero pairs — it IS already maximally sparse for k>=3.  Therefore
    k=3,4,5 all give the same result as the dense synchronous PIMI sampler.
    The sparse hypothesis does not help when the input graph is already sparse.
    This confirms retirement of iCE40 PIMI (see ops/exclusion_manifest.yaml).

Spec: REQ-HW-041
"""

from __future__ import annotations

import numpy as np


class SparsePIMISampler:
    """PIMI sampler with copy-node sparsification: keeps only top-k couplings per spin.

    **Detailed explanation for engineers:**
        Standard PIMI (SynchronousPIMISampler from Exp 889) uses the full dense
        coupling matrix J for each spin's local-field computation.  This class
        first sparsifies J by zeroing out all but the k strongest couplings per
        spin (measured by |J[i,j]| magnitude), then runs PIMI with the resulting
        sparse J.

        The key structural change vs dense PIMI:
          - Dense: each spin reads ALL N-1 neighbors every cycle
          - Sparse: each spin reads only its top-k neighbors

        In hardware (Verilog), this reduces the adder tree in STAGE 1 from N-1
        inputs to k inputs per spin — fewer LUTs and potentially shorter critical path.

        In Python, we store J_sparse as a dense ndarray (same shape as J) but with
        most entries zeroed.  The matmul J_sparse @ s_current is still O(N^2) here;
        a production implementation would use scipy.sparse for O(k*N) cost.

    Algorithm per cycle (same as SynchronousPIMISampler, different J):
        STEP 1: h_local[i] = sum_{j in top-k(i)} J_sparse[i,j] * s_current[j] + h[i]
        STEP 2: h_ema = alpha * h_ema + (1 - alpha) * h_local
        STEP 3: p_flip[i] = sigmoid(-2 * beta * h_ema[i] * s_current[i])
                s_new[i] sampled from p_flip[i]

    Args:
        n_spins: Number of spin variables N.
        J_dense: Dense coupling matrix, shape (N, N).  Will be sparsified to top-k.
        h: External field vector, shape (N,).
        k: Number of nearest neighbors to keep per spin (by |J[i,j]| magnitude).
           k >= n_spins means no sparsification (keep all couplings).
        alpha: EMA decay factor in [0, 1).  0 = no memory; higher = more inertia.
        beta: Inverse temperature.  Higher beta = sharper flip decisions.

    Spec: REQ-HW-041
    """

    def __init__(
        self,
        n_spins: int,
        J_dense: np.ndarray,
        h: np.ndarray,
        k: int = 3,
        alpha: float = 0.5,
        beta: float = 1.0,
    ) -> None:
        self.n_spins = n_spins
        self.h = np.asarray(h, dtype=np.float64)
        self.k = k
        self.alpha = float(alpha)
        self.beta = float(beta)

        # Sparsify the dense coupling matrix to top-k per spin.
        # J_sparse has the same shape as J_dense but most off-diagonal entries
        # are zeroed out — only the k largest |J[i,j]| per row i remain.
        self.J_sparse = self._sparsify(np.asarray(J_dense, dtype=np.float64), k)

        # EMA state: running average of per-spin local fields.
        self.h_ema = np.zeros(n_spins, dtype=np.float64)

    def _sparsify(self, J_dense: np.ndarray, k: int) -> np.ndarray:
        """Zero out all but top-k couplings per spin (by |J[i,j]| magnitude).

        **Detailed explanation:**
            For spin i, we look at all off-diagonal entries J[i, j] (j != i) and
            rank them by |J[i,j]|.  The top-k largest-magnitude couplings are kept;
            all others are zeroed out.

            Tie-breaking: numpy's argsort uses a stable sort — when multiple entries
            have the same magnitude, the lower-indexed j values are preferred.  This
            matters for the ring+chord graph where all J=1 couplings are equal.

            Symmetry: the result is NOT guaranteed to be symmetric.  If spin 0 keeps
            spin 4 as a top-k neighbor but spin 4 doesn't keep spin 0, the coupling
            is asymmetric.  For the ring+chord graph this doesn't happen (all spins
            have degree exactly k), but for general dense J it can.  Asymmetric
            couplings are unusual in physics but valid for this benchmarking purpose.

            When k >= n_spins - 1 (no diagonal), the full J_dense is returned unchanged.

        Args:
            J_dense: Symmetric coupling matrix, shape (N, N), diagonal=0.
            k: Number of strongest couplings to keep per row.

        Returns:
            J_sparse: Same shape as J_dense; most entries zeroed out.

        Spec: REQ-HW-041
        """
        N = J_dense.shape[0]
        J_sparse = np.zeros_like(J_dense)

        for i in range(N):
            # Get magnitudes of off-diagonal row i entries.
            row = J_dense[i].copy()
            row[i] = 0.0  # exclude self-coupling (diagonal)

            # Indices sorted by descending |J[i,j]| magnitude.
            # argsort returns ascending; we reverse for descending.
            sorted_idx = np.argsort(-np.abs(row))  # largest magnitude first

            # Keep only top-k off-diagonal couplings.
            keep = sorted_idx[:k]
            J_sparse[i, keep] = J_dense[i, keep]

        return J_sparse

    def reset(self) -> None:
        """Reset EMA history to zero for a fresh independent trial.

        **Why this matters:**
            Between independent trials we must clear h_ema.  Leftover EMA
            from a previous trial biases the starting conditions, making
            convergence artificially fast or slow.

        Spec: REQ-HW-041
        """
        self.h_ema = np.zeros(self.n_spins, dtype=np.float64)

    def sample_once(self, s_current: np.ndarray) -> np.ndarray:
        """One fully-parallel sparse PIMI cycle (uses internal RNG).

        Same algorithm as SynchronousPIMISampler.sample_once() but using
        J_sparse instead of J_dense for the local-field computation.

        Args:
            s_current: Current spin configuration, shape (N,), values in {-1, +1}.

        Returns:
            s_new: Updated spin configuration after one cycle.

        Spec: REQ-HW-041
        """
        s_current = np.asarray(s_current, dtype=np.float64)

        # STEP 1: local field using only the k retained couplings.
        # The zero entries in J_sparse contribute nothing to the matmul.
        h_local = self.J_sparse @ s_current + self.h

        # STEP 2: EMA update — blend new observation into running average.
        self.h_ema = self.alpha * self.h_ema + (1.0 - self.alpha) * h_local

        # STEP 3: flip decisions — all based on s_current snapshot (synchronous).
        argument = np.clip(2.0 * self.beta * self.h_ema * s_current, -500.0, 500.0)
        p_flip = 1.0 / (1.0 + np.exp(argument))
        rands = np.random.default_rng().random(self.n_spins)
        flip_mask = rands < p_flip

        s_new = s_current.copy()
        s_new[flip_mask] = -s_current[flip_mask]
        return s_new

    def sample_once_seeded(
        self, s_current: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """One sparse PIMI cycle with caller-supplied RNG (for reproducible benchmarks).

        Args:
            s_current: Spin snapshot, shape (N,), values in {-1, +1}.
            rng: Numpy random Generator for reproducible random numbers.

        Returns:
            s_new: Updated spin configuration.

        Spec: REQ-HW-041
        """
        s_current = np.asarray(s_current, dtype=np.float64)

        # STEP 1: sparse local field
        h_local = self.J_sparse @ s_current + self.h

        # STEP 2: EMA
        self.h_ema = self.alpha * self.h_ema + (1.0 - self.alpha) * h_local

        # STEP 3: parallel flip decisions from s_current snapshot
        argument = np.clip(2.0 * self.beta * self.h_ema * s_current, -500.0, 500.0)
        p_flip = 1.0 / (1.0 + np.exp(argument))
        flip_mask = rng.random(self.n_spins) < p_flip

        s_new = s_current.copy()
        s_new[flip_mask] = -s_current[flip_mask]
        return s_new

    def energy(self, s: np.ndarray) -> float:
        """Ising energy E(s) = -0.5 * s^T J_sparse s - h^T s.

        **Note:** This uses J_sparse (not J_dense), so the energy is relative
        to the sparsified coupling graph, not the original dense problem.
        For the ring+chord graph with k>=3, J_sparse == J_dense so results
        are identical to the dense energy.

        Args:
            s: Spin configuration, shape (N,), values in {-1, +1}.

        Returns:
            Scalar energy (lower = better / more constrained).

        Spec: REQ-HW-041
        """
        s = np.asarray(s, dtype=np.float64)
        return -0.5 * float(s @ self.J_sparse @ s) - float(self.h @ s)

    def run(
        self,
        n_sweeps: int,
        init_state: np.ndarray,
        seed: int = 0,
    ) -> tuple[np.ndarray, list[float]]:
        """Run the sampler for n_sweeps cycles.

        Args:
            n_sweeps: Number of PIMI cycles to run.
            init_state: Starting spin configuration, shape (N,).
            seed: RNG seed for reproducibility.

        Returns:
            Tuple of (final_state, energy_trajectory).

        Spec: REQ-HW-041
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

        For each trial: initialize a fresh random spin state, reset EMA, run
        PIMI cycles until energy < target_energy or max_sweeps is reached.
        Returns the integer mean across all trials (non-converging trials
        count as max_sweeps in the average).

        Args:
            n_trials: Number of independent trials to average.
            target_energy: Energy threshold — trial converges when E < target_energy.
            max_sweeps: Maximum cycles per trial before giving up.
            base_seed: Seed offset; trial k uses seed base_seed + k.

        Returns:
            Mean sweeps-to-converge, rounded to nearest integer.

        Spec: REQ-HW-041
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
