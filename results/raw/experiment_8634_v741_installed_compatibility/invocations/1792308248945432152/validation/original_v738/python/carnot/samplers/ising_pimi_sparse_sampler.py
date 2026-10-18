"""PIMMISparseAdjacency — top-sparsity% coupling PIMI sampler (Exp 914 FINAL attempt).

**Research context (why this module exists):**
    Four prior strategies all failed to reach the 5x sweep-reduction target
    from arXiv 2604.17109 on Ising problems at hardware-feasible scales:

    - Exp 860: EMA checkerboard updates — 2x (wrong algorithm, not truly parallel)
    - Exp 876: EMA alpha sweep (checkerboard) — 2–4x (same root cause)
    - Exp 889: Fully synchronous PIMI updates — 4.33x (correct algorithm, N=8)
    - Exp 901: Copy-node sparsification at N=8 — 4.33x (N=8 graph already sparse)

    Exp 914 tests the FINAL hypothesis: at N=64 (KV260 FPGA capacity, see ops/status.md)
    with a genuinely dense random J matrix, keeping only the top-20% strongest couplings
    (by |J[i,j]| magnitude) should:
      1. Reduce per-sweep FLOP count by 5x (0.2 * N^2 non-zeros vs full N^2)
      2. Possibly reduce convergence sweeps by removing weak/noisy couplings

    If the TOTAL sweep count does not drop by 5x, this retirement trigger fires:
      retire_if_same_verdict=True (RETRO-INERTIA-SWEEPS-TARGET-MISSED)

**What percentage-based sparsification does differently than Exp 901:**
    Exp 901 used per-spin top-k sparsification: each spin keeps its k strongest
    neighbors.  For the N=8 ring+chord graph, this had NO effect because every
    spin already had exactly 3 neighbors (degree = k).

    This module uses a GLOBAL percentile threshold: we compute
    threshold = percentile(|J_flat|, (1-sparsity)*100) and zero out every
    coupling below that threshold.  For a dense random J at N=64, this removes
    80% of entries, leaving a genuinely sparse O(0.2*N^2) coupling graph.

    The sparse matrix is stored as scipy.sparse.csr_matrix for O(nnz) matmul
    instead of O(N^2) — this is the actual per-sweep cost reduction.

**Expected outcome (from analogous literature):**
    arXiv 2604.17109's 15–25x result used specific frustrated spin-glass
    problems where PIMI's parallel independence provides maximal benefit.
    The 4.33x plateau in Exps 889/901 suggests that the N=8 ferromagnetic
    ring+chord graph is too simple for PIMI to shine.  Whether N=64 frustrated
    random J crosses the threshold where sparsification helps is the open
    question this experiment settles.

Spec: REQ-HW-041
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class PIMMISparseAdjacency:
    """PIMI sampler using a global percentile-threshold sparse coupling matrix.

    **Detailed explanation for engineers:**
        Standard PIMI (SynchronousPIMISampler from Exp 889) computes each spin's
        local field as h_local = J @ s, a full O(N^2) dense matrix-vector product.
        At N=64, this is 4096 multiply-adds per sweep.

        This class first builds J_sparse by zeroing out all couplings whose
        magnitude |J[i,j]| falls below the top-sparsity percentile.  With
        sparsity=0.2, we keep only the 20% strongest couplings globally.  The
        remaining 80% are set to zero.  J_sparse is stored as a CSR matrix so
        the h_local computation costs O(nnz) ≈ O(0.2 * N^2) instead of O(N^2).

        The sweep algorithm is identical to SynchronousPIMISampler:
          STEP 1: h_local = J_sparse.dot(s)   # O(nnz) via CSR
          STEP 2: h_ema = alpha * h_ema + (1-alpha) * h_local
          STEP 3: p_flip = sigmoid(-2 * beta * h_ema * s)
                  s_new = flip where rand() < p_flip  [synchronous]

        The EMA inertia_alpha=0.85 is higher than Exp 889's best_alpha=0.5,
        reflecting stronger momentum to compensate for the reduced information
        per sweep from sparsification.

    Args:
        n_spins: Number of spin variables N.
        sparsity: Fraction of couplings to KEEP (0.2 = keep top-20% by |J|).
                  Must be in (0, 1].  1.0 means keep all (no sparsification).
        inertia_alpha: EMA decay factor in [0, 1).  Higher = more inertia.
                       0 = no memory; 1 = never update (pathological).

    Spec: REQ-HW-041
    """

    def __init__(
        self,
        n_spins: int,
        sparsity: float = 0.2,
        inertia_alpha: float = 0.85,
    ) -> None:
        if not (0.0 < sparsity <= 1.0):
            raise ValueError(f"sparsity must be in (0, 1], got {sparsity}")
        if not (0.0 <= inertia_alpha < 1.0):
            raise ValueError(f"inertia_alpha must be in [0, 1), got {inertia_alpha}")

        self.n = n_spins
        self.sparsity = sparsity
        self.alpha = inertia_alpha

        # EMA state — reset to zero before each independent trial.
        self._h_ema: np.ndarray = np.zeros(n_spins, dtype=np.float64)
        # RNG — seeded per-trial in measure_convergence for reproducibility.
        self._rng: np.random.Generator = np.random.default_rng(0)
        # Inverse temperature — beta=1.0 matches Exp 889 config.
        self._beta: float = 1.0

    def build_sparse_J(self, J_full: np.ndarray) -> sp.csr_matrix:
        """Build a sparse coupling matrix by zeroing weak couplings globally.

        **How the threshold is computed:**
            We look at ALL |J[i,j]| values (including both upper and lower
            triangles) and find the (1-sparsity)*100 th percentile.  Any
            coupling with |J[i,j]| < threshold is set to zero.

            For sparsity=0.2:
              threshold = 80th percentile of |J|
              We keep only the top-20% strongest couplings.

            This differs from Exp 901's per-spin top-k approach in two ways:
              1. Global threshold vs per-spin threshold: a spin with no strong
                 couplings may lose ALL its connections here, vs always keeping
                 exactly k neighbors per spin.
              2. Percentage of TOTAL entries vs count per row.

            The resulting nnz / N^2 ratio equals approximately `sparsity`,
            making the effective_sweep_cost predictable.

        Args:
            J_full: Dense coupling matrix, shape (N, N).  Must have zero diagonal.

        Returns:
            J_sparse: scipy.sparse.csr_matrix, shape (N, N).
                      Non-zero entries are the top-sparsity fraction by |J[i,j]|.

        Spec: REQ-HW-041
        """
        J_full = np.asarray(J_full, dtype=np.float64)

        # Compute global threshold on all |J[i,j]| values.
        # percentile of flat array: (1-sparsity)*100 = keep above this.
        # E.g. sparsity=0.2 → percentile=80 → keep top-20%.
        threshold = float(np.percentile(np.abs(J_full), (1.0 - self.sparsity) * 100.0))

        J_sparse_dense = J_full.copy()
        # Zero out entries below threshold (including any exact-threshold ties
        # we choose to keep for consistency — scipy will drop exact zeros).
        J_sparse_dense[np.abs(J_sparse_dense) < threshold] = 0.0

        return sp.csr_matrix(J_sparse_dense)

    def _reset_ema(self) -> None:
        """Reset EMA history to zero for a fresh independent trial.

        Must be called before each convergence trial to prevent EMA from
        carrying over information from a previous run.  See SynchronousPIMISampler.reset()
        for why this matters.

        Spec: REQ-HW-041
        """
        self._h_ema = np.zeros(self.n, dtype=np.float64)

    def sweep_sparse(
        self,
        spins: np.ndarray,
        J_sparse: sp.csr_matrix,
        h: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """One fully-synchronous PIMI cycle using the sparse coupling matrix.

        All spins compute their local fields using the SAME spin snapshot
        (s_current from the start of this cycle), then flip decisions are
        made simultaneously.  This preserves the parallel independence that
        defines PIMI.

        The key computational difference from SynchronousPIMISampler: we use
        J_sparse.dot(s) (CSR sparse matmul, O(nnz)) instead of J @ s (dense
        matmul, O(N^2)).  For sparsity=0.2 and N=64, nnz ≈ 0.2*64^2 = 819
        vs 64^2 = 4096 for the dense case — a ~5x per-sweep compute reduction.

        Args:
            spins: Current spin configuration, shape (N,), values in {-1, +1}.
                   NOT modified in-place; a copy is returned.
            J_sparse: Sparse coupling matrix from build_sparse_J(), CSR format.
            h: External field vector, shape (N,).  Usually zeros.

        Returns:
            Tuple of (s_new, n_flips):
              - s_new: Updated spin configuration, shape (N,), values in {-1, +1}.
              - n_flips: Number of spins that flipped this cycle.

        Spec: REQ-HW-041
        """
        s = np.asarray(spins, dtype=np.float64)

        # STEP 1: local field — O(nnz) via sparse matmul (not O(N^2)).
        # J_sparse.dot(s) only sums over the non-zero coupling pairs.
        h_local = J_sparse.dot(s) + h

        # STEP 2: EMA — blend new observation into running momentum estimate.
        self._h_ema = self.alpha * self._h_ema + (1.0 - self.alpha) * h_local

        # STEP 3: synchronous flip decisions — all spins read s (not s_new).
        argument = np.clip(
            2.0 * self._beta * self._h_ema * s,
            -500.0,
            500.0,
        )
        p_flip = 1.0 / (1.0 + np.exp(argument))
        flip_mask = self._rng.random(self.n) < p_flip

        s_new = s.copy()
        s_new[flip_mask] = -s[flip_mask]
        n_flips = int(np.sum(flip_mask))

        return s_new, n_flips

    def run(
        self,
        n_sweeps: int,
        J_full: np.ndarray,
        h: np.ndarray,
        seed: int = 0,
        init_state: np.ndarray | None = None,
    ) -> dict:
        """Run the sparse PIMI sampler for n_sweeps cycles.

        Builds J_sparse from J_full, runs n_sweeps cycles from init_state
        (random ±1 if None), and returns a results dict including the
        effective_sweep_cost (actual nnz fraction) and sweeps_reduction
        (theoretical speedup from sparsification alone).

        Note: sweeps_reduction here is the COMPUTATIONAL cost ratio
        (dense nnz / sparse nnz = 1.0 / effective_sweep_cost), NOT the
        convergence sweep count ratio.  The experiment script computes
        the convergence sweep ratio from measure_convergence().

        Args:
            n_sweeps: Number of PIMI cycles to run.
            J_full: Dense coupling matrix, shape (N, N).
            h: External field vector, shape (N,).
            seed: RNG seed for reproducibility.
            init_state: Starting spin configuration, or None for random ±1.

        Returns:
            Dict with keys:
              - final_state: np.ndarray of shape (N,)
              - energy_trajectory: list of float energies per sweep
              - effective_sweep_cost: float, nnz / N^2 (fraction of dense)
              - sweeps_reduction: float, N^2 / nnz (theoretical speedup)
              - total_flips: int, total flip events across all sweeps

        Spec: REQ-HW-041
        """
        J_sparse = self.build_sparse_J(J_full)
        h_arr = np.asarray(h, dtype=np.float64)

        # effective_sweep_cost = fraction of N^2 entries that are non-zero.
        # For sparsity=0.2, this is approximately 0.2 (20% of dense cost per sweep).
        nnz = J_sparse.nnz
        effective_sweep_cost = nnz / (self.n**2)

        self._rng = np.random.default_rng(seed)
        self._reset_ema()

        if init_state is None:
            s = (2 * self._rng.integers(0, 2, size=self.n) - 1).astype(np.float64)
        else:
            s = np.asarray(init_state, dtype=np.float64).copy()

        energy_trajectory: list[float] = []
        total_flips = 0

        for _ in range(n_sweeps):
            s, n_flips = self.sweep_sparse(s, J_sparse, h_arr)
            total_flips += n_flips
            # Energy uses sparse J: E = -0.5 * s^T J_sparse s - h^T s
            e = -0.5 * float(s @ J_sparse.dot(s)) - float(h_arr @ s)
            energy_trajectory.append(e)

        return {
            "final_state": s,
            "energy_trajectory": energy_trajectory,
            "effective_sweep_cost": effective_sweep_cost,
            "sweeps_reduction": 1.0 / effective_sweep_cost if effective_sweep_cost > 0 else 0.0,
            "total_flips": total_flips,
        }

    def measure_convergence(
        self,
        J_full: np.ndarray,
        h: np.ndarray,
        n_trials: int,
        target_energy: float,
        max_sweeps: int,
        base_seed: int = 0,
    ) -> tuple[int, float]:
        """Measure mean sweeps-to-converge and effective_sweep_cost over n_trials.

        For each trial: random ±1 init, reset EMA, run sparse PIMI cycles until
        energy(s) < target_energy or max_sweeps reached.  Returns the integer mean
        sweep count and the nnz fraction (effective_sweep_cost).

        Args:
            J_full: Dense coupling matrix, shape (N, N).
            h: External field vector, shape (N,).
            n_trials: Number of independent trials to average.
            target_energy: Convergence threshold (lower = stricter).
            max_sweeps: Cap per trial.
            base_seed: Trial k uses seed base_seed + k.

        Returns:
            Tuple of (mean_sweeps_int, effective_sweep_cost):
              - mean_sweeps_int: int, mean sweeps-to-converge (rounded).
              - effective_sweep_cost: float, nnz / N^2 fraction.

        Spec: REQ-HW-041
        """
        J_sparse = self.build_sparse_J(J_full)
        h_arr = np.asarray(h, dtype=np.float64)
        nnz = J_sparse.nnz
        effective_sweep_cost = nnz / (self.n**2)

        trial_sweeps: list[int] = []

        for trial in range(n_trials):
            self._rng = np.random.default_rng(base_seed + trial)
            self._reset_ema()

            # Random ±1 initialization
            s = (2 * self._rng.integers(0, 2, size=self.n) - 1).astype(np.float64)
            converged_at = max_sweeps

            for sweep in range(1, max_sweeps + 1):
                s, _ = self.sweep_sparse(s, J_sparse, h_arr)
                e = -0.5 * float(s @ J_sparse.dot(s)) - float(h_arr @ s)
                if e < target_energy:
                    converged_at = sweep
                    break

            trial_sweeps.append(converged_at)

        return int(round(float(np.mean(trial_sweeps)))), effective_sweep_cost
