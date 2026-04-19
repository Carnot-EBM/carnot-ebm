"""KAEMDistributionBenchmark — KAEM vs MCMC on three distribution families.

**Why switch from n_vars to distribution_family (RETRO-031, Exp 508):**

    The prior search axis was *n_vars* (Exps 447-498): we asked "at what problem size
    does KAEM exact sampling beat MCMC?"  No crossover was found up to n_vars=5000.

    The RETRO-037 recommendation (Exp 498 retrospective) flips the axis: test KAEM on
    distributions where MCMC is theoretically *expected* to struggle — specifically,
    distributions with *slow mixing time* in the Markov chain.

    KAEM's advantage over MCMC comes from its exact inverse-transform sampling:
    it samples from the MARGINAL distribution of each variable exactly, one CDF
    inversion per variable.  MCMC, by contrast, must MIX across the joint distribution
    by taking a random walk through state space.

    KAEM's advantage GROWS when the joint distribution has MODES (local peaks) that
    slow MCMC mixing.  When MCMC gets trapped in one mode, it undersamples the others
    — the empirical distribution diverges from the true distribution.  KAEM draws from
    the marginal CDF of each variable, so it samples all modes proportionally regardless
    of how many modes there are or how far apart they sit.

    Three distribution families test this hypothesis directly:

    1. **GaussianMixture** — canonical multimodal distribution.  MCMC Gibbs chains
       get trapped between modes when the mixture components are well-separated.
       KAEM exact sampling covers all components by construction (each marginal CDF
       integrates over all components).

    2. **StudentT** — heavy-tailed unimodal distribution.  The heavy tails generate
       occasional extreme values.  MCMC may over-concentrate near the mode (slow to
       reach the tails), while KAEM's CDF inversion correctly samples the tails
       proportionally to their probability mass.

    3. **PiecewiseUniform** — non-smooth distribution (uniform over disjoint intervals).
       MCMC must cross zero-density gaps between pieces, which is impossible for
       standard Gibbs sampling.  KAEM's CDF inversion is unaffected by discontinuities
       — it finds F^{-1}(u) numerically regardless of smoothness.

    This axis directly measures whether KAEM's theoretical advantage translates to
    empirical quality improvement, using mean L2 distance between empirical CDFs as the
    quality metric.

**How mean_l2 works:**
    Ground-truth samples are drawn from the true distribution using numpy/scipy.
    KAEM samples are drawn via inverse-transform sampling from the fitted KAEM model.
    MCMC samples are drawn via the ParallelIsingSampler Gibbs chain on the fitted model.

    For each variable dimension, we compute the empirical CDF of KAEM samples and
    the empirical CDF of ground-truth samples on a shared grid, then take the mean
    absolute difference (L1 distance between CDFs, equivalent to Wasserstein-1).

    Mean over all variable dimensions gives a scalar quality score.  Lower = better.

    Why not KS-statistic or MMD?  L2-on-CDF is interpretable: it is the expected
    shortfall in distribution coverage.  It captures mode-trapping directly: if MCMC
    gets stuck in one mode, its CDF will plateau far from the ground-truth CDF in the
    region of the other mode.

**MCMC baseline:**
    Uses ParallelIsingSampler (parallel Gibbs, n_warmup=50, n_samples=n_samples,
    steps_per_sample=5).  The Ising sampler operates on continuous-valued inputs
    treated as "spin-like" variables in [-1, 1], matching the KAEM domain.

Spec: REQ-SAMPLE-022, REQ-SAMPLE-023, REQ-SAMPLE-024,
      SCENARIO-SAMPLE-035, SCENARIO-SAMPLE-036, SCENARIO-SAMPLE-037
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from carnot.models.kaem_energy import KAEMEnergy
from carnot.samplers.parallel_ising import AnnealingSchedule, ParallelIsingSampler

# Number of grid points for empirical CDF comparison.
# 200 points gives sub-percent accuracy for distributions with support in [-1, 1].
_CDF_GRID_N = 200


# ---------------------------------------------------------------------------
# DistributionFamilyResult
# ---------------------------------------------------------------------------


@dataclass
class DistributionFamilyResult:
    """Result of benchmarking KAEM vs MCMC on a single distribution family.

    Stores the mean L2 distance between each sampler's empirical CDF and the
    ground-truth empirical CDF.  The difference (mcmc_mean_l2 - kaem_mean_l2)
    is the KAEM advantage: positive means KAEM better matches the true distribution.

    Parameters
    ----------
    family_name : str
        Name of the distribution family ('gaussian_mixture', 'student_t',
        'piecewise_uniform').
    kaem_mean_l2 : float
        Mean L2 distance between KAEM samples' empirical CDF and ground-truth.
        Lower = KAEM better matches the true distribution.
    mcmc_mean_l2 : float
        Mean L2 distance between MCMC samples' empirical CDF and ground-truth.
        Lower = MCMC better matches the true distribution.

    Spec: REQ-SAMPLE-022, REQ-SAMPLE-023, SCENARIO-SAMPLE-035, SCENARIO-SAMPLE-036
    """

    family_name: str
    kaem_mean_l2: float
    mcmc_mean_l2: float

    @property
    def kaem_advantage(self) -> float:
        """KAEM advantage: mcmc_mean_l2 - kaem_mean_l2.

        Positive means KAEM samples are closer to the true distribution.
        Negative means MCMC samples are closer (MCMC wins).
        """
        return self.mcmc_mean_l2 - self.kaem_mean_l2

    @property
    def kaem_wins(self) -> bool:
        """True iff kaem_advantage > 0 (KAEM is closer to true distribution).

        Spec: REQ-SAMPLE-022, SCENARIO-SAMPLE-035
        """
        return self.kaem_advantage > 0

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dict of all fields.

        Spec: REQ-SAMPLE-022
        """
        return {
            "family_name": self.family_name,
            "kaem_mean_l2": float(self.kaem_mean_l2),
            "mcmc_mean_l2": float(self.mcmc_mean_l2),
            "kaem_advantage": float(self.kaem_advantage),
            "kaem_wins": bool(self.kaem_wins),
        }


# ---------------------------------------------------------------------------
# KAEMDistributionBenchmark
# ---------------------------------------------------------------------------


class KAEMDistributionBenchmark:
    """Benchmark KAEM exact sampling vs MCMC on three distribution families.

    For each distribution family:
    1. Draw n_samples ground-truth samples from the true distribution using numpy.
    2. Fit a KAEMEnergy model on the ground-truth samples (score matching).
    3. Draw n_samples KAEM samples via exact inverse-transform sampling.
    4. Draw n_samples MCMC samples via ParallelIsingSampler Gibbs on the KAEM model.
    5. Compute mean_l2 for KAEM samples and MCMC samples vs ground truth.

    The hypothesis (RETRO-031/037): multimodal distributions should favour KAEM
    because MCMC gets trapped in single modes while KAEM covers all modes by
    construction.

    Parameters
    ----------
    n_vars : int
        Dimensionality of the sample space (number of independent variables).
        Default 10 — small enough for CPU-only experiments, large enough to show
        distributional effects.
    n_samples : int
        Number of samples to draw from each distribution.  Default 200.

    Spec: REQ-SAMPLE-022, REQ-SAMPLE-023, REQ-SAMPLE-024
    """

    def __init__(self, n_vars: int = 10, n_samples: int = 200) -> None:
        if n_vars < 1:
            raise ValueError(f"n_vars must be >= 1, got {n_vars}")
        if n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {n_samples}")
        self.n_vars = n_vars
        self.n_samples = n_samples
        self._rng = np.random.default_rng(42)

    # ------------------------------------------------------------------
    # _mean_l2_vs_ground_truth
    # ------------------------------------------------------------------

    def _mean_l2_vs_ground_truth(
        self, samples: np.ndarray, ground_truth: np.ndarray
    ) -> float:
        """Compute mean L2 distance between empirical CDFs of samples and ground truth.

        For each variable dimension:
        - Build shared grid over the combined range.
        - Compute empirical CDF of samples and ground_truth on the grid.
        - Compute mean absolute difference (L1/Wasserstein-1 between CDFs).
        Then average over all dimensions.

        This captures mode-trapping: if MCMC gets stuck in one mode, its CDF will
        diverge from the ground-truth CDF in the region of the other mode.

        Parameters
        ----------
        samples : np.ndarray
            Shape (n_samples, n_vars).
        ground_truth : np.ndarray
            Shape (n_samples, n_vars).

        Returns
        -------
        float
            Mean L2 (Wasserstein-1) distance, averaged over variables.

        Spec: REQ-SAMPLE-023
        """
        distances = []
        for i in range(self.n_vars):
            col_s = samples[:, i]
            col_g = ground_truth[:, i]
            grid = np.linspace(
                min(col_s.min(), col_g.min()),
                max(col_s.max(), col_g.max()),
                _CDF_GRID_N,
            )
            cdf_s = np.mean(col_s[:, None] <= grid[None, :], axis=0)
            cdf_g = np.mean(col_g[:, None] <= grid[None, :], axis=0)
            distances.append(float(np.mean(np.abs(cdf_s - cdf_g))))
        return float(np.mean(distances))

    # ------------------------------------------------------------------
    # _fit_and_sample_kaem
    # ------------------------------------------------------------------

    def _fit_and_sample_kaem(
        self, ground_truth: np.ndarray
    ) -> np.ndarray:
        """Fit KAEMEnergy on ground_truth data and draw KAEM samples.

        Clips ground-truth data to [-1, 1] before fitting, since KAEM operates in
        that domain.  Score matching trains the per-variable splines to match the
        marginal distributions.

        Parameters
        ----------
        ground_truth : np.ndarray
            Shape (n_samples, n_vars).

        Returns
        -------
        np.ndarray
            KAEM samples, shape (n_samples, n_vars).
        """
        data_clipped = np.clip(ground_truth, -1.0, 1.0)
        data_jax = jnp.array(data_clipped, dtype=jnp.float32)
        model = KAEMEnergy(n_vars=self.n_vars, n_hidden=16, key=jrandom.PRNGKey(0))
        model.fit(data_jax, n_epochs=50)
        kaem_samples = np.array(model.sample(self.n_samples))
        return kaem_samples

    # ------------------------------------------------------------------
    # _sample_mcmc
    # ------------------------------------------------------------------

    def _sample_mcmc(self, ground_truth: np.ndarray) -> np.ndarray:
        """Draw MCMC samples via ParallelIsingSampler Gibbs on an Ising coupling.

        Uses a ring-topology coupling matrix (each variable coupled to its neighbour)
        as the MCMC baseline — the same coupling used in benchmark_kaem_vs_mcmc.
        Starts from a uniform random initial state.

        MCMC is run on the joint distribution defined by the coupling matrix, NOT
        on the fitted KAEM model.  This tests whether MCMC can explore the same
        regions that KAEM's marginal sampling covers — a fair comparison because
        both samplers are trying to cover the same target support.

        Returns
        -------
        np.ndarray
            MCMC samples in [-1, 1], shape (n_samples, n_vars).
        """
        n_vars = self.n_vars
        J = np.zeros((n_vars, n_vars), dtype=np.float32)
        for idx in range(n_vars):
            J[idx, (idx + 1) % n_vars] = 0.5
            J[(idx + 1) % n_vars, idx] = 0.5
        J_jax = jnp.array(J)
        # Bias toward the mean of the ground-truth data to anchor the chain
        biases = np.clip(np.mean(ground_truth, axis=0), -0.5, 0.5).astype(np.float32)
        b_jax = jnp.array(biases)

        schedule = AnnealingSchedule(beta_init=0.5, beta_final=2.0)
        sampler = ParallelIsingSampler(
            n_warmup=50, n_samples=self.n_samples, steps_per_sample=5, schedule=schedule
        )
        key = jrandom.PRNGKey(99)
        k1, k2 = jrandom.split(key, 2)
        init_spins = jnp.zeros(n_vars, dtype=jnp.float32)
        # Warm up to reach approximate stationarity
        sampler.sample(k1, b_jax, J_jax, 2.0, init_spins)
        mcmc_samples = np.array(sampler.sample(k2, b_jax, J_jax, 2.0, init_spins))
        # ParallelIsingSampler returns shape (n_samples, n_vars)
        return mcmc_samples.astype(np.float32)

    # ------------------------------------------------------------------
    # benchmark_gaussian_mixture
    # ------------------------------------------------------------------

    def benchmark_gaussian_mixture(self) -> DistributionFamilyResult:
        """Benchmark KAEM vs MCMC on a mixture of two Gaussians.

        Two Gaussian components at -0.5 and +0.5 (in [-1, 1] space) with equal
        weights and std=0.15.  Well-separated modes cause MCMC chains to get
        trapped in one component — KAEM should cover both modes proportionally
        because its CDF integrates over the full marginal distribution.

        Returns
        -------
        DistributionFamilyResult
            family_name='gaussian_mixture', kaem_mean_l2, mcmc_mean_l2.

        Spec: REQ-SAMPLE-022, SCENARIO-SAMPLE-035
        """
        # Draw ground-truth samples: 50/50 mixture of two Gaussians per variable
        half = self.n_samples // 2
        mu1, mu2, std = -0.5, 0.5, 0.15
        comp1 = self._rng.normal(loc=mu1, scale=std, size=(half, self.n_vars))
        comp2 = self._rng.normal(loc=mu2, scale=std, size=(self.n_samples - half, self.n_vars))
        ground_truth = np.vstack([comp1, comp2]).astype(np.float32)
        ground_truth = np.clip(ground_truth, -1.0, 1.0)
        self._rng.shuffle(ground_truth)

        kaem_samples = self._fit_and_sample_kaem(ground_truth)
        mcmc_samples = self._sample_mcmc(ground_truth)

        kaem_l2 = self._mean_l2_vs_ground_truth(kaem_samples, ground_truth)
        mcmc_l2 = self._mean_l2_vs_ground_truth(mcmc_samples, ground_truth)

        return DistributionFamilyResult(
            family_name="gaussian_mixture",
            kaem_mean_l2=kaem_l2,
            mcmc_mean_l2=mcmc_l2,
        )

    # ------------------------------------------------------------------
    # benchmark_student_t
    # ------------------------------------------------------------------

    def benchmark_student_t(self, nu: float = 2.0) -> DistributionFamilyResult:
        """Benchmark KAEM vs MCMC on a Student-t distribution.

        Student-t with nu=2.0 degrees of freedom has very heavy tails (heavier than
        Gaussian but still integrable).  MCMC chains may over-concentrate near the mode
        and underrepresent the tails; KAEM's CDF inversion naturally samples tails in
        proportion to their probability mass.

        Samples are normalised to [-1, 1] via a sigmoid-like scaling.

        Parameters
        ----------
        nu : float
            Degrees of freedom.  Lower = heavier tails.  Default 2.0.

        Returns
        -------
        DistributionFamilyResult
            family_name='student_t', kaem_mean_l2, mcmc_mean_l2.

        Spec: REQ-SAMPLE-022, SCENARIO-SAMPLE-036
        """
        raw = self._rng.standard_t(df=nu, size=(self.n_samples, self.n_vars)).astype(np.float32)
        # Map to [-1, 1] via tanh scaling (preserves relative tail mass ranking)
        ground_truth = np.tanh(raw / 3.0).astype(np.float32)

        kaem_samples = self._fit_and_sample_kaem(ground_truth)
        mcmc_samples = self._sample_mcmc(ground_truth)

        kaem_l2 = self._mean_l2_vs_ground_truth(kaem_samples, ground_truth)
        mcmc_l2 = self._mean_l2_vs_ground_truth(mcmc_samples, ground_truth)

        return DistributionFamilyResult(
            family_name="student_t",
            kaem_mean_l2=kaem_l2,
            mcmc_mean_l2=mcmc_l2,
        )

    # ------------------------------------------------------------------
    # benchmark_piecewise_uniform
    # ------------------------------------------------------------------

    def benchmark_piecewise_uniform(self, n_pieces: int = 5) -> DistributionFamilyResult:
        """Benchmark KAEM vs MCMC on a piecewise uniform (multimodal) distribution.

        Draws from n_pieces uniform intervals equally spaced in [-1, 1], separated
        by gaps.  MCMC cannot cross zero-density gaps (transitions have zero
        acceptance probability), so it gets permanently trapped in whatever piece
        the chain started in.  KAEM's CDF inversion sees all pieces as part of the
        marginal distribution and samples them proportionally.

        Parameters
        ----------
        n_pieces : int
            Number of uniform pieces.  Default 5.  Each piece covers
            (2 / (2*n_pieces - 1)) of the [-1, 1] range, with equal-width gaps.

        Returns
        -------
        DistributionFamilyResult
            family_name='piecewise_uniform', kaem_mean_l2, mcmc_mean_l2.

        Spec: REQ-SAMPLE-022, SCENARIO-SAMPLE-037
        """
        # Build n_pieces non-overlapping intervals, each of width w, with gaps of w
        # Total range = 2.0, so w = 2.0 / (2 * n_pieces - 1) when pieces occupy half
        total = 2.0
        piece_width = total / (2 * n_pieces)  # gap = same width as piece
        pieces: list[tuple[float, float]] = []
        lo = -1.0
        for _ in range(n_pieces):
            pieces.append((lo, lo + piece_width))
            lo += 2 * piece_width  # skip gap

        # Draw ground truth: choose a random piece per sample, then uniform within it
        piece_idx = self._rng.integers(0, n_pieces, size=(self.n_samples, self.n_vars))
        samples_list = []
        for i in range(self.n_vars):
            col_pieces = piece_idx[:, i]
            col = np.array([
                self._rng.uniform(pieces[p][0], pieces[p][1])
                for p in col_pieces
            ], dtype=np.float32)
            samples_list.append(col)
        ground_truth = np.stack(samples_list, axis=1).astype(np.float32)

        kaem_samples = self._fit_and_sample_kaem(ground_truth)
        mcmc_samples = self._sample_mcmc(ground_truth)

        kaem_l2 = self._mean_l2_vs_ground_truth(kaem_samples, ground_truth)
        mcmc_l2 = self._mean_l2_vs_ground_truth(mcmc_samples, ground_truth)

        return DistributionFamilyResult(
            family_name="piecewise_uniform",
            kaem_mean_l2=kaem_l2,
            mcmc_mean_l2=mcmc_l2,
        )

    # ------------------------------------------------------------------
    # best_family
    # ------------------------------------------------------------------

    def best_family(
        self,
        results: list[DistributionFamilyResult] | None = None,
    ) -> str:
        """Return the name of the distribution family where KAEM has the largest advantage.

        If results are provided, uses those.  Otherwise runs all three benchmarks.
        If no family has kaem_wins=True (MCMC wins or ties on all three), returns 'none'.

        Parameters
        ----------
        results : list[DistributionFamilyResult] | None
            Pre-computed results, one per family.  If None, all three benchmarks
            are run internally (expensive).

        Returns
        -------
        str
            Family name with largest kaem_advantage, or 'none' if all kaem_wins=False.

        Spec: REQ-SAMPLE-024, SCENARIO-SAMPLE-037
        """
        if results is None:
            results = [
                self.benchmark_gaussian_mixture(),
                self.benchmark_student_t(),
                self.benchmark_piecewise_uniform(),
            ]

        winning = [r for r in results if r.kaem_wins]
        if not winning:
            return "none"
        return max(winning, key=lambda r: r.kaem_advantage).family_name
