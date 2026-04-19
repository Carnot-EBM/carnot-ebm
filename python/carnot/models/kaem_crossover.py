"""KAEMCrossoverResult — data class summarising where KAEM becomes faster than MCMC.

**Why this module exists (RETRO-031):**
    Exp 447 measured KAEM exact-sampling speedup at n_vars <= 100 and found
    mean_speedup=1.29x — well below the 5x production-viability threshold.
    arXiv 2506.14167 predicts that the KAEM advantage grows with n_vars because
    MCMC mixing time scales as O(n^2) while KAEM inverse-transform sampling is
    O(n log n).  The crossover should occur between n_vars=100 and n_vars=500.

    This module provides ``KAEMCrossoverResult`` to aggregate the per-n_vars
    benchmark results from Exp 483 and determine whether a crossover was found.

**What "crossover" means:**
    The crossover n_vars is the smallest n_vars in the tested list where the
    KAEM speedup ratio (ising_mcmc_ms / kaem_ms) reaches or exceeds 5.0x.
    Below that threshold, MCMC is competitive enough that the operational and
    accuracy trade-offs of using KAEM are not justified.

Spec: REQ-SAMPLE-019, SCENARIO-SAMPLE-032
"""

from __future__ import annotations


class KAEMCrossoverResult:
    """Aggregate KAEM vs MCMC crossover profile across multiple n_vars values.

    Holds per-n_vars speedup ratios and derives the crossover point where
    KAEM first achieves >= 5x speedup over ParallelIsingSampler MCMC.

    Parameters
    ----------
    n_vars_tested : list[int]
        Ordered list of n_vars values that were benchmarked, e.g.
        [100, 200, 300, 500, 1000].  Must be non-empty.
    speedups : list[float]
        Corresponding speedup ratios (ising_mcmc_ms / kaem_ms) for each
        entry in n_vars_tested.  Must be the same length as n_vars_tested.

    Raises
    ------
    ValueError
        If n_vars_tested and speedups have different lengths, or if either
        list is empty.

    Spec: REQ-SAMPLE-019, SCENARIO-SAMPLE-032
    """

    # Minimum speedup ratio considered viable for production use.
    # Derived from the RETRO-031 threshold: 5x was chosen as the point where
    # KAEM's lack of cross-variable interaction modelling is outweighed by the
    # throughput gain from MCMC-free sampling.
    VIABILITY_THRESHOLD: float = 5.0

    def __init__(self, n_vars_tested: list[int], speedups: list[float]) -> None:
        if not n_vars_tested:
            raise ValueError("n_vars_tested must be non-empty")
        if len(n_vars_tested) != len(speedups):
            raise ValueError(
                f"n_vars_tested (len={len(n_vars_tested)}) and speedups "
                f"(len={len(speedups)}) must have the same length"
            )

        self._n_vars_tested = list(n_vars_tested)
        self._speedups = list(speedups)
        self._speedup_map: dict[int, float] = dict(zip(n_vars_tested, speedups))

    # ------------------------------------------------------------------
    # crossover_n_vars
    # ------------------------------------------------------------------

    @property
    def crossover_n_vars(self) -> int | None:
        """First n_vars in the tested list where speedup >= VIABILITY_THRESHOLD.

        Returns None if no crossover was found within the tested range.
        A None result means the experiment must be extended to larger n_vars
        before KAEM can be recommended for production constraint verification.

        Spec: REQ-SAMPLE-019
        """
        for n, s in zip(self._n_vars_tested, self._speedups):
            if s >= self.VIABILITY_THRESHOLD:
                return n
        return None

    # ------------------------------------------------------------------
    # max_speedup
    # ------------------------------------------------------------------

    @property
    def max_speedup(self) -> float:
        """Maximum speedup observed across all tested n_vars values.

        Even when no crossover is found, this value characterises the
        growth trajectory of the KAEM advantage and informs whether a
        larger-n_vars experiment is likely to find the crossover.

        Spec: REQ-SAMPLE-019
        """
        return max(self._speedups)

    # ------------------------------------------------------------------
    # kaem_viable_for_production
    # ------------------------------------------------------------------

    @property
    def kaem_viable_for_production(self) -> bool:
        """True iff a crossover was found within the tested n_vars range.

        When True: KAEM achieves >= 5x speedup at crossover_n_vars and is
        recommended for constraint-verification problems of that scale or larger.

        When False: MCMC remains competitive at all tested sizes; additional
        profiling at larger n_vars is needed before adopting KAEM in production.

        Spec: REQ-SAMPLE-019
        """
        return self.crossover_n_vars is not None

    # ------------------------------------------------------------------
    # speedup_at
    # ------------------------------------------------------------------

    def speedup_at(self, n_vars: int) -> float:
        """Return the measured speedup ratio at the given n_vars.

        Parameters
        ----------
        n_vars : int
            Must be one of the values in n_vars_tested.

        Returns
        -------
        float
            Speedup ratio (ising_mcmc_ms / kaem_ms) at that n_vars.

        Raises
        ------
        KeyError
            If n_vars was not in the original n_vars_tested list.

        Spec: REQ-SAMPLE-019, SCENARIO-SAMPLE-032
        """
        if n_vars not in self._speedup_map:
            raise KeyError(
                f"n_vars={n_vars} was not in the tested list {self._n_vars_tested}"
            )
        return self._speedup_map[n_vars]
