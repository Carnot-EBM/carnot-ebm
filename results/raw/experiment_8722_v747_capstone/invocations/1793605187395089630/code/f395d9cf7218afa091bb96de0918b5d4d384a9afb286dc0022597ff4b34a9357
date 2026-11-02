"""KAEMExtendedResult — extends KAEMCrossoverResult for the n_vars=1000..5000 profile.

**Why this module exists (RETRO-031, Exp 498):**
    Exp 483 profiled KAEM at n_vars up to 1000 and found no 5x speedup crossover.
    The theoretical prediction (O(n^2) MCMC mixing time vs O(n log n) KAEM
    inverse-transform sampling) suggests crossover between n=1000 and n=5000.

    This module extends KAEMCrossoverResult with two new verdicts:
    1. ``kaem_viable_for_cpu`` — True only if a crossover was found within the
       extended range.  This is distinct from the parent's ``kaem_viable_for_production``
       which was true when ANY crossover was found at any tested size.
    2. ``fpga_path_recommended`` — True when kaem_viable_for_cpu is False, which
       signals that FPGA bisection arithmetic (where O(log n) per step is native)
       should be the next hardware target for KAEM.

**What "prior_max_n" means:**
    The previous experiment (Exp 483) tested up to n_vars=1000.  We record
    that ceiling here so that the result artifact can unambiguously express
    "this is an extension of the prior work, not a standalone benchmark."

**What "vs_prior" means:**
    A short label that tells readers scanning the conductor log whether the
    extended profile resolved the RETRO-031 question:
    - ``'crossover_found'`` — we found the 5x crossover; KAEM is viable on CPU.
    - ``'no_crossover_extended'`` — even at n=5000 no crossover; FPGA path recommended.

Spec: REQ-SAMPLE-020, REQ-SAMPLE-021,
      SCENARIO-SAMPLE-033, SCENARIO-SAMPLE-034
"""

from __future__ import annotations

from carnot.models.kaem_crossover import KAEMCrossoverResult


class KAEMExtendedResult(KAEMCrossoverResult):
    """Extended KAEM crossover profile for n_vars=(1000, 2000, 3000, 5000).

    Inherits all fields and properties from KAEMCrossoverResult and adds
    CPU-viability verdict fields that close RETRO-031.

    Parameters
    ----------
    n_vars_tested : list[int]
        Ordered list of n_vars values benchmarked in this extended run.
        Typically [1000, 2000, 3000, 5000] with possible early stop.
    speedups : list[float]
        Corresponding speedup ratios (ising_mcmc_ms / kaem_ms) for each
        entry in n_vars_tested.  Same length as n_vars_tested.
    prior_max_n : int
        Largest n_vars that the preceding experiment (Exp 483) tested.
        Stored for provenance — lets the conductor log show that this run
        is a direct continuation, not an independent benchmark.

    Spec: REQ-SAMPLE-020, REQ-SAMPLE-021,
          SCENARIO-SAMPLE-033, SCENARIO-SAMPLE-034
    """

    def __init__(
        self,
        n_vars_tested: list[int],
        speedups: list[float],
        prior_max_n: int,
    ) -> None:
        super().__init__(n_vars_tested, speedups)
        self._prior_max_n = prior_max_n

    # ------------------------------------------------------------------
    # prior_max_n
    # ------------------------------------------------------------------

    @property
    def prior_max_n(self) -> int:
        """Largest n_vars tested by the preceding experiment (Exp 483).

        This is a provenance field: it lets the result artifact express that
        the current benchmark started where Exp 483 left off, so reviewers
        can confirm there is no gap in the profiling range.

        Spec: REQ-SAMPLE-020
        """
        return self._prior_max_n

    # ------------------------------------------------------------------
    # kaem_viable_for_cpu
    # ------------------------------------------------------------------

    @property
    def kaem_viable_for_cpu(self) -> bool:
        """True iff a 5x speedup crossover was found within the extended n_vars range.

        When True: KAEM achieves >= 5x speedup over MCMC at some CPU-feasible n_vars.
        The crossover_n_vars property (inherited from KAEMCrossoverResult) gives the
        exact size.

        When False: MCMC remains competitive at all tested sizes up to and including
        n_vars=5000, which is the maximum practical size for this CPU-only profiling
        pass.  The FPGA path (where bisection is native arithmetic) is recommended.

        This property directly closes RETRO-031 by giving a boolean verdict that
        the conductor can record without ambiguity.

        Spec: REQ-SAMPLE-021, SCENARIO-SAMPLE-033, SCENARIO-SAMPLE-034
        """
        return self.crossover_n_vars is not None

    # ------------------------------------------------------------------
    # fpga_path_recommended
    # ------------------------------------------------------------------

    @property
    def fpga_path_recommended(self) -> bool:
        """True iff KAEM is NOT viable for CPU at the tested n_vars range.

        When True: the inverse-transform sampling bisection at the heart of KAEM
        is fast enough to matter only when implemented in hardware where each
        bisection step is a single clock cycle.  An FPGA Ising machine (e.g.
        on the KV260 arriving 2026-04-20) should be the next experimental target.

        This is the logical complement of kaem_viable_for_cpu and exists as a
        separate field so the conductor can check both the negative verdict AND
        the forward recommendation in a single JSON read.

        Spec: REQ-SAMPLE-021
        """
        return not self.kaem_viable_for_cpu

    # ------------------------------------------------------------------
    # vs_prior
    # ------------------------------------------------------------------

    @property
    def vs_prior(self) -> str:
        """Short label comparing this result to the prior Exp 483 result.

        Returns
        -------
        str
            ``'crossover_found'`` if kaem_viable_for_cpu is True (crossover
            was found in the extended range — resolves RETRO-031 affirmatively).
            ``'no_crossover_extended'`` if kaem_viable_for_cpu is False (the
            extended range also found no crossover — RETRO-031 closed as FPGA).

        Spec: REQ-SAMPLE-020, SCENARIO-SAMPLE-033, SCENARIO-SAMPLE-034
        """
        if self.kaem_viable_for_cpu:
            return "crossover_found"
        return "no_crossover_extended"
