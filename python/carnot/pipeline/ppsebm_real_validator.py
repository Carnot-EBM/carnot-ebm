"""PPSEBM Real-Data Validator — validates PPSConstraintLearner on naturally-interleaved
real violation sequences (RETRO-043, Exp 485).

**Why interleaved validation is harder than batched validation (the RETRO-043 insight):**
    Exp 470 (PPSConstraintLearner synthetic benchmark) achieved partition_isolation_score
    > 0.8 on SYNTHETIC data with INDEPENDENT domain batches.  The benchmark fed arithmetic
    violations as a batch, then code violations as a batch, then logical violations as a
    batch.  That is easy mode for partition isolation because gradient updates from one
    domain never overlap in time with gradient updates from another.

    In REAL production sessions, a multi-step math problem may reference code snippets,
    draw on logical deductions, and do arithmetic — all within a single 3-step reasoning
    chain.  The domains interleave within the session.  PPSEBM must maintain partition
    isolation (cosine distance > threshold between domain gradient directions) even when
    arithmetic and logic violations occur within the same sliding window.

    This module provides two dataclasses:
    - InterleavedViolationSequence: wraps a list of labeled steps in their natural
      occurrence order, computes the interleaving rate, and produces training batches
      that preserve cross-domain adjacency.
    - PPSEBMRealValidationResult: holds before/after isolation scores and verdicts.

**Connection to arXiv 2512.15658 (PPSEBM):**
    Section 4.3 of the PPSEBM paper notes that partition isolation degrades by 8-15%
    when domain transitions occur at rate > 0.4 per step.  Our interleaved validator
    measures exactly this: what is the actual interleaving_rate in real CoT data, and
    does the learned isolation hold above the 0.7 threshold (slightly relaxed from the
    synthetic 0.8 threshold to account for real-data noise)?

Spec: REQ-SELFLEARN-019, REQ-SELFLEARN-020,
      SCENARIO-SELFLEARN-019, SCENARIO-SELFLEARN-020
      RETRO-043
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# InterleavedViolationSequence
# ---------------------------------------------------------------------------


@dataclass
class InterleavedViolationSequence:
    """A sequence of labeled CoT steps in their NATURAL occurrence order.

    **Why natural ordering matters (RETRO-043):**
        Sorting by domain before creating training batches is what Exp 470 did implicitly
        — it gave the PPSConstraintLearner domain-pure batches, which is best-case for
        partition isolation.  Real sessions produce an interleaved sequence:
        step 1 (arithmetic), step 2 (arithmetic), step 3 (code), step 4 (arithmetic),
        step 5 (logical), ...  Training batches drawn from this sequence will contain
        mixed domains, stressing the partition walls.

    **Interleaving rate:**
        A step pair (i, i+1) is a "domain transition" if step[i].domain != step[i+1].domain.
        interleaving_rate = number_of_transitions / (n_steps - 1).
        Range: [0.0, 1.0].  0.0 = all steps in same domain; 1.0 = every adjacent pair
        switches domain.

    **domain_sequence:**
        The ordered list of domain labels (strings: 'arithmetic', 'code', 'logical')
        for each step, in the order they appeared in the live CoT chain.

    Args:
        steps: List of labeled step dicts.  Each dict must have:
               - 'step_text': str — the raw CoT step text.
               - 'domain': str — one of 'arithmetic', 'code', 'logical'.
               Any additional keys (e.g. 'label', 'confidence') are preserved.

    Raises:
        ValueError: if steps is empty or any step is missing 'domain' key.

    Spec: REQ-SELFLEARN-019, REQ-SELFLEARN-020, SCENARIO-SELFLEARN-019
    """

    steps: list[dict]

    def __post_init__(self) -> None:
        # Validate all steps have a domain label.
        for i, s in enumerate(self.steps):
            if "domain" not in s:
                raise ValueError(
                    f"InterleavedViolationSequence: step[{i}] missing 'domain' key. "
                    "All steps must be annotated with a domain label before constructing "
                    "this sequence.  Use FOVERAnnotator.annotate_corpus() first, then "
                    "assign domain labels from the z3_label result."
                )

    @property
    def domain_sequence(self) -> list[str]:
        """Return the ordered list of domain labels as they appeared in the CoT chain.

        **Why preserve order (not sort):**
            Sorting would destroy the interleaving signal.  The whole point of this
            validator is that real sessions have domains interspersed — we want to see
            the actual sequence: [arithmetic, arithmetic, code, arithmetic, logical, ...].

        Returns:
            List[str] of domain labels, one per step, in natural occurrence order.
        """
        return [s["domain"] for s in self.steps]

    @property
    def interleaving_rate(self) -> float:
        """Fraction of adjacent step pairs where the domain changes.

        A value of 1.0 means every adjacent pair switches domain (maximally interleaved).
        A value of 0.0 means all steps share the same domain (no interleaving — easy mode).

        **Why this metric:**
            When interleaving_rate > 0.4, PPSEBM paper (arXiv 2512.15658, Section 4.3)
            reports 8-15% degradation in partition isolation.  We need to measure the
            actual rate in our real data to contextualise the isolation scores.

        Returns:
            Float in [0.0, 1.0], or 0.0 if fewer than 2 steps.
        """
        seq = self.domain_sequence
        if len(seq) < 2:
            return 0.0
        transitions = sum(1 for a, b in zip(seq, seq[1:]) if a != b)
        return transitions / (len(seq) - 1)

    def to_training_batches(self, batch_size: int) -> list[list[dict]]:
        """Partition the step sequence into batches of *batch_size* in natural order.

        **Why not sort by domain before batching:**
            Sorting by domain turns this into the Exp 470 easy-mode benchmark.  Instead,
            we slice the natural-order sequence into consecutive windows.  Each window will
            contain a mix of domains at whatever interleaving rate exists in the data.
            This is what stresses PPSConstraintLearner's per-update domain isolation.

        Args:
            batch_size: Number of steps per batch (last batch may be smaller).

        Returns:
            List of batches; each batch is a list of step dicts in natural order.
            Returns a list with one empty list if steps is empty.
        """
        if not self.steps:
            return [[]]
        return [
            self.steps[i : i + batch_size]
            for i in range(0, len(self.steps), batch_size)
        ]


# ---------------------------------------------------------------------------
# PPSEBMRealValidationResult
# ---------------------------------------------------------------------------


@dataclass
class PPSEBMRealValidationResult:
    """Holds before/after partition isolation scores for the real-data PPSEBM validation.

    **Why two score fields (before and after):**
        We measure isolation_score_before to establish the untrained baseline.  Training
        on the interleaved sequence updates domain partitions; isolation_score_after
        captures whether the partition walls held up under real interleaved input.
        The delta (after - before) is not the metric — we care about absolute threshold.

    **Why 0.7 threshold (not 0.8 as in Exp 470):**
        The synthetic benchmark used clean domain-pure batches where isolation of 0.8+ is
        achievable.  Real data adds noise: domain label assignment is imperfect (FOVERAnnotator
        assigns 'arithmetic' to all verifiable steps, heuristics for code/logical), and natural
        interleaving introduces realistic cross-domain pressure.  A threshold of 0.7 gives 0.1
        slack while still validating that partition isolation is meaningful in production.

    **better_than_synthetic interpretation:**
        If isolation_score_after >= synthetic_isolation_score, it means real-data interleaving
        did NOT degrade isolation below the synthetic baseline.  This is the strong result
        we want: PPSEBM remains as isolated on real interleaved sequences as it was on synthetic
        clean batches.

    Args:
        n_steps: Number of CoT steps in the validation corpus.
        interleaving_rate: Fraction of adjacent step pairs with different domains.
        isolation_score_before: PartitionIsolationScore before training.
        isolation_score_after: PartitionIsolationScore after training on interleaved steps.
        fp_rate_real: False-positive proxy rate on real test questions.
        synthetic_isolation_score: Baseline isolation score from Exp 470 (synthetic data).
            Defaults to 1.0 if not provided.

    Spec: REQ-SELFLEARN-019, SCENARIO-SELFLEARN-020
    """

    n_steps: int
    interleaving_rate: float
    isolation_score_before: float
    isolation_score_after: float
    fp_rate_real: float
    synthetic_isolation_score: float = 1.0

    @property
    def isolation_maintained(self) -> bool:
        """Return True iff isolation_score_after > 0.7.

        WHY 0.7: see class docstring.  This is RETRO-043's closure condition.

        Spec: REQ-SELFLEARN-019, SCENARIO-SELFLEARN-019
        """
        return self.isolation_score_after > 0.7

    @property
    def better_than_synthetic(self) -> bool:
        """Return True iff isolation_score_after >= synthetic_isolation_score.

        WHY >= (not >): equal performance is sufficient — we are NOT claiming real-data
        training improves isolation, just that it does not degrade it below the synthetic
        baseline.

        Spec: REQ-SELFLEARN-020, SCENARIO-SELFLEARN-020
        """
        return self.isolation_score_after >= self.synthetic_isolation_score


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "InterleavedViolationSequence",
    "PPSEBMRealValidationResult",
]
