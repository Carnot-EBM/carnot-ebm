"""HISR — Hindsight Importance Score Reweighting for constraint violations.

**Researcher summary (arXiv 2603.18683):**
    HISR (Hindsight Importance Score Reweighting) addresses a credit-assignment
    problem in constraint learning: not all observed violations equally predict
    a wrong final answer.  By looking BACKWARD from the outcome — incorrect or
    correct — HISR assigns each violation a score proportional to its proximity
    to the final error.

    This implements the Carnot-specific adaptation of HISR for
    ``ConstraintAdditionFromMemory``.  Instead of RL trajectories, our
    "trajectory" is the sequence of constraint violations within a single
    verify-repair chain.  The final "outcome" is whether the pipeline produced
    a correct or incorrect answer at the end of the chain.

**Why hindsight matters for constraint addition:**
    Violations early in a long reasoning chain may be unrelated to the final
    wrong answer (they got self-corrected).  Violations just before the final
    error are the strongest causal signal.  Standard constraint counting treats
    all violations equally; HISR down-weights early violations and up-weights
    those that immediately preceded failure.

**Score formula:**
    For an incorrect chain with N violations, violation at position i (0-indexed
    from start) receives:

        score_i = 1.0 / (1 + (N - 1 - i))

    This is 1.0 for the LAST violation (distance 0 from the final error),
    0.5 for the second-to-last, 0.333 for the third-to-last, etc.

    For a CORRECT chain all violations are false positives and receive score 0.0.

Spec: REQ-LEARN-072, SCENARIO-LEARN-113, SCENARIO-LEARN-114
"""

from __future__ import annotations

from dataclasses import dataclass

from carnot.pipeline.constraint_addition import ViolationPattern


@dataclass
class HISRViolationWeight:
    """Weight record pairing a violation with its hindsight importance score.

    **Detailed explanation for engineers:**
        This is the output unit of ``HISRWeighter``.  It bundles everything
        needed to decide whether to promote a violation into a new constraint
        template: the violation's type label, which question it came from, the
        binary outcome (correct / incorrect), and the scalar hindsight score.

    Fields
    ------
    violation_type
        Short string identifying the error class, matching ``ViolationPattern.type``
        (e.g. ``'carry'``, ``'sign'``).
    question_id
        Opaque identifier for the question/problem instance this violation appeared in.
        Used to group violations by source for batch analysis.
    final_incorrect
        True when the pipeline's final answer for this question was WRONG.
        False when the final answer was correct (violation was a false positive).
    hindsight_score
        Float in [0.0, 1.0].  Higher = stronger causal signal that this violation
        predicted the final error.  Always 0.0 when final_incorrect=False.
    """

    violation_type: str
    question_id: str
    final_incorrect: bool
    hindsight_score: float


class HISRWeighter:
    """Compute hindsight importance scores for constraint violation sequences.

    **Detailed explanation for engineers:**
        This class is stateless — every call to ``compute_hindsight_score``
        is a pure function of its inputs.  No model loading, no JAX, no GPU.
        It is designed to run in the constraint-addition hot path on CPU.

        Usage pattern::

            weighter = HISRWeighter()
            weights = weighter.compute_hindsight_score(violations, final_correct=False)
            promoted = weighter.weighted_violations(weights, threshold=0.5)

    Spec: REQ-LEARN-072, SCENARIO-LEARN-113, SCENARIO-LEARN-114
    """

    def __init__(self) -> None:
        pass

    def compute_hindsight_score(
        self,
        violations: list[ViolationPattern],
        final_correct: bool,
    ) -> list[HISRViolationWeight]:
        """Assign a hindsight importance score to each violation in a chain.

        **Detailed explanation for engineers:**
            The list ``violations`` is treated as an ordered sequence — position
            in the list encodes temporal order within the verify-repair chain.
            Index 0 is the earliest violation, index N-1 is the latest.

            When final_correct=True, every violation was a false alarm (the
            model recovered or was never wrong).  Giving them score 0.0 prevents
            HISR from polluting the constraint-addition signal with noise.

            When final_correct=False, violations closer to the end of the chain
            are more causally responsible for the final error.  The formula
            1.0 / (1 + distance_from_last) gives score 1.0 to the final
            violation and decays as you move earlier in the sequence.

        Args:
            violations: Ordered list of ``ViolationPattern`` objects representing
                one verify-repair chain.  Order must match temporal occurrence.
            final_correct: True if the pipeline's final answer was correct,
                False if the final answer was wrong.

        Returns:
            List of ``HISRViolationWeight`` in the same order as ``violations``.
            Each weight has a ``question_id`` of ``""`` (caller may fill it in).

        Spec: REQ-LEARN-072
        """
        if not violations:
            return []

        n = len(violations)
        weights: list[HISRViolationWeight] = []

        for i, v in enumerate(violations):
            if final_correct:
                # All violations in a correct chain are false positives.
                score = 0.0
            else:
                # Distance from last violation: the final violation has distance 0.
                distance_from_last = (n - 1) - i
                score = 1.0 / (1 + distance_from_last)

            weights.append(
                HISRViolationWeight(
                    violation_type=v.type,
                    question_id="",
                    final_incorrect=not final_correct,
                    hindsight_score=score,
                )
            )

        return weights

    def weighted_violations(
        self,
        weights: list[HISRViolationWeight],
        threshold: float = 0.5,
    ) -> list[ViolationPattern]:
        """Filter violation weights to those above the hindsight score threshold.

        **Detailed explanation for engineers:**
            Only weights with hindsight_score >= threshold are considered
            high-signal.  The threshold of 0.5 retains violations in the final
            half of the error chain (i.e., the last two violations in a 3-step
            chain).  Callers pass the original ``ViolationPattern`` list in
            parallel with ``weights`` so this method can reconstruct them.

            WHY return ViolationPattern instead of HISRViolationWeight:
            ConstraintAdditionFromMemory.observe() expects ViolationPattern,
            so we reconstruct minimal ViolationPattern objects from the type
            label.  The count and example_steps can be set by the caller after
            the fact.

        Args:
            weights: Output from ``compute_hindsight_score``.
            threshold: Minimum hindsight_score to retain.  Default 0.5.

        Returns:
            List of ``ViolationPattern`` for weights that meet the threshold.
            Count is set to 1; example_steps is empty.  Caller may enrich them.

        Spec: REQ-LEARN-072
        """
        return [
            ViolationPattern(type=w.violation_type, count=1, example_steps=[])
            for w in weights
            if w.hindsight_score >= threshold
        ]
