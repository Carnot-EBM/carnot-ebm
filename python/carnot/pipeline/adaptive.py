"""Adaptive constraint weighting from accumulated tracker statistics (Tier 1).

**Researcher summary:**
    Uses ConstraintTracker precision/count statistics to compute per-type
    weights for ComposedEnergy. High-precision, high-count constraint types
    get more weight; noisy or rarely-seen types get downweighted toward a
    minimum floor (0.1) to avoid being zeroed out entirely.

**Detailed explanation for engineers:**
    This is the second building block of Continuous Self-Learning (Tier 1
    from research-program.md). Exp 132 built the ConstraintTracker that
    accumulates precision/recall statistics per constraint type. This module
    uses those statistics to automatically tune the weights fed to
    ComposedEnergy so that the EBM focuses on the most reliable signals.

    The weight formula is:
        w_i = max(precision_i * log(fired_i + 1), WEIGHT_FLOOR)

    Why this formula?
    - ``precision_i``: How often this constraint type actually caught real
      errors (high precision = reliable signal, low = noise). Range [0, 1].
    - ``log(fired_i + 1)``: A count confidence term. A constraint type that
      has fired 1000 times has much stronger statistics than one that has
      fired twice. The log compresses this so a heavily-sampled type doesn't
      dwarf everything else.
    - ``WEIGHT_FLOOR = 0.1``: Prevents any type from being completely zeroed
      out. A zero weight would mean the EBM ignores that constraint entirely,
      which could mask real errors if our statistics are still sparse.

    Workflow:
        # Phase 1: warm-up -- run 100 questions, collecting tracker stats
        tracker = ConstraintTracker()
        for q, r in warmup_questions:
            pipeline.verify(q, r, tracker=tracker)

        # Phase 2: adaptive -- compute weights from warm-up data
        weights = AdaptiveWeighter.from_tracker(tracker)
        AdaptiveWeighter.apply_to_pipeline(pipeline, weights)

        # Phase 3: subsequent verify() calls now use the adaptive weights
        for q, r in eval_questions:
            result = pipeline.verify(q, r)

    The weights dict is keyed by constraint_type string (e.g. "arithmetic",
    "logic") and maps to a positive float weight. Types not present in the
    dict use the default weight of 1.0 (no change from baseline).

    Benchmark helper:
        The ``run_comparison`` function runs a fixed-vs-adaptive comparison
        on a list of (question, response, is_correct) triples. It uses the
        first ``warmup_n`` items to build tracker statistics, then evaluates
        the rest under both fixed (all weights = 1.0) and adaptive weighting.
        Returns a ComparisonResult with accuracy for both conditions.

Spec: REQ-LEARN-002, SCENARIO-LEARN-002
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from carnot.pipeline.tracker import ConstraintTracker

if TYPE_CHECKING:
    from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHT_FLOOR: float = 0.1
"""Minimum weight for any constraint type.

Prevents noisy constraint types from being completely zeroed out. A weight
of 0.1 means the constraint still contributes to the energy, but with 10x
less influence than a fully-trusted (w=1.0) constraint.
"""


# ---------------------------------------------------------------------------
# AdaptiveWeighter
# ---------------------------------------------------------------------------


class AdaptiveWeighter:
    """Compute and apply adaptive constraint weights from tracker statistics.

    **Researcher summary:**
        Converts ConstraintTracker precision/count stats into a weight dict
        that upgrades reliable constraint types and downweights noisy ones.
        Applies the weights to a VerifyRepairPipeline instance so subsequent
        verify() calls use them.

    **Detailed explanation for engineers:**
        This is a stateless utility class (all methods are static). There is
        no instance state -- it is just a namespace for the two core
        operations:

        1. ``from_tracker(tracker)`` -- read stats, compute weights, return
           a plain dict. Callers can inspect or further adjust the dict
           before applying it.

        2. ``apply_to_pipeline(pipeline, weights)`` -- store the weights dict
           on the pipeline as ``_adaptive_weights``. The pipeline's
           ``_evaluate_constraints`` method checks for this attribute and uses
           the stored weights instead of the default 1.0 when building the
           ComposedEnergy for energy-backed constraint terms.

    Spec: REQ-LEARN-002, SCENARIO-LEARN-002
    """

    @staticmethod
    def from_tracker(tracker: ConstraintTracker) -> dict[str, float]:
        """Compute per-type weights from tracker statistics.

        **Detailed explanation for engineers:**
            For each constraint type recorded in the tracker:
              1. Read ``precision`` (caught / fired) and ``fired`` count.
              2. Compute raw weight: precision * log(fired + 1).
              3. Clamp to WEIGHT_FLOOR (0.1) so no type is ever zeroed.

            Constraint types with zero precision (fired but never caught an
            error) get weight = WEIGHT_FLOOR rather than 0. This is
            intentional: a constraint that has never caught an error might
            still be useful once we see the right examples, and we don't
            want to permanently disable it based on sparse data.

            Constraint types not present in the tracker are NOT included in
            the returned dict. The pipeline uses 1.0 as the default weight
            for any constraint type absent from the dict, so unknown types
            are treated as moderately trusted (no change from baseline).

            Example output for a tracker with two types:
                {
                    "arithmetic": 1.732,   # high precision, many fires
                    "logic": 0.1,          # zero precision -> floor
                }

        Args:
            tracker: ConstraintTracker with accumulated statistics.

        Returns:
            Dict mapping constraint_type -> adaptive weight (float >= 0.1).

        Spec: REQ-LEARN-002
        """
        weights: dict[str, float] = {}
        for ctype, entry in tracker.stats().items():
            precision: float = entry["precision"]
            fired: int = entry["fired"]
            # Weight formula: precision * log(count + 1), floor at WEIGHT_FLOOR.
            # log(1 + 1) = 0.693, so a type fired once with perfect precision
            # gets weight 0.693, which is less than the default 1.0.
            # This reflects that we trust a type more the more evidence we have.
            raw_weight = precision * math.log(fired + 1)
            weights[ctype] = max(raw_weight, WEIGHT_FLOOR)
        return weights

    @staticmethod
    def apply_to_pipeline(
        pipeline: "VerifyRepairPipeline",
        weights: dict[str, float],
    ) -> None:
        """Store adaptive weights on the pipeline for use in verify().

        **Detailed explanation for engineers:**
            Sets ``pipeline._adaptive_weights = weights``. The pipeline's
            ``_evaluate_constraints`` method reads this attribute (via
            ``getattr(self, '_adaptive_weights', {})``) and uses the stored
            weight for each constraint's ``constraint_type`` when calling
            ``composed.add_constraint(term, weight)``.

            The method is designed for in-place mutation so callers can
            progressively update the same pipeline instance as more tracker
            data accumulates, without needing to recreate the pipeline.

            To revert to uniform weighting, pass an empty dict or set
            ``pipeline._adaptive_weights = {}`` directly.

        Args:
            pipeline: VerifyRepairPipeline instance to update.
            weights: Dict from ``from_tracker()``, mapping type -> weight.

        Spec: REQ-LEARN-002
        """
        # We store a copy so the caller's dict can be modified without
        # affecting the weights already installed on the pipeline.
        pipeline._adaptive_weights = dict(weights)


# ---------------------------------------------------------------------------
# ComparisonResult
# ---------------------------------------------------------------------------


@dataclass
class ComparisonResult:
    """Result of a fixed-vs-adaptive weight comparison experiment.

    **Detailed explanation for engineers:**
        Returned by ``run_comparison()``. Captures how many questions each
        weighting strategy "agreed with" ground truth (i.e., the pipeline
        verified a correct answer and rejected an incorrect one, or vice
        versa). The accuracy improvement ``delta`` is positive when adaptive
        weighting outperforms fixed.

    Attributes:
        fixed_accuracy: Fraction of eval items where fixed weights gave the
            correct verification verdict.
        adaptive_accuracy: Fraction of eval items where adaptive weights
            gave the correct verification verdict.
        delta: adaptive_accuracy - fixed_accuracy. Positive = improvement.
        warmup_n: Number of items used to build the tracker.
        eval_n: Number of items used in evaluation.
        weights: Adaptive weights computed from the warmup tracker.

    Spec: REQ-LEARN-002, SCENARIO-LEARN-002
    """

    fixed_accuracy: float
    adaptive_accuracy: float
    delta: float
    warmup_n: int
    eval_n: int
    weights: dict[str, float]


# ---------------------------------------------------------------------------
# run_comparison
# ---------------------------------------------------------------------------


def run_comparison(
    questions: list[tuple[str, str, bool]],
    warmup_n: int = 100,
    domain: str | None = "arithmetic",
) -> ComparisonResult:
    """Compare fixed vs adaptive weights on a set of labelled questions.

    **Researcher summary:**
        Uses first ``warmup_n`` items to build a ConstraintTracker, then
        evaluates accuracy on the remaining items under fixed (all 1.0) and
        adaptive weights from the warmup tracker. Returns a ComparisonResult.

    **Detailed explanation for engineers:**
        ``questions`` is a list of (question, response, is_correct) triples:
        - ``question``: The original question text (e.g., from GSM8K).
        - ``response``: A candidate response to verify.
        - ``is_correct``: Ground-truth label -- True if the response is
          actually correct (so a well-calibrated verifier should output
          ``verified=True``), False if the response is wrong (verifier
          should output ``verified=False``).

        The comparison works as follows:
        1. **Warmup phase**: Run the first ``warmup_n`` items through
           ``pipeline.verify()`` with a ConstraintTracker, accumulating
           fired/caught statistics.
        2. **Compute adaptive weights**: Call ``AdaptiveWeighter.from_tracker``
           on the warmup tracker.
        3. **Eval phase (fixed)**: Run remaining items with default weights
           (no ``_adaptive_weights`` set), count items where ``verified``
           matches ``is_correct``.
        4. **Eval phase (adaptive)**: Apply adaptive weights via
           ``AdaptiveWeighter.apply_to_pipeline``, re-run the same eval
           items, count matches with ``is_correct``.

        If ``len(questions) <= warmup_n``, all items are used for warmup
        and the eval phase is empty (returns 0.0 accuracy for both).

        A fresh ``VerifyRepairPipeline`` is created internally so this
        function has no side effects on any externally-held pipeline.

    Args:
        questions: List of (question, response, is_correct) triples.
        warmup_n: Number of items to use for building the tracker (default 100).
        domain: Domain hint for constraint extraction (default "arithmetic").

    Returns:
        ComparisonResult with fixed_accuracy, adaptive_accuracy, delta, etc.

    Spec: REQ-LEARN-002, SCENARIO-LEARN-002
    """
    from carnot.pipeline.verify_repair import VerifyRepairPipeline

    pipeline = VerifyRepairPipeline()

    warmup_items = questions[:warmup_n]
    eval_items = questions[warmup_n:]

    # Phase 1: warmup -- accumulate tracker statistics.
    warmup_tracker = ConstraintTracker()
    for question, response, _is_correct in warmup_items:
        pipeline.verify(question, response, domain=domain, tracker=warmup_tracker)

    # Compute adaptive weights from warmup data.
    weights = AdaptiveWeighter.from_tracker(warmup_tracker)

    if not eval_items:
        return ComparisonResult(
            fixed_accuracy=0.0,
            adaptive_accuracy=0.0,
            delta=0.0,
            warmup_n=len(warmup_items),
            eval_n=0,
            weights=weights,
        )

    # Phase 2: eval with fixed weights (baseline -- no _adaptive_weights set).
    fixed_correct = 0
    for question, response, is_correct in eval_items:
        result = pipeline.verify(question, response, domain=domain)
        if result.verified == is_correct:
            fixed_correct += 1

    # Phase 3: eval with adaptive weights.
    AdaptiveWeighter.apply_to_pipeline(pipeline, weights)
    adaptive_correct = 0
    for question, response, is_correct in eval_items:
        result = pipeline.verify(question, response, domain=domain)
        if result.verified == is_correct:
            adaptive_correct += 1

    eval_n = len(eval_items)
    fixed_acc = fixed_correct / eval_n
    adaptive_acc = adaptive_correct / eval_n

    return ComparisonResult(
        fixed_accuracy=fixed_acc,
        adaptive_accuracy=adaptive_acc,
        delta=adaptive_acc - fixed_acc,
        warmup_n=len(warmup_items),
        eval_n=eval_n,
        weights=weights,
    )
