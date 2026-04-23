"""Model-adaptive constraint thresholds and selective CaseMemory consolidation.

**Researcher summary:**
    Exp 331 FP autopsy showed that certain constraint types (e.g. NL2Z3 range checks)
    have systematically high false-positive rates on small models like Qwen3.5-0.8B
    but are accurate on larger models.  Blindly applying all constraint types to all
    models wastes repair budget and introduces regressions where the pipeline "fixes"
    correct answers.

    This module provides:
    1. ``PerModelFPTracker`` — tracks fp_count and tp_count per (model_id,
       constraint_type) pair and surfaces a ``should_disable()`` predicate.
       When fp_rate > tp_rate AND enough observations exist, the constraint
       type is suppressed for that model.

    2. ``ModelAdaptiveThresholds`` — wraps any ConstraintExtractor and post-filters
       its violations through ``PerModelFPTracker.get_active_constraint_types()``.

    3. ``SelectiveConsolidation`` — implements the ATLAS (arXiv 2511.01093)
       selective memory strategy.  Only high-contrast interactions are retained
       in CaseMemory: those where the verified violation energy DISAGREES with
       the model's confidence direction.  Low-contrast traces (where verification
       agreed with the model) carry weak learning signal and inflate memory.
       Target: 0.3–0.5 consolidation ratio while maintaining precision.

**Detailed explanation for engineers:**
    The core insight is that not all observed cases are equally informative.
    A case where the EBM reports high violation energy AND the model already
    indicated low confidence is expected — the model knew it was wrong.  The
    cases we want to remember are the SURPRISING ones: high energy but the
    model seemed confident (the model was overconfident and the EBM caught it),
    or low energy but the model seemed uncertain (the EBM cleared a case the
    model was unsure about).  These surprising disagreements are exactly the
    training signal that makes CaseMemory useful for future retrieval.

Spec: REQ-LEARN-015, REQ-LEARN-016,
      SCENARIO-LEARN-025, SCENARIO-LEARN-026,
      SCENARIO-LEARN-027, SCENARIO-LEARN-028
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from carnot.pipeline.extract import ConstraintExtractor
    from carnot.pipeline.fr11_event_bus import ViolationEvent

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WeightState — per-constraint weight snapshot for convergence auditing
# ---------------------------------------------------------------------------


@dataclass
class WeightState:
    """Snapshot of a single constraint type's learned weight.

    **Why these fields:**
        ``weight`` is the current scalar amplifier applied to energy contributions
        from this constraint type at inference time.  Starts at 1.0 and increments
        by 0.01 on each un-throttled ViolationEvent (capped at 2.0).

        ``update_count`` is the number of times the weight was actually changed
        (i.e., un-throttled events only).  Throttled events (1-in-10 cadence) do
        NOT increment this counter, so update_count reflects real learning steps.

        ``last_updated_at`` is an ISO-8601 UTC timestamp string set whenever the
        weight changes.  This lets auditors see recency: a weight that hasn't
        been updated in hours may be stale.

    Spec: REQ-FR11-007, REQ-FR11-007-2
    """

    weight: float
    update_count: int
    last_updated_at: Optional[str]


# ---------------------------------------------------------------------------
# PerModelFPTracker
# ---------------------------------------------------------------------------


class PerModelFPTracker:
    """Track false-positive and true-positive rates per (model_id, constraint_type).

    **Detailed explanation for engineers:**
        Each call to ``update()`` increments running counters for the given
        (model_id, constraint_type) pair.  When ``should_disable()`` is queried,
        it computes fp_rate = fp_count / n_observations and tp_rate = tp_count /
        n_observations and returns True if fp_rate strictly exceeds tp_rate AND
        the pair has been seen at least ``min_observations`` times.

        The ``min_observations`` guard prevents premature disabling on noisy
        early data — with fewer than 10 observations the rate estimates are
        too unreliable to act on.

        Use ``to_dict()`` / ``from_dict()`` to persist the tracker across
        experiment runs so the calibration accumulates over time rather than
        resetting on each run.

    Args:
        min_observations: Minimum number of observations before a constraint
                          type can be disabled.  Default 10.

    Spec: REQ-LEARN-015-1, REQ-LEARN-015-2, REQ-LEARN-015-3, REQ-LEARN-015-5
    """

    def __init__(self, min_observations: int = 10) -> None:
        self._min_observations = min_observations
        # Keys: (model_id, constraint_type); values: dict with fp_count, tp_count, n_observations
        self._stats: dict[tuple[str, str], dict[str, int]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(
        self,
        model_id: str,
        constraint_type: str,
        *,
        was_fp: bool,
        was_tp: bool,
    ) -> None:
        """Record one observation for the (model_id, constraint_type) pair.

        **Detailed explanation for engineers:**
            was_fp and was_tp are not mutually exclusive in theory (an observation
            could be both if tracking is granular), but in practice each
            observation is one of: FP, TP, or neither (the constraint fired but
            we cannot determine ground truth).  The counters are independent to
            support future extensions where partial attribution is needed.

        Args:
            model_id:        Identifier for the model (e.g. "qwen3.5-0.8b").
            constraint_type: Identifier for the constraint class (e.g. "range_check").
            was_fp:          True if this observation was confirmed as a false positive.
            was_tp:          True if this observation was confirmed as a true positive.

        Spec: REQ-LEARN-015-1
        """
        key = (model_id, constraint_type)
        if key not in self._stats:
            self._stats[key] = {"fp_count": 0, "tp_count": 0, "n_observations": 0}
        entry = self._stats[key]
        entry["n_observations"] += 1
        if was_fp:
            entry["fp_count"] += 1
        if was_tp:
            entry["tp_count"] += 1

    def should_disable(self, model_id: str, constraint_type: str) -> bool:
        """Return True when fp_rate strictly exceeds tp_rate with sufficient data.

        **Detailed explanation for engineers:**
            The disable decision requires two conditions:
            1. n_observations >= min_observations (enough data to trust the estimate)
            2. fp_count > tp_count (FP rate strictly exceeds TP rate)

            We use raw counts rather than rates for the comparison to avoid
            floating-point precision issues: fp_count > tp_count is equivalent
            to fp_rate > tp_rate when dividing by the same n_observations.

        Args:
            model_id:        Model to check.
            constraint_type: Constraint type to check.

        Returns:
            True if this constraint type should be disabled for this model.

        Spec: REQ-LEARN-015-2, SCENARIO-LEARN-025, SCENARIO-LEARN-026
        """
        key = (model_id, constraint_type)
        entry = self._stats.get(key)
        if entry is None:
            return False
        if entry["n_observations"] < self._min_observations:
            return False
        return entry["fp_count"] > entry["tp_count"]

    def get_active_constraint_types(self, model_id: str) -> frozenset[str]:
        """Return the frozenset of constraint types that are NOT disabled for model_id.

        **Detailed explanation for engineers:**
            Iterates over all (model_id, constraint_type) pairs that have been
            observed for this model and returns those that are not currently
            disabled.  If a model has never been observed, returns an empty
            frozenset (caller interprets this as "no constraint types are
            actively tracked yet" — the wrapping logic should then pass all
            violations through unchanged).

        Args:
            model_id: Model to look up.

        Returns:
            frozenset of constraint type strings that are active for this model.

        Spec: REQ-LEARN-015-3, SCENARIO-LEARN-026
        """
        active: set[str] = set()
        for (mid, ctype) in self._stats:
            if mid == model_id and not self.should_disable(model_id, ctype):
                active.add(ctype)
        return frozenset(active)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise tracker state to a JSON-compatible dict.

        **Detailed explanation for engineers:**
            The tuple keys ``(model_id, constraint_type)`` are encoded as
            "{model_id}|||{constraint_type}" strings so the result is valid
            JSON without nested objects.

        Returns:
            Dict with keys "min_observations" and "stats" (list of records).

        Spec: REQ-LEARN-015-5
        """
        stats_list = []
        for (model_id, constraint_type), counts in self._stats.items():
            stats_list.append({
                "model_id": model_id,
                "constraint_type": constraint_type,
                "fp_count": counts["fp_count"],
                "tp_count": counts["tp_count"],
                "n_observations": counts["n_observations"],
            })
        return {
            "min_observations": self._min_observations,
            "stats": stats_list,
        }

    def on_violation(self, event: "ViolationEvent") -> None:
        """Increment constraint_weight for event.constraint_type (REQ-FR11-002).

        **Why a constraint_weight dict rather than the existing fp/tp stats?**
            The fp/tp counters track historical accuracy rates across all models.
            The constraint_weight is a per-type scalar used at inference time to
            amplify energy contributions from frequently-violated constraint types.
            These are separate concerns with different update cadences.

        **Throttle contract (REQ-FR11-004):**
            Weight updates are throttled to at most 1 update per 10 queries.
            ``_violation_call_count`` is incremented on every call.  The update
            only fires when ``_violation_call_count % 10 == 0``.  This prevents
            weight thrash when many violations arrive in a short burst.

        **Cap contract (REQ-FR11-002):**
            Weights are capped at 2.0 to prevent a single frequent constraint type
            from dominating the energy signal by an unbounded factor.

        Args:
            event: ViolationEvent from FR11EventBus.

        Spec: REQ-FR11-002, REQ-FR11-004
        """
        import datetime

        if not hasattr(self, "_constraint_weights"):
            self._constraint_weights: dict[str, float] = {}
        if not hasattr(self, "_violation_call_count"):
            self._violation_call_count: int = 0
        if not hasattr(self, "_weight_update_counts"):
            self._weight_update_counts: dict[str, int] = {}
        if not hasattr(self, "_weight_last_updated"):
            self._weight_last_updated: dict[str, str] = {}

        self._violation_call_count += 1
        throttled = (self._violation_call_count % 10) != 0

        if not throttled:
            ctype = event.constraint_type
            current = self._constraint_weights.get(ctype, 1.0)
            new_weight = min(current + 0.01, 2.0)
            self._constraint_weights[ctype] = new_weight
            self._weight_update_counts[ctype] = self._weight_update_counts.get(ctype, 0) + 1
            self._weight_last_updated[ctype] = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            _log.debug(
                "FR11 weight update: constraint_type=%s new_weight=%.3f throttled=False",
                ctype,
                new_weight,
            )
        else:
            _log.debug(
                "FR11 weight update skipped: constraint_type=%s throttled=True call=%d",
                event.constraint_type,
                self._violation_call_count,
            )

    def get_weight_state(self) -> dict[str, WeightState]:
        """Return current per-constraint weight state for convergence auditing.

        **Detailed explanation for engineers:**
            This is the audit surface for REQ-FR11-007.  It surfaces the current
            learned weights and their update histories so an auditor (Exp 747) can
            verify that the relay is actually discriminating between constraint types
            rather than updating all weights uniformly.

            Only constraint types that have received at least one un-throttled update
            are returned (update_count > 0).  Types that were seen only in throttled
            calls have no weight change and are excluded — they would show weight=1.0
            (the initial default) which would falsely appear to have been "set" when
            they were never actually updated.

        Returns:
            Dict mapping constraint_type string to WeightState(weight, update_count,
            last_updated_at).  Empty dict when no un-throttled updates have occurred.

        Spec: REQ-FR11-007, REQ-FR11-007-1, REQ-FR11-007-2, REQ-FR11-007-3
        """
        if not hasattr(self, "_constraint_weights"):
            return {}
        if not hasattr(self, "_weight_update_counts"):
            return {}

        result: dict[str, WeightState] = {}
        for ctype, update_count in self._weight_update_counts.items():
            if update_count > 0:
                result[ctype] = WeightState(
                    weight=self._constraint_weights.get(ctype, 1.0),
                    update_count=update_count,
                    last_updated_at=self._weight_last_updated.get(ctype),
                )
        return result

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PerModelFPTracker:
        """Restore a PerModelFPTracker from a previously serialised dict.

        Args:
            payload: Dict produced by ``to_dict()``.

        Returns:
            Restored PerModelFPTracker instance.

        Spec: REQ-LEARN-015-5
        """
        tracker = cls(min_observations=int(payload.get("min_observations", 10)))
        for entry in payload.get("stats", []):
            key = (str(entry["model_id"]), str(entry["constraint_type"]))
            tracker._stats[key] = {
                "fp_count": int(entry.get("fp_count", 0)),
                "tp_count": int(entry.get("tp_count", 0)),
                "n_observations": int(entry.get("n_observations", 0)),
            }
        return tracker


# ---------------------------------------------------------------------------
# ModelAdaptiveThresholds
# ---------------------------------------------------------------------------


class ModelAdaptiveThresholds:
    """Wraps a ConstraintExtractor and filters violations by active constraint types.

    **Detailed explanation for engineers:**
        Calls the wrapped extractor's ``extract()`` method, then removes any
        violation whose ``constraint_type`` attribute appears in the disabled
        set for the given model_id.  If a violation has no ``constraint_type``
        attribute, it is treated as type "unknown" and is always kept (fail-safe).

        The filtering is applied lazily on each ``extract()`` call — the
        tracker's disable decisions are re-evaluated each time, so calibration
        improvements in the tracker take effect immediately without restarting.

    Args:
        extractor: The underlying constraint extractor to wrap.
        tracker:   PerModelFPTracker providing per-model disable decisions.

    Spec: REQ-LEARN-015-4
    """

    def __init__(
        self,
        extractor: ConstraintExtractor,
        tracker: PerModelFPTracker,
    ) -> None:
        self._extractor = extractor
        self._tracker = tracker

    def extract(
        self,
        question: str,
        response: str,
        model_id: str,
    ) -> list[object]:
        """Extract constraints and filter out disabled constraint types for model_id.

        **Detailed explanation for engineers:**
            Delegates to the underlying extractor then applies the model-specific
            filter.  The active set contains constraint types that have been
            observed but NOT disabled.  Constraint types that have NEVER been
            observed for this model are also kept (we do not suppress what we
            have not measured yet).

        Args:
            question: The original question / prompt text.
            response: The model response to verify.
            model_id: Identifier for the model that produced the response.

        Returns:
            Filtered list of constraint violations (disabled types removed).

        Spec: REQ-LEARN-015-4
        """
        all_violations = self._extractor.extract(question, response)

        # Check whether this model has any observations at all.
        # If not, nothing can be disabled — return everything unchanged.
        model_has_observations = any(
            mid == model_id for (mid, _) in self._tracker._stats
        )
        if not model_has_observations:
            return list(all_violations)

        filtered = []
        for violation in all_violations:
            ctype = getattr(violation, "constraint_type", "unknown")
            key = (model_id, ctype)
            # Keep if: (a) this type was never observed for this model (no data → no disable),
            # or (b) it was observed and is currently not disabled.
            never_observed = key not in self._tracker._stats
            if never_observed or not self._tracker.should_disable(model_id, ctype):
                filtered.append(violation)
        return filtered


# ---------------------------------------------------------------------------
# SelectiveConsolidation
# ---------------------------------------------------------------------------


class SelectiveConsolidation:
    """Selective CaseMemory consolidation: retain only high-contrast interactions.

    **Detailed explanation for engineers:**
        Implements the ATLAS (arXiv 2511.01093) insight that not all observed
        cases are equally informative for future retrieval.  High-contrast
        interactions are those where:

        - The EBM reports high violation energy BUT the model appeared confident
          (surprising: the model was wrong despite seeming sure of itself), OR
        - The EBM reports low violation energy BUT the model appeared uncertain
          (surprising: the EBM cleared a case the model was unsure about).

        In both cases the absolute difference between violation_energy and
        model_confidence_score exceeds the contrast_threshold.  Interactions
        where the EBM and model agreed (both confident and violation found,
        or both uncertain and no violation) carry less learning signal.

        Target consolidation ratio: 0.3–0.5 of all traces retained (matching
        the ATLAS paper's 40% figure on their benchmarks).

    Args:
        contrast_threshold: Minimum |violation_energy - model_confidence| required
                            to retain a trace.  Default 0.5.

    Spec: REQ-LEARN-016-1, REQ-LEARN-016-2
    """

    def __init__(self, contrast_threshold: float = 0.5) -> None:
        self._contrast_threshold = contrast_threshold

    def should_retain(
        self,
        verified_violation_energy: float,
        model_confidence_score: float,
    ) -> bool:
        """Return True when the trace is high-contrast and worth retaining.

        **Detailed explanation for engineers:**
            Contrast is defined as ``abs(verified_violation_energy -
            model_confidence_score)``.  When these two signals are far apart
            the interaction was surprising to at least one of them — that
            surprise is informative for future retrieval.  When they agree
            (both high or both low) the contrast is small and the trace is
            discarded.

            The comparison is STRICT (``>``) so a trace at exactly the
            threshold is NOT retained — the threshold is a floor, not a ceiling.

        Args:
            verified_violation_energy: Energy output from EBM verification
                                       (higher = more confident violation).
            model_confidence_score:    Model's own confidence in its response
                                       in [0, 1] (higher = more confident).

        Returns:
            True if the trace should be stored in CaseMemory.

        Spec: REQ-LEARN-016-1, SCENARIO-LEARN-027
        """
        contrast = abs(verified_violation_energy - model_confidence_score)
        return contrast > self._contrast_threshold

    def consolidation_ratio(self, total_traces: int, retained_traces: int) -> float:
        """Return the fraction of traces retained after selective consolidation.

        **Detailed explanation for engineers:**
            A simple utility for experiment reporting: divides retained by total
            and handles the zero-total edge case safely.  Target range per the
            ATLAS paper: 0.3–0.5.

        Args:
            total_traces:    Number of candidate traces evaluated.
            retained_traces: Number of traces actually stored.

        Returns:
            Fraction in [0, 1]. Returns 0.0 when total_traces == 0.

        Spec: REQ-LEARN-016-2
        """
        if total_traces == 0:
            return 0.0
        return retained_traces / total_traces


__all__ = [
    "ModelAdaptiveThresholds",
    "PerModelFPTracker",
    "SelectiveConsolidation",
    "WeightState",
]
