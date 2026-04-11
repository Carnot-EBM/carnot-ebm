"""Constraint performance tracker for online self-learning (Tier 1).

**Researcher summary:**
    Tracks which constraint types fire (are extracted) and which ones
    actually catch real errors. Accumulates statistics over multiple
    verification calls so the system can learn which constraints are
    worth running and which add noise without catching bugs.

**Detailed explanation for engineers:**
    This is the first building block of Continuous Self-Learning (Tier 1
    from research-program.md). Every time verify() runs, we know:
      1. Which constraints FIRED (were extracted from the text).
      2. Which constraints CAUGHT AN ERROR (fired AND violated).

    From these two signals we can compute:
      - Precision per type = caught / fired  (how often a firing constraint
        actually found a real problem -- low precision = noisy)
      - Recall per type = caught / total_errors  (how often the constraint
        type caught errors when errors existed -- low recall = missing bugs)

    Over many calls this lets the pipeline upweight high-precision
    constraint types and downweight noisy ones, without any matrix ops --
    just integer counters, runnable on plain CPU.

    The ConstraintTracker is:
      - Lightweight: all O(1) per-call with dict lookups.
      - Persistent: save/load to JSON for cross-session accumulation.
      - Mergeable: combine trackers from parallel workers or multiple runs.
      - Optional: VerifyRepairPipeline accepts tracker=None (default) for
        full backward compatibility.

Spec: REQ-LEARN-001, SCENARIO-LEARN-001 (Tier 1 Online Constraint Learning)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Per-type statistics accumulator
# ---------------------------------------------------------------------------


@dataclass
class _TypeStats:
    """Raw counters for one constraint type.

    **Detailed explanation for engineers:**
        Three counters are all we need per constraint type:
          - fired: how many times this type produced at least one constraint
            result during extraction (regardless of whether it caught an error).
          - caught: how many times this type found a real violation (fired AND
            the constraint was in the violations list).
          - total_errors: total number of verifications where at least one
            error was present, across all constraint types. Used for recall
            denominator -- "how often could this type have caught something?"

        We deliberately do NOT store a running average here. Precision and
        recall are computed on-demand from the raw counters so that merging
        two trackers is just counter addition.
    """

    fired: int = 0
    caught: int = 0
    total_errors: int = 0


# ---------------------------------------------------------------------------
# ConstraintTracker
# ---------------------------------------------------------------------------


class ConstraintTracker:
    """Online constraint performance tracker for self-learning pipelines.

    **Researcher summary:**
        Accumulates fired/caught counts per constraint type so the pipeline
        can learn which extractors are signal vs. noise over time.

    **Detailed explanation for engineers:**
        Usage pattern in a pipeline:

            tracker = ConstraintTracker()
            result = pipeline.verify(question, response, tracker=tracker)

        After the call, tracker has updated counts for every constraint type
        that was active during that verification. Query with:

            tracker.precision("arithmetic")  # 1.0 if always catches real errors
            tracker.recall("arithmetic")     # fraction of errors it found
            tracker.stats()                  # full summary dict

        Persist across sessions:

            tracker.save("/tmp/constraint_stats.json")
            tracker2 = ConstraintTracker.load("/tmp/constraint_stats.json")

        Merge parallel trackers (e.g., from multi-worker evaluation):

            combined = tracker_a.merge(tracker_b)

    Spec: REQ-LEARN-001, SCENARIO-LEARN-001
    """

    def __init__(self) -> None:
        # Dict mapping constraint_type string -> _TypeStats counters.
        # Grows lazily: first record() call for a new type creates the entry.
        self._data: dict[str, _TypeStats] = {}

    # -------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------

    def record(
        self,
        constraint_type: str,
        fired: bool,
        caught_error: bool,
        any_error_in_batch: bool = True,
    ) -> None:
        """Record one observation for a constraint type.

        **Detailed explanation for engineers:**
            Called once per constraint result after verification completes.
            The caller (VerifyRepairPipeline.verify) iterates over all
            extracted constraints and calls record() for each one.

            Parameters:
              - fired: True if this constraint type produced an extracted
                result (i.e., the extractor found something). Always True
                when called from the pipeline -- we only record constraints
                that were actually extracted.
              - caught_error: True if this specific constraint fired AND
                is in the violations list (i.e., found a real problem).
              - any_error_in_batch: True if the current verification round
                had ANY violation at all (used for recall denominator).
                Defaults to True to avoid penalising recall when not known.

            The ``total_errors`` counter increments by 1 per verification
            call when any_error_in_batch=True, NOT once per constraint.
            The pipeline passes this flag consistently for all constraints
            in the same verification call, so the total_errors count is
            correct even though record() is called multiple times per call.
            To achieve this, the pipeline should pass any_error_in_batch
            only as True on the FIRST record call for each type per
            verification (the helper _update_tracker handles this correctly).

        Args:
            constraint_type: Category string, e.g. "arithmetic", "type_check".
            fired: True if this constraint result was extracted.
            caught_error: True if this constraint detected a violation.
            any_error_in_batch: True if the full verification found errors.
        """
        if constraint_type not in self._data:
            self._data[constraint_type] = _TypeStats()

        stats = self._data[constraint_type]
        if fired:
            stats.fired += 1
        if caught_error:
            stats.caught += 1
        if any_error_in_batch:
            stats.total_errors += 1

    # -------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------

    def precision(self, constraint_type: str) -> float:
        """Precision for one constraint type: caught / fired.

        **Detailed explanation for engineers:**
            Answers: "When this extractor fires, how often does it catch
            a real error?" High precision (near 1.0) = signal. Low
            precision (near 0.0) = noisy extractor firing on false alarms.

            Returns 0.0 if the constraint type has never fired. This is a
            conservative default that treats unknown extractors as noisy
            until proven otherwise.

        Args:
            constraint_type: The type tag to query.

        Returns:
            Float in [0.0, 1.0]. 0.0 if never fired.
        """
        stats = self._data.get(constraint_type)
        if stats is None or stats.fired == 0:
            return 0.0
        return stats.caught / stats.fired

    def recall(self, constraint_type: str) -> float:
        """Recall for one constraint type: caught / total_errors.

        **Detailed explanation for engineers:**
            Answers: "When there IS an error in the verification batch,
            how often does this extractor catch it?" High recall (near 1.0)
            = extractor almost never misses a real problem. Low recall =
            the extractor misses many real errors.

            Returns 0.0 if no errors have been recorded for this type.
            This is conservative -- if we've never seen errors for a type,
            we can't claim it has good recall.

        Args:
            constraint_type: The type tag to query.

        Returns:
            Float in [0.0, 1.0]. 0.0 if total_errors is zero.
        """
        stats = self._data.get(constraint_type)
        if stats is None or stats.total_errors == 0:
            return 0.0
        return stats.caught / stats.total_errors

    def stats(self) -> dict[str, dict[str, Any]]:
        """Return full per-type statistics as a plain dict.

        **Detailed explanation for engineers:**
            Returns a dict keyed by constraint_type. Each value has:
              - "fired": total times this type was extracted
              - "caught": total times it detected a real error
              - "total_errors": total verification calls with any error
              - "precision": caught / fired (0.0 if never fired)
              - "recall": caught / total_errors (0.0 if no errors recorded)

            Useful for logging, analysis, and feeding into downstream
            learning algorithms (Tier 2 Trace2Skill, Tier 3 JEPA).

        Returns:
            Dict of {constraint_type: {metric: value, ...}}.
        """
        result: dict[str, dict[str, Any]] = {}
        for ctype, s in self._data.items():
            result[ctype] = {
                "fired": s.fired,
                "caught": s.caught,
                "total_errors": s.total_errors,
                "precision": self.precision(ctype),
                "recall": self.recall(ctype),
            }
        return result

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist tracker statistics to a JSON file.

        **Detailed explanation for engineers:**
            Serialises the raw counters (fired, caught, total_errors) for
            each constraint type. Precision/recall are NOT stored because
            they are derived and can be recomputed exactly from counters.
            This keeps the file format stable as we add new derived metrics.

        Args:
            path: Filesystem path to write. Will overwrite if exists.

        Raises:
            OSError: If the path is not writable.
        """
        payload: dict[str, Any] = {
            "version": 1,
            "stats": {
                ctype: {
                    "fired": s.fired,
                    "caught": s.caught,
                    "total_errors": s.total_errors,
                }
                for ctype, s in self._data.items()
            },
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "ConstraintTracker":
        """Restore a ConstraintTracker from a JSON file written by save().

        **Detailed explanation for engineers:**
            Reads the file, validates the version field, and reconstructs
            _TypeStats for each stored constraint type. Returns a fresh
            ConstraintTracker instance populated with the stored counters.

        Args:
            path: Path to a JSON file previously written by save().

        Returns:
            A new ConstraintTracker with counters loaded from the file.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If the file format is invalid or version is unsupported.
        """
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError(
                f"Unsupported tracker file format (expected version=1): {path}"
            )

        tracker = cls()
        for ctype, raw in payload.get("stats", {}).items():
            stats = _TypeStats(
                fired=int(raw.get("fired", 0)),
                caught=int(raw.get("caught", 0)),
                total_errors=int(raw.get("total_errors", 0)),
            )
            tracker._data[ctype] = stats
        return tracker

    # -------------------------------------------------------------------
    # Merging
    # -------------------------------------------------------------------

    def merge(self, other: "ConstraintTracker") -> "ConstraintTracker":
        """Combine statistics from two trackers into a new tracker.

        **Detailed explanation for engineers:**
            Merging is simple counter addition: for each constraint type
            present in either tracker, add fired + fired, caught + caught,
            total_errors + total_errors. The result is equivalent to what
            you would get from a single tracker that had seen all events
            from both input trackers.

            This property makes it safe to run trackers in parallel (e.g.,
            one per worker process) and merge them at the end, or to
            accumulate statistics across daily batches and merge nightly.

            Neither input tracker is modified. Returns a brand-new
            ConstraintTracker with the merged counters.

        Args:
            other: Another ConstraintTracker whose counters to add in.

        Returns:
            New ConstraintTracker containing the combined counts.
        """
        merged = ConstraintTracker()
        # Collect all known types from both trackers.
        all_types = set(self._data.keys()) | set(other._data.keys())
        for ctype in all_types:
            self_s = self._data.get(ctype, _TypeStats())
            other_s = other._data.get(ctype, _TypeStats())
            merged._data[ctype] = _TypeStats(
                fired=self_s.fired + other_s.fired,
                caught=self_s.caught + other_s.caught,
                total_errors=self_s.total_errors + other_s.total_errors,
            )
        return merged

    def __repr__(self) -> str:
        n = len(self._data)
        return f"ConstraintTracker(types={n})"
