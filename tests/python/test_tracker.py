"""Tests for carnot.pipeline.tracker -- ConstraintTracker.

Covers online self-learning Tier 1: per-constraint-type precision/recall
tracking, persistence, merge, and integration with VerifyRepairPipeline.

Spec: REQ-LEARN-001, SCENARIO-LEARN-001
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from carnot.pipeline.tracker import ConstraintTracker
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# REQ-LEARN-001: record(), precision(), recall()
# ---------------------------------------------------------------------------


class TestRecordAndMetrics:
    """REQ-LEARN-001: Basic recording and metric computation."""

    def test_precision_perfect(self) -> None:
        """SCENARIO-LEARN-001: precision = 1.0 when every fire catches an error."""
        tracker = ConstraintTracker()
        tracker.record("arithmetic", fired=True, caught_error=True)
        tracker.record("arithmetic", fired=True, caught_error=True)
        assert tracker.precision("arithmetic") == 1.0

    def test_precision_zero(self) -> None:
        """REQ-LEARN-001: precision = 0.0 when fires but never catches errors."""
        tracker = ConstraintTracker()
        tracker.record("arithmetic", fired=True, caught_error=False)
        tracker.record("arithmetic", fired=True, caught_error=False)
        assert tracker.precision("arithmetic") == 0.0

    def test_precision_partial(self) -> None:
        """REQ-LEARN-001: precision = 0.5 when half the fires catch errors."""
        tracker = ConstraintTracker()
        tracker.record("logic", fired=True, caught_error=True)
        tracker.record("logic", fired=True, caught_error=False)
        assert tracker.precision("logic") == pytest.approx(0.5)

    def test_precision_unknown_type_is_zero(self) -> None:
        """REQ-LEARN-001: Unknown type returns 0.0 precision (conservative default)."""
        tracker = ConstraintTracker()
        assert tracker.precision("nonexistent") == 0.0

    def test_recall_perfect(self) -> None:
        """SCENARIO-LEARN-001: recall = 1.0 when every error batch is caught."""
        tracker = ConstraintTracker()
        # 3 calls, each with an error, each caught by this type.
        tracker.record("arithmetic", fired=True, caught_error=True, any_error_in_batch=True)
        tracker.record("arithmetic", fired=True, caught_error=True, any_error_in_batch=True)
        tracker.record("arithmetic", fired=True, caught_error=True, any_error_in_batch=True)
        assert tracker.recall("arithmetic") == 1.0

    def test_recall_partial(self) -> None:
        """REQ-LEARN-001: recall = 0.5 when type catches only half the error batches."""
        tracker = ConstraintTracker()
        # 2 calls with errors: type catches one, misses one.
        tracker.record("logic", fired=True, caught_error=True, any_error_in_batch=True)
        tracker.record("logic", fired=True, caught_error=False, any_error_in_batch=True)
        assert tracker.recall("logic") == pytest.approx(0.5)

    def test_recall_zero_no_errors(self) -> None:
        """REQ-LEARN-001: recall = 0.0 when no error batches ever recorded."""
        tracker = ConstraintTracker()
        tracker.record("code", fired=True, caught_error=False, any_error_in_batch=False)
        assert tracker.recall("code") == 0.0

    def test_recall_unknown_type_is_zero(self) -> None:
        """REQ-LEARN-001: Unknown type returns 0.0 recall."""
        tracker = ConstraintTracker()
        assert tracker.recall("nonexistent") == 0.0

    def test_multiple_types_independent(self) -> None:
        """REQ-LEARN-001: Different constraint types tracked independently."""
        tracker = ConstraintTracker()
        tracker.record("arithmetic", fired=True, caught_error=True)
        tracker.record("arithmetic", fired=True, caught_error=True)
        tracker.record("logic", fired=True, caught_error=False)
        tracker.record("logic", fired=True, caught_error=False)

        assert tracker.precision("arithmetic") == 1.0
        assert tracker.precision("logic") == 0.0

    def test_stats_returns_all_types(self) -> None:
        """REQ-LEARN-001: stats() returns dict with entries for every recorded type."""
        tracker = ConstraintTracker()
        tracker.record("arithmetic", fired=True, caught_error=True)
        tracker.record("code", fired=True, caught_error=False)
        s = tracker.stats()

        assert "arithmetic" in s
        assert "code" in s

    def test_stats_fields_present(self) -> None:
        """REQ-LEARN-001: Each stats entry has fired, caught, total_errors, precision, recall."""
        tracker = ConstraintTracker()
        tracker.record("arithmetic", fired=True, caught_error=True, any_error_in_batch=True)
        entry = tracker.stats()["arithmetic"]

        assert entry["fired"] == 1
        assert entry["caught"] == 1
        assert entry["total_errors"] == 1
        assert entry["precision"] == 1.0
        assert entry["recall"] == 1.0

    def test_stats_empty_tracker(self) -> None:
        """REQ-LEARN-001: Empty tracker returns empty stats dict."""
        tracker = ConstraintTracker()
        assert tracker.stats() == {}


# ---------------------------------------------------------------------------
# REQ-LEARN-001: save/load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """REQ-LEARN-001: Persistence — save to JSON and restore exact counts."""

    def test_round_trip(self) -> None:
        """SCENARIO-LEARN-001: save() + load() restores exact counter values."""
        tracker = ConstraintTracker()
        tracker.record("arithmetic", fired=True, caught_error=True, any_error_in_batch=True)
        tracker.record("arithmetic", fired=True, caught_error=False, any_error_in_batch=True)
        tracker.record("code", fired=True, caught_error=False, any_error_in_batch=False)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            tracker.save(path)
            restored = ConstraintTracker.load(path)
        finally:
            os.unlink(path)

        assert restored.precision("arithmetic") == pytest.approx(0.5)
        assert restored.recall("arithmetic") == pytest.approx(0.5)
        assert restored.precision("code") == 0.0
        assert restored.stats()["arithmetic"]["fired"] == 2
        assert restored.stats()["arithmetic"]["caught"] == 1

    def test_saved_file_is_valid_json(self) -> None:
        """REQ-LEARN-001: save() writes parseable JSON with version=1."""
        tracker = ConstraintTracker()
        tracker.record("arithmetic", fired=True, caught_error=True)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            tracker.save(path)
            with open(path) as fh:
                payload = json.load(fh)
        finally:
            os.unlink(path)

        assert payload["version"] == 1
        assert "arithmetic" in payload["stats"]

    def test_load_invalid_version_raises(self) -> None:
        """REQ-LEARN-001: load() raises ValueError on unsupported version."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump({"version": 99, "stats": {}}, f)
            path = f.name
        try:
            with pytest.raises(ValueError, match="version"):
                ConstraintTracker.load(path)
        finally:
            os.unlink(path)

    def test_load_empty_stats(self) -> None:
        """REQ-LEARN-001: load() handles file with zero recorded types."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump({"version": 1, "stats": {}}, f)
            path = f.name
        try:
            restored = ConstraintTracker.load(path)
        finally:
            os.unlink(path)

        assert restored.stats() == {}


# ---------------------------------------------------------------------------
# REQ-LEARN-001: merge()
# ---------------------------------------------------------------------------


class TestMerge:
    """REQ-LEARN-001: Merging two trackers combines their counters."""

    def test_merge_additive_counts(self) -> None:
        """SCENARIO-LEARN-001: merge() sums fired/caught/total_errors for shared types."""
        a = ConstraintTracker()
        a.record("arithmetic", fired=True, caught_error=True, any_error_in_batch=True)
        a.record("arithmetic", fired=True, caught_error=True, any_error_in_batch=True)

        b = ConstraintTracker()
        b.record("arithmetic", fired=True, caught_error=False, any_error_in_batch=True)
        b.record("arithmetic", fired=True, caught_error=False, any_error_in_batch=True)

        merged = a.merge(b)
        s = merged.stats()["arithmetic"]
        assert s["fired"] == 4
        assert s["caught"] == 2
        assert s["total_errors"] == 4
        assert merged.precision("arithmetic") == pytest.approx(0.5)

    def test_merge_disjoint_types(self) -> None:
        """REQ-LEARN-001: merge() preserves types that only appear in one tracker."""
        a = ConstraintTracker()
        a.record("arithmetic", fired=True, caught_error=True)

        b = ConstraintTracker()
        b.record("logic", fired=True, caught_error=False)

        merged = a.merge(b)
        assert "arithmetic" in merged.stats()
        assert "logic" in merged.stats()

    def test_merge_does_not_modify_inputs(self) -> None:
        """REQ-LEARN-001: merge() returns new tracker; inputs are unchanged."""
        a = ConstraintTracker()
        a.record("arithmetic", fired=True, caught_error=True)

        b = ConstraintTracker()
        b.record("arithmetic", fired=True, caught_error=False)

        _merged = a.merge(b)

        # a and b should still have their original single entry each.
        assert a.stats()["arithmetic"]["fired"] == 1
        assert b.stats()["arithmetic"]["fired"] == 1

    def test_merge_empty_trackers(self) -> None:
        """REQ-LEARN-001: Merging two empty trackers yields empty tracker."""
        a = ConstraintTracker()
        b = ConstraintTracker()
        merged = a.merge(b)
        assert merged.stats() == {}

    def test_merge_with_empty(self) -> None:
        """REQ-LEARN-001: Merging with an empty tracker is identity."""
        a = ConstraintTracker()
        a.record("arithmetic", fired=True, caught_error=True, any_error_in_batch=True)

        b = ConstraintTracker()
        merged = a.merge(b)
        assert merged.stats()["arithmetic"]["fired"] == 1
        assert merged.precision("arithmetic") == 1.0


# ---------------------------------------------------------------------------
# REQ-LEARN-001: Integration with VerifyRepairPipeline
# ---------------------------------------------------------------------------


class TestIntegrationWithPipeline:
    """REQ-LEARN-001, REQ-VERIFY-001: Tracker integrates with VerifyRepairPipeline."""

    def test_verify_without_tracker_is_backward_compatible(self) -> None:
        """REQ-VERIFY-001: verify() works unchanged when tracker=None (default)."""
        pipeline = VerifyRepairPipeline()
        result = pipeline.verify(
            question="What is 3 + 4?",
            response="3 + 4 = 7.",
            domain="arithmetic",
        )
        # Just check it runs without error and returns the right type.
        assert result.verified is True

    def test_verify_with_tracker_records_fired_type(self) -> None:
        """REQ-LEARN-001: verify() records the extracted constraint type in tracker."""
        pipeline = VerifyRepairPipeline()
        tracker = ConstraintTracker()

        pipeline.verify(
            question="What is 10 + 5?",
            response="10 + 5 = 15.",
            domain="arithmetic",
            tracker=tracker,
        )

        s = tracker.stats()
        # Arithmetic extractor fired.
        assert "arithmetic" in s
        assert s["arithmetic"]["fired"] == 1

    def test_verify_correct_response_no_caught_errors(self) -> None:
        """REQ-LEARN-001: Correct response records fired=1 but caught=0."""
        pipeline = VerifyRepairPipeline()
        tracker = ConstraintTracker()

        pipeline.verify(
            question="What is 10 + 5?",
            response="10 + 5 = 15.",
            domain="arithmetic",
            tracker=tracker,
        )

        s = tracker.stats()["arithmetic"]
        assert s["fired"] == 1
        assert s["caught"] == 0

    def test_verify_wrong_response_records_caught_error(self) -> None:
        """REQ-LEARN-001: Wrong response sets caught=1 in tracker for arithmetic."""
        pipeline = VerifyRepairPipeline()
        tracker = ConstraintTracker()

        pipeline.verify(
            question="What is 10 + 5?",
            response="10 + 5 = 99.",  # wrong
            domain="arithmetic",
            tracker=tracker,
        )

        s = tracker.stats()["arithmetic"]
        assert s["fired"] == 1
        assert s["caught"] == 1

    def test_tracker_accumulates_across_multiple_verify_calls(self) -> None:
        """REQ-LEARN-001: Repeated verify() calls accumulate into same tracker."""
        pipeline = VerifyRepairPipeline()
        tracker = ConstraintTracker()

        # Call verify twice -- correct then wrong.
        pipeline.verify(
            question="What is 2 + 2?",
            response="2 + 2 = 4.",
            domain="arithmetic",
            tracker=tracker,
        )
        pipeline.verify(
            question="What is 3 + 3?",
            response="3 + 3 = 99.",  # wrong
            domain="arithmetic",
            tracker=tracker,
        )

        s = tracker.stats()["arithmetic"]
        assert s["fired"] == 2
        assert s["caught"] == 1
        assert tracker.precision("arithmetic") == pytest.approx(0.5)

    def test_repr(self) -> None:
        """REQ-LEARN-001: __repr__ includes type count."""
        tracker = ConstraintTracker()
        assert "ConstraintTracker" in repr(tracker)
        tracker.record("arithmetic", fired=True, caught_error=False)
        assert "1" in repr(tracker)

    def test_tracker_does_not_double_count_same_type(self) -> None:
        """REQ-LEARN-001: Multiple constraints of same type in one verify = one record."""
        pipeline = VerifyRepairPipeline()
        tracker = ConstraintTracker()

        # Response with two arithmetic constraints, both correct.
        pipeline.verify(
            question="Arithmetic",
            response="2 + 2 = 4. 3 + 3 = 6.",
            domain="arithmetic",
            tracker=tracker,
        )

        # Should record once per verify call per type, not once per constraint.
        s = tracker.stats()["arithmetic"]
        assert s["fired"] == 1
