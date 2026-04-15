"""Tests for carnot.pipeline.adaptive_thresholds.

Model-adaptive constraint thresholds and selective CaseMemory consolidation:
- PerModelFPTracker learns per-model, per-constraint-type FP/TP rates and
  disables noisy constraint types when FP rate exceeds TP rate.
- ModelAdaptiveThresholds wraps a ConstraintExtractor and filters violations
  for disabled constraint types.
- SelectiveConsolidation implements the ATLAS selective memory strategy:
  retain only high-contrast traces (where verification energy disagrees
  with model confidence direction).
- CaseMemory.add_trace_selective only stores a CaseRecord when the contrast
  between violation energy and model confidence exceeds min_contrast.

Spec: REQ-LEARN-015, REQ-LEARN-016,
      SCENARIO-LEARN-025, SCENARIO-LEARN-026,
      SCENARIO-LEARN-027, SCENARIO-LEARN-028
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from carnot.pipeline.adaptive_thresholds import (
    ModelAdaptiveThresholds,
    PerModelFPTracker,
    SelectiveConsolidation,
)
from carnot.pipeline.case_memory import CaseMemory, CaseRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(model_name: str = "test_model") -> CaseRecord:
    """Build a minimal CaseRecord for testing add_trace_selective."""
    return CaseRecord.normalize(
        benchmark="gsm8k",
        benchmark_slice="arithmetic",
        model_name=model_name,
        case_id="case_001",
        violation_types=("arithmetic:wrong_result",),
        prompt_text="What is 2 + 2?",
        baseline_success=False,
        repair_success=True,
        confidence=0.9,
    )


def _make_violation(constraint_type: str, description: str | None = None):
    """Create a minimal mock constraint result with a constraint_type attribute."""
    v = MagicMock()
    v.constraint_type = constraint_type
    v.description = description or constraint_type
    v.metadata = {}
    return v


# ---------------------------------------------------------------------------
# PerModelFPTracker — REQ-LEARN-015
# ---------------------------------------------------------------------------


class TestPerModelFPTrackerUpdate:
    """REQ-LEARN-015-1: update() increments fp/tp counts correctly."""

    def test_update_fp_increments_fp_count(self) -> None:
        """A false-positive observation increments fp_count."""
        tracker = PerModelFPTracker(min_observations=1)
        tracker.update("model_a", "range_check", was_fp=True, was_tp=False)
        stats = tracker._stats[("model_a", "range_check")]
        assert stats["fp_count"] == 1
        assert stats["tp_count"] == 0

    def test_update_tp_increments_tp_count(self) -> None:
        """A true-positive observation increments tp_count."""
        tracker = PerModelFPTracker(min_observations=1)
        tracker.update("model_a", "arithmetic", was_fp=False, was_tp=True)
        stats = tracker._stats[("model_a", "arithmetic")]
        assert stats["tp_count"] == 1
        assert stats["fp_count"] == 0

    def test_update_neither_increments_n_observations(self) -> None:
        """n_observations increases on every update, even when both flags are False."""
        tracker = PerModelFPTracker(min_observations=1)
        tracker.update("model_a", "logic", was_fp=False, was_tp=False)
        stats = tracker._stats[("model_a", "logic")]
        assert stats["n_observations"] == 1

    def test_update_accumulates_over_multiple_calls(self) -> None:
        """Multiple updates accumulate counts correctly."""
        tracker = PerModelFPTracker(min_observations=1)
        for _ in range(5):
            tracker.update("m", "ct", was_fp=True, was_tp=False)
        for _ in range(3):
            tracker.update("m", "ct", was_fp=False, was_tp=True)
        stats = tracker._stats[("m", "ct")]
        assert stats["fp_count"] == 5
        assert stats["tp_count"] == 3
        assert stats["n_observations"] == 8

    def test_different_pairs_are_independent(self) -> None:
        """Updates to different (model_id, constraint_type) pairs do not cross-contaminate."""
        tracker = PerModelFPTracker(min_observations=1)
        tracker.update("model_a", "range_check", was_fp=True, was_tp=False)
        tracker.update("model_b", "range_check", was_fp=False, was_tp=True)
        assert tracker._stats[("model_a", "range_check")]["fp_count"] == 1
        assert tracker._stats[("model_b", "range_check")]["fp_count"] == 0


class TestPerModelFPTrackerShouldDisable:
    """REQ-LEARN-015-2: should_disable() returns True when fp_rate > tp_rate
    and n_observations >= min_observations.

    SCENARIO-LEARN-025: High-FP constraint type gets disabled.
    SCENARIO-LEARN-026: Low-FP constraint type stays active.
    """

    def test_disable_when_fp_rate_exceeds_tp_rate(self) -> None:
        """SCENARIO-LEARN-025: fp_rate > tp_rate → should_disable returns True."""
        tracker = PerModelFPTracker(min_observations=10)
        # 9 FP, 3 TP out of 12 total: fp_rate=0.75, tp_rate=0.25
        for _ in range(9):
            tracker.update("qwen3.5-0.8b", "range_check", was_fp=True, was_tp=False)
        for _ in range(3):
            tracker.update("qwen3.5-0.8b", "range_check", was_fp=False, was_tp=True)
        assert tracker.should_disable("qwen3.5-0.8b", "range_check") is True

    def test_no_disable_when_tp_rate_exceeds_fp_rate(self) -> None:
        """SCENARIO-LEARN-026: tp_rate > fp_rate → should_disable returns False."""
        tracker = PerModelFPTracker(min_observations=10)
        # 3 FP, 9 TP out of 12 total: fp_rate=0.25, tp_rate=0.75
        for _ in range(3):
            tracker.update("qwen3.5-0.8b", "arithmetic", was_fp=True, was_tp=False)
        for _ in range(9):
            tracker.update("qwen3.5-0.8b", "arithmetic", was_fp=False, was_tp=True)
        assert tracker.should_disable("qwen3.5-0.8b", "arithmetic") is False

    def test_not_disabled_below_min_observations(self) -> None:
        """Below min_observations threshold: never disable regardless of rates."""
        tracker = PerModelFPTracker(min_observations=10)
        # Only 5 observations — not enough data
        for _ in range(5):
            tracker.update("model", "range_check", was_fp=True, was_tp=False)
        assert tracker.should_disable("model", "range_check") is False

    def test_equal_rates_not_disabled(self) -> None:
        """Equal fp_rate == tp_rate → NOT disabled (fp must STRICTLY exceed tp)."""
        tracker = PerModelFPTracker(min_observations=10)
        for _ in range(6):
            tracker.update("model", "ct", was_fp=True, was_tp=False)
        for _ in range(6):
            tracker.update("model", "ct", was_fp=False, was_tp=True)
        assert tracker.should_disable("model", "ct") is False

    def test_unknown_pair_not_disabled(self) -> None:
        """A constraint type with no observations is never disabled."""
        tracker = PerModelFPTracker(min_observations=10)
        assert tracker.should_disable("unknown_model", "unknown_type") is False

    def test_exactly_at_min_observations_applies_rule(self) -> None:
        """At exactly min_observations, the disable rule is applied."""
        tracker = PerModelFPTracker(min_observations=5)
        for _ in range(4):
            tracker.update("m", "ct", was_fp=True, was_tp=False)
        for _ in range(1):
            tracker.update("m", "ct", was_fp=False, was_tp=True)
        # 5 observations exactly: fp=4, tp=1 → should disable
        assert tracker.should_disable("m", "ct") is True


class TestPerModelFPTrackerGetActiveConstraintTypes:
    """REQ-LEARN-015-3: get_active_constraint_types returns frozenset excluding disabled."""

    def test_all_active_before_threshold(self) -> None:
        """Before enough observations, all constraint types are active."""
        tracker = PerModelFPTracker(min_observations=10)
        tracker.update("model", "type_a", was_fp=True, was_tp=False)
        active = tracker.get_active_constraint_types("model")
        # Not yet disabled (only 1 observation vs min 10)
        assert "type_a" in active

    def test_disabled_type_excluded(self) -> None:
        """After disabling, the constraint type is not in active set."""
        tracker = PerModelFPTracker(min_observations=5)
        for _ in range(4):
            tracker.update("model", "bad_type", was_fp=True, was_tp=False)
        for _ in range(1):
            tracker.update("model", "bad_type", was_fp=False, was_tp=True)
        active = tracker.get_active_constraint_types("model")
        assert "bad_type" not in active

    def test_get_active_returns_frozenset(self) -> None:
        """Return type must be frozenset."""
        tracker = PerModelFPTracker(min_observations=10)
        result = tracker.get_active_constraint_types("model")
        assert isinstance(result, frozenset)

    def test_model_with_no_observations_returns_empty(self) -> None:
        """Model with no tracked observations returns empty frozenset."""
        tracker = PerModelFPTracker(min_observations=10)
        result = tracker.get_active_constraint_types("ghost_model")
        assert result == frozenset()

    def test_multiple_types_only_bad_one_disabled(self) -> None:
        """Only the high-FP type is excluded; good type remains active."""
        tracker = PerModelFPTracker(min_observations=5)
        # bad_type: 5 FP, 0 TP
        for _ in range(5):
            tracker.update("model", "bad_type", was_fp=True, was_tp=False)
        # good_type: 0 FP, 5 TP
        for _ in range(5):
            tracker.update("model", "good_type", was_fp=False, was_tp=True)
        active = tracker.get_active_constraint_types("model")
        assert "bad_type" not in active
        assert "good_type" in active


class TestPerModelFPTrackerPersistence:
    """REQ-LEARN-015-5: to_dict / from_dict round-trip."""

    def test_round_trip_preserves_counts(self) -> None:
        """Serialise and deserialise preserves all internal counts."""
        tracker = PerModelFPTracker(min_observations=10)
        for _ in range(7):
            tracker.update("model_a", "range_check", was_fp=True, was_tp=False)
        for _ in range(3):
            tracker.update("model_a", "range_check", was_fp=False, was_tp=True)
        restored = PerModelFPTracker.from_dict(tracker.to_dict())
        assert restored._stats == tracker._stats
        assert restored._min_observations == tracker._min_observations

    def test_round_trip_preserves_min_observations(self) -> None:
        """min_observations survives serialisation."""
        tracker = PerModelFPTracker(min_observations=20)
        restored = PerModelFPTracker.from_dict(tracker.to_dict())
        assert restored._min_observations == 20

    def test_to_dict_is_json_serialisable(self) -> None:
        """to_dict output must be directly JSON-serialisable (no exotic types)."""
        import json
        tracker = PerModelFPTracker(min_observations=5)
        tracker.update("m", "ct", was_fp=True, was_tp=False)
        # Should not raise
        json.dumps(tracker.to_dict())


# ---------------------------------------------------------------------------
# ModelAdaptiveThresholds — REQ-LEARN-015-4
# ---------------------------------------------------------------------------


class TestModelAdaptiveThresholds:
    """REQ-LEARN-015-4: extract() filters violations using active constraint types."""

    def _make_extractor(self, violations: list) -> MagicMock:
        extractor = MagicMock()
        extractor.extract.return_value = violations
        return extractor

    def test_passes_through_active_violations(self) -> None:
        """Violations whose constraint_type is active are returned unchanged."""
        tracker = PerModelFPTracker(min_observations=10)
        extractor = self._make_extractor([_make_violation("arithmetic")])
        adaptive = ModelAdaptiveThresholds(extractor, tracker)
        # "arithmetic" has no observations → active (not disabled)
        result = adaptive.extract("What is 2+2?", "4", "qwen3.5-0.8b")
        assert len(result) == 1

    def test_filters_disabled_constraint_type(self) -> None:
        """Violations whose constraint_type is disabled are removed from output."""
        tracker = PerModelFPTracker(min_observations=5)
        for _ in range(5):
            tracker.update("qwen3.5-0.8b", "range_check", was_fp=True, was_tp=False)
        extractor = self._make_extractor([
            _make_violation("range_check"),
            _make_violation("arithmetic"),
        ])
        adaptive = ModelAdaptiveThresholds(extractor, tracker)
        result = adaptive.extract("q", "r", "qwen3.5-0.8b")
        types = {v.constraint_type for v in result}
        assert "range_check" not in types
        assert "arithmetic" in types

    def test_empty_violations_returns_empty(self) -> None:
        """Empty extractor output → empty filtered output."""
        tracker = PerModelFPTracker(min_observations=10)
        extractor = self._make_extractor([])
        adaptive = ModelAdaptiveThresholds(extractor, tracker)
        assert adaptive.extract("q", "r", "model") == []

    def test_model_without_observations_keeps_all(self) -> None:
        """Model with no tracked data → no constraint types disabled → all pass."""
        tracker = PerModelFPTracker(min_observations=10)
        violations = [_make_violation("type_a"), _make_violation("type_b")]
        extractor = self._make_extractor(violations)
        adaptive = ModelAdaptiveThresholds(extractor, tracker)
        result = adaptive.extract("q", "r", "brand_new_model")
        assert len(result) == 2

    def test_calls_underlying_extractor_with_correct_args(self) -> None:
        """extract() forwards question and response to the underlying extractor."""
        tracker = PerModelFPTracker(min_observations=10)
        extractor = self._make_extractor([])
        adaptive = ModelAdaptiveThresholds(extractor, tracker)
        adaptive.extract("my_question", "my_response", "model_x")
        extractor.extract.assert_called_once_with("my_question", "my_response")


# ---------------------------------------------------------------------------
# SelectiveConsolidation — REQ-LEARN-016
# ---------------------------------------------------------------------------


class TestSelectiveConsolidationShouldRetain:
    """REQ-LEARN-016-1: should_retain returns True for high-contrast pairs.

    SCENARIO-LEARN-027: High-contrast trace is retained.
    """

    def test_high_energy_low_confidence_retained(self) -> None:
        """SCENARIO-LEARN-027: energy=0.9, confidence=0.1 → contrast=0.8 > 0.5."""
        sc = SelectiveConsolidation()
        assert sc.should_retain(0.9, 0.1) is True

    def test_low_energy_high_confidence_retained(self) -> None:
        """Low energy + high confidence is also surprising (expected violation but none)."""
        sc = SelectiveConsolidation()
        assert sc.should_retain(0.1, 0.9) is True

    def test_matching_signals_not_retained(self) -> None:
        """High energy + high confidence (agreed) → low contrast → not retained."""
        sc = SelectiveConsolidation()
        # contrast = abs(0.9 - 0.9) = 0.0 < 0.5
        assert sc.should_retain(0.9, 0.9) is False

    def test_equal_mid_values_not_retained(self) -> None:
        """0.5 and 0.5 → contrast=0.0 → not retained."""
        sc = SelectiveConsolidation()
        assert sc.should_retain(0.5, 0.5) is False

    def test_exact_threshold_boundary(self) -> None:
        """Contrast exactly at default threshold 0.5 → NOT retained (strictly greater)."""
        sc = SelectiveConsolidation()
        # abs(1.0 - 0.5) = 0.5 → not strictly > 0.5
        assert sc.should_retain(1.0, 0.5) is False

    def test_above_threshold_retained(self) -> None:
        """Contrast 0.51 just above threshold → retained."""
        sc = SelectiveConsolidation()
        assert sc.should_retain(1.0, 0.49) is True

    def test_custom_threshold(self) -> None:
        """Custom threshold parameter controls retain decision."""
        sc = SelectiveConsolidation(contrast_threshold=0.2)
        # contrast = 0.4 → > 0.2 → retained
        assert sc.should_retain(0.7, 0.3) is True

    def test_custom_threshold_blocks_below(self) -> None:
        """Custom threshold blocks traces below it."""
        sc = SelectiveConsolidation(contrast_threshold=0.8)
        # contrast = 0.6 → < 0.8 → not retained
        assert sc.should_retain(0.9, 0.3) is False


class TestSelectiveConsolidationRatio:
    """REQ-LEARN-016-2: consolidation_ratio returns fraction retained."""

    def test_ratio_all_retained(self) -> None:
        """When all 10 traces are retained, ratio is 1.0."""
        sc = SelectiveConsolidation()
        assert sc.consolidation_ratio(10, 10) == pytest.approx(1.0)

    def test_ratio_none_retained(self) -> None:
        """When 0 out of 10 are retained, ratio is 0.0."""
        sc = SelectiveConsolidation()
        assert sc.consolidation_ratio(10, 0) == pytest.approx(0.0)

    def test_ratio_partial(self) -> None:
        """4 out of 10 traces retained → ratio = 0.4."""
        sc = SelectiveConsolidation()
        assert sc.consolidation_ratio(10, 4) == pytest.approx(0.4)

    def test_ratio_zero_total_returns_zero(self) -> None:
        """Zero total traces → ratio = 0.0 (no division by zero)."""
        sc = SelectiveConsolidation()
        assert sc.consolidation_ratio(0, 0) == pytest.approx(0.0)

    def test_target_range_achievable(self) -> None:
        """Verify that target 0.3–0.5 is achievable with typical high-contrast filtering."""
        sc = SelectiveConsolidation()
        ratio = sc.consolidation_ratio(100, 40)
        assert 0.3 <= ratio <= 0.5


# ---------------------------------------------------------------------------
# CaseMemory.add_trace_selective — REQ-LEARN-016-3, REQ-LEARN-016-4
# SCENARIO-LEARN-028
# ---------------------------------------------------------------------------


class TestCaseMemoryAddTraceSelective:
    """REQ-LEARN-016-3 / REQ-LEARN-016-4: add_trace_selective filters by contrast.

    SCENARIO-LEARN-028: Low-contrast trace is discarded.
    """

    def test_high_contrast_stored_returns_true(self) -> None:
        """High contrast trace is stored and True is returned."""
        memory = CaseMemory()
        record = _make_record()
        # contrast = abs(0.9 - 0.05) = 0.85 > 0.5
        result = memory.add_trace_selective(record, violation_energy=0.9,
                                            model_confidence=0.05, min_contrast=0.5)
        assert result is True
        assert len(memory) == 1

    def test_low_contrast_discarded_returns_false(self) -> None:
        """SCENARIO-LEARN-028: Low contrast → not stored → returns False."""
        memory = CaseMemory()
        record = _make_record()
        # contrast = abs(0.6 - 0.55) = 0.05 < 0.5
        result = memory.add_trace_selective(record, violation_energy=0.6,
                                            model_confidence=0.55, min_contrast=0.5)
        assert result is False
        assert len(memory) == 0

    def test_exact_threshold_not_stored(self) -> None:
        """Contrast exactly at min_contrast → not stored (strictly greater required)."""
        memory = CaseMemory()
        record = _make_record()
        # contrast = abs(1.0 - 0.5) = 0.5 → NOT strictly > 0.5
        result = memory.add_trace_selective(record, violation_energy=1.0,
                                            model_confidence=0.5, min_contrast=0.5)
        assert result is False
        assert len(memory) == 0

    def test_existing_record_method_unaffected(self) -> None:
        """add_trace_selective is purely additive — original record() method still works."""
        memory = CaseMemory()
        record = _make_record()
        memory.record(record)
        assert len(memory) == 1
        # Selective add should also add when high-contrast
        record2 = _make_record(model_name="other_model")
        memory.add_trace_selective(record2, violation_energy=0.9,
                                   model_confidence=0.1, min_contrast=0.5)
        assert len(memory) == 2

    def test_custom_min_contrast(self) -> None:
        """Custom min_contrast parameter changes acceptance threshold."""
        memory = CaseMemory()
        record = _make_record()
        # contrast = 0.3, min_contrast = 0.2 → stored
        result = memory.add_trace_selective(record, violation_energy=0.7,
                                            model_confidence=0.4, min_contrast=0.2)
        assert result is True
        assert len(memory) == 1

    def test_multiple_high_contrast_traces_all_stored(self) -> None:
        """All high-contrast traces from different models are stored."""
        memory = CaseMemory()
        for i in range(3):
            rec = _make_record(model_name=f"model_{i}")
            memory.add_trace_selective(rec, violation_energy=0.95,
                                       model_confidence=0.05, min_contrast=0.5)
        assert len(memory) == 3
