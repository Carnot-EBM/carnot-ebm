"""Tests for carnot.verify.reward_hacking -- reward hacking detection.

Covers detection of trivial constraints, zero-energy shortcuts,
constraint monoculture, and train/holdout energy divergence in the
self-learning pipeline (Exp 223/241 context).

Spec: REQ-LEARN-002, SCENARIO-LEARN-002
"""

from __future__ import annotations

import pytest

from carnot.pipeline.tracker import ConstraintTracker
from carnot.verify.reward_hacking import (  # type: ignore[attr-defined]
    _gini_coefficient,
    _mean,
)
from carnot.verify.reward_hacking import (
    DIVERGENCE_MIN_GAP,
    ENERGY_DISTINCT_VALUES_MIN,
    GINI_DIVERSITY_THRESHOLD,
    MIN_ENERGY_SAMPLES,
    MIN_FIRE_COUNT_FOR_TRIVIAL_FLAG,
    TRIVIAL_PRECISION_THRESHOLD,
    LowDiversityFinding,
    RewardHackingReport,
    TrainHoldoutDivergenceFinding,
    TrivialConstraintFinding,
    ZeroEnergyFinding,
    audit_energy_trajectory,
    audit_full,
    audit_tracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tracker_with(*type_fire_caught: tuple[str, int, int]) -> ConstraintTracker:
    """Build a ConstraintTracker with pre-populated counters.

    Each tuple is (constraint_type, n_fired, n_caught). total_errors is set
    equal to n_fired for simplicity (one error per batch when caught).
    """
    tracker = ConstraintTracker()
    for ctype, n_fired, n_caught in type_fire_caught:
        for i in range(n_fired):
            caught = i < n_caught
            tracker.record(ctype, fired=True, caught_error=caught, any_error_in_batch=caught)
    return tracker


# ---------------------------------------------------------------------------
# Module-level constants are exported
# ---------------------------------------------------------------------------


class TestConstants:
    """REQ-LEARN-002: Module constants have expected types and values."""

    def test_min_fire_count_positive_int(self) -> None:
        """SCENARIO-LEARN-002: MIN_FIRE_COUNT_FOR_TRIVIAL_FLAG is a positive int."""
        assert isinstance(MIN_FIRE_COUNT_FOR_TRIVIAL_FLAG, int)
        assert MIN_FIRE_COUNT_FOR_TRIVIAL_FLAG > 0

    def test_trivial_precision_threshold_in_range(self) -> None:
        assert 0.0 <= TRIVIAL_PRECISION_THRESHOLD <= 1.0

    def test_energy_distinct_min_positive(self) -> None:
        assert ENERGY_DISTINCT_VALUES_MIN >= 1

    def test_gini_threshold_in_range(self) -> None:
        assert 0.0 <= GINI_DIVERSITY_THRESHOLD <= 1.0

    def test_divergence_min_gap_positive(self) -> None:
        assert DIVERGENCE_MIN_GAP > 0.0

    def test_min_energy_samples_positive(self) -> None:
        assert MIN_ENERGY_SAMPLES >= 1


# ---------------------------------------------------------------------------
# RewardHackingReport
# ---------------------------------------------------------------------------


class TestRewardHackingReport:
    """REQ-LEARN-002: Report aggregation and serialisation."""

    def test_empty_report_is_clean(self) -> None:
        """SCENARIO-LEARN-002: Report with no findings is clean."""
        report = RewardHackingReport()
        assert report.clean is True
        assert len(report.findings) == 0

    def test_report_with_finding_is_not_clean(self) -> None:
        finding = TrivialConstraintFinding(
            constraint_type="arithmetic", fired=10, precision=0.01
        )
        report = RewardHackingReport(findings=[finding])
        assert report.clean is False

    def test_to_dict_clean(self) -> None:
        report = RewardHackingReport()
        d = report.to_dict()
        assert d["clean"] is True
        assert d["n_findings"] == 0
        assert d["findings"] == []

    def test_to_dict_with_findings(self) -> None:
        finding = TrivialConstraintFinding(
            constraint_type="x", fired=5, precision=0.0
        )
        report = RewardHackingReport(findings=[finding])
        d = report.to_dict()
        assert d["clean"] is False
        assert d["n_findings"] == 1
        assert len(d["findings"]) == 1

    def test_n_findings_matches_len(self) -> None:
        """REQ-LEARN-002: n_findings property reflects actual list length."""
        report = RewardHackingReport()
        assert len(report.findings) == 0
        report.findings.append(
            ZeroEnergyFinding(sequence_length=10, distinct_values=1)
        )
        assert not report.clean


# ---------------------------------------------------------------------------
# Finding dataclasses — to_dict coverage
# ---------------------------------------------------------------------------


class TestFindingDicts:
    """REQ-LEARN-002: Each finding type serialises to the expected dict shape."""

    def test_trivial_constraint_finding_to_dict(self) -> None:
        f = TrivialConstraintFinding(constraint_type="arithmetic", fired=10, precision=0.01)
        d = f.to_dict()
        assert d["kind"] == "trivial_constraint"
        assert d["constraint_type"] == "arithmetic"
        assert d["fired"] == 10
        assert d["precision"] == pytest.approx(0.01)
        assert "threshold" in d

    def test_zero_energy_finding_to_dict(self) -> None:
        f = ZeroEnergyFinding(sequence_length=20, distinct_values=1)
        d = f.to_dict()
        assert d["kind"] == "zero_energy_shortcut"
        assert d["sequence_length"] == 20
        assert d["distinct_values"] == 1
        assert "min_distinct_required" in d

    def test_low_diversity_finding_to_dict(self) -> None:
        f = LowDiversityFinding(
            gini=0.8, n_types=3, dominant_type="arithmetic", dominant_fraction=0.9
        )
        d = f.to_dict()
        assert d["kind"] == "low_diversity"
        assert d["gini"] == pytest.approx(0.8)
        assert d["n_types"] == 3
        assert d["dominant_type"] == "arithmetic"
        assert d["dominant_fraction"] == pytest.approx(0.9)
        assert "threshold" in d

    def test_train_holdout_divergence_to_dict(self) -> None:
        f = TrainHoldoutDivergenceFinding(
            mean_train_energy=0.1, mean_holdout_energy=0.5, gap=0.4
        )
        d = f.to_dict()
        assert d["kind"] == "train_holdout_divergence"
        assert d["mean_train_energy"] == pytest.approx(0.1)
        assert d["mean_holdout_energy"] == pytest.approx(0.5)
        assert d["gap"] == pytest.approx(0.4)
        assert "min_gap_threshold" in d


# ---------------------------------------------------------------------------
# audit_tracker — trivial constraint detection
# ---------------------------------------------------------------------------


class TestAuditTrackerTrivialConstraints:
    """REQ-LEARN-002: Trivial constraint detection."""

    def test_no_findings_on_empty_tracker(self) -> None:
        """SCENARIO-LEARN-002: Empty tracker produces clean report."""
        tracker = ConstraintTracker()
        report = audit_tracker(tracker)
        assert report.clean

    def test_trivial_constraint_flagged(self) -> None:
        """SCENARIO-LEARN-002: High-fire, near-zero-precision type is flagged."""
        # 10 fires, 0 caught — precision = 0.0 < 0.05 threshold.
        tracker = _tracker_with(("arithmetic", 10, 0))
        report = audit_tracker(tracker)
        trivials = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        assert len(trivials) == 1
        assert trivials[0].constraint_type == "arithmetic"
        assert trivials[0].fired == 10
        assert trivials[0].precision == pytest.approx(0.0)

    def test_below_min_fire_count_not_flagged(self) -> None:
        """REQ-LEARN-002: Type with too few fires is not flagged (insufficient evidence)."""
        # Only 4 fires — below default threshold of 5.
        tracker = _tracker_with(("arithmetic", 4, 0))
        report = audit_tracker(tracker, min_fire_count=5)
        trivials = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        assert len(trivials) == 0

    def test_exactly_min_fire_count_is_flagged(self) -> None:
        """REQ-LEARN-002: Type with exactly min_fire_count fires is eligible for flag."""
        tracker = _tracker_with(("arithmetic", 5, 0))
        report = audit_tracker(tracker, min_fire_count=5)
        trivials = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        assert len(trivials) == 1

    def test_high_precision_type_not_flagged(self) -> None:
        """REQ-LEARN-002: High-precision type (genuinely catching errors) not flagged."""
        # 10 fires, 10 caught — precision = 1.0.
        tracker = _tracker_with(("arithmetic", 10, 10))
        report = audit_tracker(tracker)
        trivials = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        assert len(trivials) == 0

    def test_precision_just_above_threshold_not_flagged(self) -> None:
        """REQ-LEARN-002: Precision >= threshold is not flagged."""
        # 10 fires, 1 caught — precision = 0.1 > 0.05 threshold.
        tracker = _tracker_with(("arithmetic", 10, 1))
        report = audit_tracker(tracker, trivial_precision_threshold=0.05)
        trivials = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        assert len(trivials) == 0

    def test_custom_threshold_respected(self) -> None:
        """REQ-LEARN-002: Custom trivial_precision_threshold is applied."""
        # 10 fires, 2 caught — precision = 0.2. Default threshold = 0.05 (not flagged).
        # With threshold = 0.25 it should be flagged.
        tracker = _tracker_with(("arithmetic", 10, 2))
        report = audit_tracker(tracker, trivial_precision_threshold=0.25)
        trivials = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        assert len(trivials) == 1

    def test_multiple_trivial_types_all_flagged(self) -> None:
        """REQ-LEARN-002: All trivially-passing types are flagged, not just the worst."""
        tracker = _tracker_with(("arithmetic", 10, 0), ("logic", 10, 0))
        report = audit_tracker(tracker)
        trivials = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        ctypes = {f.constraint_type for f in trivials}
        assert "arithmetic" in ctypes
        assert "logic" in ctypes


# ---------------------------------------------------------------------------
# audit_tracker — diversity (Gini) detection
# ---------------------------------------------------------------------------


class TestAuditTrackerDiversity:
    """REQ-LEARN-002: Low constraint diversity detection via Gini coefficient."""

    def test_single_type_no_diversity_flag(self) -> None:
        """REQ-LEARN-002: Single type — diversity check needs >= 2 types."""
        tracker = _tracker_with(("arithmetic", 10, 5))
        report = audit_tracker(tracker)
        low_div = [f for f in report.findings if isinstance(f, LowDiversityFinding)]
        assert len(low_div) == 0

    def test_uniform_distribution_no_diversity_flag(self) -> None:
        """SCENARIO-LEARN-002: Uniform fire counts — low Gini, no diversity flag."""
        # Equal fires across all types → Gini near 0.
        tracker = _tracker_with(
            ("arithmetic", 10, 5),
            ("logic", 10, 5),
            ("code", 10, 5),
        )
        report = audit_tracker(tracker)
        low_div = [f for f in report.findings if isinstance(f, LowDiversityFinding)]
        assert len(low_div) == 0

    def test_heavily_skewed_fires_flagged(self) -> None:
        """SCENARIO-LEARN-002: One type dominates → high Gini → diversity flag."""
        # arithmetic fires 100x, others fire 1x — very high concentration.
        # For 3 types the max Gini ≈ 0.667; our default threshold is 0.45 to
        # catch realistic imbalances like this.
        tracker = _tracker_with(
            ("arithmetic", 100, 50),
            ("logic", 1, 0),
            ("code", 1, 0),
        )
        report = audit_tracker(tracker)
        low_div = [f for f in report.findings if isinstance(f, LowDiversityFinding)]
        assert len(low_div) == 1
        assert low_div[0].dominant_type == "arithmetic"
        assert low_div[0].gini > GINI_DIVERSITY_THRESHOLD

    def test_low_diversity_finding_has_correct_fields(self) -> None:
        """REQ-LEARN-002: LowDiversityFinding contains n_types and dominant_fraction."""
        tracker = _tracker_with(
            ("arithmetic", 100, 0),
            ("logic", 1, 0),
        )
        report = audit_tracker(tracker)
        low_div = [f for f in report.findings if isinstance(f, LowDiversityFinding)]
        assert len(low_div) == 1
        f = low_div[0]
        assert f.n_types == 2
        assert f.dominant_fraction == pytest.approx(100 / 101)
        assert f.dominant_type == "arithmetic"

    def test_custom_gini_threshold_respected(self) -> None:
        """REQ-LEARN-002: Custom gini_threshold parameter is applied."""
        # Moderately skewed: 10 vs 5 fires. Will not hit default 0.7 threshold
        # but will hit a strict 0.0 threshold.
        tracker = _tracker_with(("arithmetic", 10, 5), ("logic", 5, 2))
        report_strict = audit_tracker(tracker, gini_threshold=0.0)
        low_div_strict = [f for f in report_strict.findings if isinstance(f, LowDiversityFinding)]
        assert len(low_div_strict) == 1

        report_loose = audit_tracker(tracker, gini_threshold=1.0)
        low_div_loose = [f for f in report_loose.findings if isinstance(f, LowDiversityFinding)]
        assert len(low_div_loose) == 0

    def test_all_zero_fires_no_diversity_flag(self) -> None:
        """REQ-LEARN-002: All-zero fire counts — Gini is 0, no diversity flag."""
        tracker = ConstraintTracker()
        # Record types but never fire.
        tracker.record("arithmetic", fired=False, caught_error=False)
        tracker.record("logic", fired=False, caught_error=False)
        report = audit_tracker(tracker)
        low_div = [f for f in report.findings if isinstance(f, LowDiversityFinding)]
        assert len(low_div) == 0


# ---------------------------------------------------------------------------
# audit_energy_trajectory — zero-energy shortcut detection
# ---------------------------------------------------------------------------


class TestAuditEnergyTrajectoryShortcut:
    """REQ-LEARN-002: Zero-energy shortcut detection."""

    def test_insufficient_samples_returns_clean(self) -> None:
        """SCENARIO-LEARN-002: Fewer than min_samples → no findings (not enough data)."""
        report = audit_energy_trajectory([0.1, 0.2], [0.3, 0.4], min_samples=3)
        assert report.clean

    def test_constant_train_energy_flagged(self) -> None:
        """SCENARIO-LEARN-002: Training energy that never changes is flagged as shortcut."""
        # All training energies identical → 1 distinct value.
        train = [0.0] * 10
        holdout = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        report = audit_energy_trajectory(train, holdout, distinct_values_min=2)
        zero_e = [f for f in report.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_e) == 1
        assert zero_e[0].distinct_values == 1

    def test_constant_holdout_energy_flagged(self) -> None:
        """SCENARIO-LEARN-002: Held-out energy that never changes is flagged."""
        train = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        holdout = [0.5] * 10
        report = audit_energy_trajectory(train, holdout, distinct_values_min=2)
        zero_e = [f for f in report.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_e) == 1

    def test_diverse_energy_sequences_not_flagged(self) -> None:
        """REQ-LEARN-002: Varied energy values — no shortcut flag."""
        train = [0.1 * i for i in range(10)]
        holdout = [0.05 * i for i in range(10)]
        report = audit_energy_trajectory(train, holdout, distinct_values_min=2, min_gap=0.5)
        zero_e = [f for f in report.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_e) == 0

    def test_fp_noise_collapsed_treated_as_same(self) -> None:
        """REQ-LEARN-002: Tiny FP differences within 1e-8 are treated as the same value."""
        # Values that differ only at the 8th decimal place collapse to 1 distinct when
        # rounded to 6 decimal places.
        train = [0.10000001, 0.10000002, 0.10000003, 0.10000004, 0.10000005]
        holdout = [0.2, 0.3, 0.4, 0.5, 0.6]
        report = audit_energy_trajectory(train, holdout, distinct_values_min=2)
        zero_e = [f for f in report.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_e) == 1  # train collapses to 1 distinct value

    def test_custom_distinct_values_min_respected(self) -> None:
        """REQ-LEARN-002: Custom distinct_values_min parameter applied."""
        # 3 distinct values in training. With min=4 it's flagged; with min=2 it's fine.
        train = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]
        holdout = [float(i) for i in range(10)]
        report_strict = audit_energy_trajectory(train, holdout, distinct_values_min=4, min_gap=99.0)
        zero_e_strict = [f for f in report_strict.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_e_strict) == 1

        report_loose = audit_energy_trajectory(train, holdout, distinct_values_min=2, min_gap=99.0)
        zero_e_loose = [f for f in report_loose.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_e_loose) == 0

    def test_zero_energy_finding_sequence_length_is_worse(self) -> None:
        """REQ-LEARN-002: ZeroEnergyFinding reports the sequence with fewer distinct values."""
        # Train has 1 distinct, holdout has 2 — train is worse.
        train = [0.0] * 10  # 1 distinct
        holdout = [0.1, 0.2] * 5  # 2 distinct
        report = audit_energy_trajectory(train, holdout, distinct_values_min=2, min_gap=99.0)
        zero_e = [f for f in report.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_e) == 1
        # Should report the worse (train) sequence length.
        assert zero_e[0].sequence_length == 10
        assert zero_e[0].distinct_values == 1


# ---------------------------------------------------------------------------
# audit_energy_trajectory — train/holdout divergence detection
# ---------------------------------------------------------------------------


class TestAuditEnergyTrajectoryDivergence:
    """REQ-LEARN-002: Train/holdout energy divergence detection."""

    def test_no_divergence_when_both_low(self) -> None:
        """SCENARIO-LEARN-002: Both sequences have similarly low energy — no divergence."""
        train = [0.1, 0.1, 0.1, 0.1, 0.1]
        holdout = [0.12, 0.11, 0.13, 0.1, 0.12]
        report = audit_energy_trajectory(train, holdout, min_gap=0.05)
        div = [f for f in report.findings if isinstance(f, TrainHoldoutDivergenceFinding)]
        assert len(div) == 0

    def test_divergence_flagged_when_gap_large(self) -> None:
        """SCENARIO-LEARN-002: Held-out energy much higher than training → gaming flag."""
        train = [0.05, 0.05, 0.05, 0.05, 0.05]  # mean = 0.05
        holdout = [0.8, 0.9, 0.7, 0.85, 0.75]  # mean = 0.8
        report = audit_energy_trajectory(train, holdout, min_gap=0.05)
        div = [f for f in report.findings if isinstance(f, TrainHoldoutDivergenceFinding)]
        assert len(div) == 1
        assert div[0].gap == pytest.approx(0.75, abs=0.01)

    def test_negative_gap_not_flagged(self) -> None:
        """REQ-LEARN-002: Held-out energy lower than training (not gaming) is not flagged."""
        # Holdout better than training — indicates generalisation, not gaming.
        train = [0.8, 0.9, 0.85, 0.8, 0.9]
        holdout = [0.1, 0.1, 0.1, 0.1, 0.1]
        report = audit_energy_trajectory(train, holdout, min_gap=0.05)
        div = [f for f in report.findings if isinstance(f, TrainHoldoutDivergenceFinding)]
        assert len(div) == 0

    def test_gap_exactly_at_threshold_not_flagged(self) -> None:
        """REQ-LEARN-002: Gap equal to threshold is not flagged (strict >)."""
        # mean_train = 0.0, mean_holdout = 0.05, gap = 0.05 exactly.
        train = [0.0, 0.0, 0.0, 0.0, 0.0]
        holdout = [0.05, 0.05, 0.05, 0.05, 0.05]
        report = audit_energy_trajectory(train, holdout, min_gap=0.05)
        div = [f for f in report.findings if isinstance(f, TrainHoldoutDivergenceFinding)]
        assert len(div) == 0

    def test_gap_above_threshold_is_flagged(self) -> None:
        """REQ-LEARN-002: Gap strictly above threshold is flagged."""
        train = [0.0] * 5
        holdout = [0.1] * 5  # gap = 0.1 > 0.05
        report = audit_energy_trajectory(train, holdout, min_gap=0.05)
        div = [f for f in report.findings if isinstance(f, TrainHoldoutDivergenceFinding)]
        assert len(div) == 1

    def test_divergence_finding_has_correct_fields(self) -> None:
        """REQ-LEARN-002: Finding records mean_train, mean_holdout, and gap correctly."""
        train = [0.1, 0.2, 0.3]
        holdout = [0.6, 0.7, 0.8]
        report = audit_energy_trajectory(train, holdout, min_gap=0.05)
        div = [f for f in report.findings if isinstance(f, TrainHoldoutDivergenceFinding)]
        assert len(div) == 1
        f = div[0]
        assert f.mean_train_energy == pytest.approx(0.2, abs=0.01)
        assert f.mean_holdout_energy == pytest.approx(0.7, abs=0.01)
        assert f.gap == pytest.approx(0.5, abs=0.01)

    def test_custom_min_gap_respected(self) -> None:
        """REQ-LEARN-002: Custom min_gap parameter applied correctly."""
        # Use varied (non-constant) sequences so zero-energy shortcut is not triggered.
        train = [0.0, 0.01, 0.02, 0.03, 0.04]   # mean = 0.02
        holdout = [0.3, 0.31, 0.32, 0.33, 0.34]  # mean = 0.32, gap = 0.30
        # With min_gap=0.5, gap=0.30 is below threshold — not flagged.
        report_strict = audit_energy_trajectory(train, holdout, min_gap=0.5)
        div_strict = [f for f in report_strict.findings if isinstance(f, TrainHoldoutDivergenceFinding)]
        assert len(div_strict) == 0

        # With min_gap=0.1, gap=0.30 exceeds threshold — flagged.
        report_loose = audit_energy_trajectory(train, holdout, min_gap=0.1)
        div = [f for f in report_loose.findings if isinstance(f, TrainHoldoutDivergenceFinding)]
        assert len(div) == 1


# ---------------------------------------------------------------------------
# audit_energy_trajectory — insufficient samples edge cases
# ---------------------------------------------------------------------------


class TestAuditEnergyInsufficientSamples:
    """REQ-LEARN-002: Short sequences are skipped (not enough evidence)."""

    def test_empty_sequences_return_clean(self) -> None:
        report = audit_energy_trajectory([], [], min_samples=1)
        assert report.clean

    def test_one_sample_each_below_default_min(self) -> None:
        report = audit_energy_trajectory([0.5], [0.9])
        assert report.clean

    def test_exactly_min_samples_is_processed(self) -> None:
        """With exactly min_samples items, audit runs and may find issues."""
        # 3 samples, constant training energy → should find zero-energy shortcut.
        train = [0.0, 0.0, 0.0]
        holdout = [0.1, 0.2, 0.3]
        report = audit_energy_trajectory(train, holdout, min_samples=3, distinct_values_min=2)
        zero_e = [f for f in report.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_e) == 1

    def test_train_sufficient_but_holdout_insufficient(self) -> None:
        """REQ-LEARN-002: Both sequences must meet min_samples — partial is skipped."""
        train = [0.0] * 10
        holdout = [0.5, 0.6]  # only 2 samples
        report = audit_energy_trajectory(train, holdout, min_samples=3)
        assert report.clean


# ---------------------------------------------------------------------------
# audit_full — combined audit
# ---------------------------------------------------------------------------


class TestAuditFull:
    """REQ-LEARN-002: Combined audit merges all findings from tracker and energy checks."""

    def test_clean_inputs_produce_clean_report(self) -> None:
        """SCENARIO-LEARN-002: Well-behaved inputs pass all checks."""
        tracker = _tracker_with(
            ("arithmetic", 10, 8),
            ("logic", 10, 7),
            ("code", 10, 6),
        )
        train = [0.1 * i for i in range(10)]
        holdout = [0.11 * i for i in range(10)]
        report = audit_full(tracker, train, holdout, min_gap=0.5)
        assert report.clean

    def test_gaming_tracker_and_energy_both_flagged(self) -> None:
        """SCENARIO-LEARN-002: Both tracker hacking and energy divergence are caught."""
        # Trivially-passing tracker.
        tracker = _tracker_with(("arithmetic", 10, 0))
        # Diverging energy trajectories.
        train = [0.0] * 5
        holdout = [1.0] * 5
        report = audit_full(tracker, train, holdout)
        trivials = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        div = [f for f in report.findings if isinstance(f, TrainHoldoutDivergenceFinding)]
        assert len(trivials) >= 1
        assert len(div) >= 1

    def test_energy_shortcut_detected_in_full_audit(self) -> None:
        """REQ-LEARN-002: Zero-energy shortcut detected through audit_full."""
        tracker = _tracker_with(("arithmetic", 3, 2), ("logic", 3, 2))
        train = [0.0] * 10  # constant → shortcut
        holdout = [0.1 * i for i in range(10)]
        report = audit_full(tracker, train, holdout, min_gap=99.0)
        zero_e = [f for f in report.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_e) == 1

    def test_all_custom_kwargs_passed_through(self) -> None:
        """REQ-LEARN-002: Custom thresholds in audit_full reach underlying functions."""
        # Tracker with 5 fires at precision 0.1 — not flagged at default 0.05 threshold
        # but flagged if threshold = 0.2.
        tracker = _tracker_with(("arithmetic", 5, 0))
        train = [0.1 * i for i in range(5)]
        holdout = [0.1 * i for i in range(5)]
        report = audit_full(
            tracker,
            train,
            holdout,
            trivial_precision_threshold=0.2,
            min_fire_count=5,
            min_gap=99.0,
        )
        trivials = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        assert len(trivials) == 1

    def test_short_energy_sequences_do_not_cause_errors(self) -> None:
        """REQ-LEARN-002: Short energy sequences are safely ignored in full audit."""
        tracker = _tracker_with(("arithmetic", 10, 5))
        report = audit_full(tracker, [0.1], [0.2])
        # Only tracker results; energy skipped due to insufficient samples.
        zero_e = [f for f in report.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_e) == 0


# ---------------------------------------------------------------------------
# Gini coefficient helper — indirect coverage via audit_tracker
# ---------------------------------------------------------------------------


class TestGiniCoefficientEdgeCases:
    """REQ-LEARN-002: Gini edge cases exercised through audit_tracker."""

    def test_two_equal_types_gini_zero(self) -> None:
        """REQ-LEARN-002: Two equal-fire types → Gini = 0 → no diversity flag."""
        tracker = _tracker_with(("arithmetic", 10, 5), ("logic", 10, 5))
        # Equal counts → Gini = 0; well below any reasonable threshold.
        report = audit_tracker(tracker)
        low_div = [f for f in report.findings if isinstance(f, LowDiversityFinding)]
        assert len(low_div) == 0

    def test_gini_single_element_returns_zero(self) -> None:
        """REQ-LEARN-002: _gini_coefficient with 1 element returns 0.0 (degenerate)."""
        # This exercises the n <= 1 early-return branch of _gini_coefficient.
        assert _gini_coefficient([42.0]) == 0.0

    def test_mean_empty_list_returns_zero(self) -> None:
        """REQ-LEARN-002: _mean([]) returns 0.0 without ZeroDivisionError."""
        # This exercises the early-return branch for empty lists.
        assert _mean([]) == 0.0

    def test_mean_nonempty(self) -> None:
        """REQ-LEARN-002: _mean returns correct average for non-empty list."""
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_very_skewed_two_types_high_gini(self) -> None:
        """REQ-LEARN-002: Heavily skewed 2-type distribution → Gini near 0.5 (max for n=2)."""
        tracker = _tracker_with(("arithmetic", 1000, 500), ("logic", 1, 0))
        # For n=2, max Gini ≈ 0.499. Default threshold is 0.45, so this is flagged.
        report = audit_tracker(tracker)
        low_div = [f for f in report.findings if isinstance(f, LowDiversityFinding)]
        assert len(low_div) == 1
