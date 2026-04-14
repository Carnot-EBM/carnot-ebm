"""Tests for carnot.verify.reward_hacking — reward hacking detection.

Covers all detection paths for gaming the self-learning pipeline energy
function: trivial constraints, zero-energy shortcuts, constraint monoculture,
and train/holdout divergence.

Spec: REQ-LEARN-002, SCENARIO-LEARN-002
"""

from __future__ import annotations

import pytest

from carnot.pipeline.tracker import ConstraintTracker
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
    _gini_coefficient,
    _mean,
    audit_energy_trajectory,
    audit_full,
    audit_tracker,
)


# ---------------------------------------------------------------------------
# REQ-LEARN-002: Helper functions
# ---------------------------------------------------------------------------


class TestGiniCoefficient:
    """REQ-LEARN-002: Gini coefficient computation edge cases."""

    def test_uniform_distribution_is_zero(self) -> None:
        """SCENARIO-LEARN-002: Equal values produce Gini = 0.0."""
        assert _gini_coefficient([10.0, 10.0, 10.0]) == pytest.approx(0.0)

    def test_single_element_is_zero(self) -> None:
        """REQ-LEARN-002: Single element cannot be unequal — Gini = 0.0."""
        assert _gini_coefficient([42.0]) == 0.0

    def test_empty_list_is_zero(self) -> None:
        """REQ-LEARN-002: Empty list degenerate case — Gini = 0.0."""
        assert _gini_coefficient([]) == 0.0

    def test_all_zeros_is_zero(self) -> None:
        """REQ-LEARN-002: All-zero values total=0 degenerate — Gini = 0.0."""
        assert _gini_coefficient([0.0, 0.0, 0.0]) == 0.0

    def test_highly_concentrated_is_above_threshold(self) -> None:
        """SCENARIO-LEARN-002: One dominant type produces Gini above threshold."""
        # 100:1:1 fire ratio should be very concentrated.
        gini = _gini_coefficient([100.0, 1.0, 1.0])
        assert gini > GINI_DIVERSITY_THRESHOLD

    def test_two_equal_values_is_zero(self) -> None:
        """REQ-LEARN-002: Two equal values give Gini = 0.0."""
        assert _gini_coefficient([5.0, 5.0]) == pytest.approx(0.0)

    def test_two_unequal_values_is_nonzero(self) -> None:
        """REQ-LEARN-002: Unequal two-value list produces Gini > 0."""
        gini = _gini_coefficient([1.0, 100.0])
        assert gini > 0.0

    def test_result_in_range(self) -> None:
        """REQ-LEARN-002: Gini is always in [0, 1)."""
        gini = _gini_coefficient([1.0, 2.0, 100.0, 0.5])
        assert 0.0 <= gini < 1.0


class TestMean:
    """REQ-LEARN-002: Arithmetic mean helper."""

    def test_mean_empty_is_zero(self) -> None:
        """REQ-LEARN-002: Empty list returns 0.0 (safe default)."""
        assert _mean([]) == 0.0

    def test_mean_single(self) -> None:
        """REQ-LEARN-002: Single-element list returns that element."""
        assert _mean([7.0]) == pytest.approx(7.0)

    def test_mean_multiple(self) -> None:
        """REQ-LEARN-002: Mean of [1, 2, 3] = 2.0."""
        assert _mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_mean_zeros(self) -> None:
        """REQ-LEARN-002: Mean of all zeros is 0.0."""
        assert _mean([0.0, 0.0, 0.0]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# REQ-LEARN-002: Finding data class serialisation
# ---------------------------------------------------------------------------


class TestFindingDicts:
    """REQ-LEARN-002: Each finding's to_dict() includes kind + all fields."""

    def test_trivial_constraint_to_dict(self) -> None:
        """SCENARIO-LEARN-002: TrivialConstraintFinding serialises correctly."""
        f = TrivialConstraintFinding(
            constraint_type="arithmetic",
            fired=20,
            precision=0.02,
            threshold=TRIVIAL_PRECISION_THRESHOLD,
        )
        d = f.to_dict()
        assert d["kind"] == "trivial_constraint"
        assert d["constraint_type"] == "arithmetic"
        assert d["fired"] == 20
        assert d["precision"] == pytest.approx(0.02)
        assert d["threshold"] == TRIVIAL_PRECISION_THRESHOLD

    def test_zero_energy_to_dict(self) -> None:
        """SCENARIO-LEARN-002: ZeroEnergyFinding serialises correctly."""
        f = ZeroEnergyFinding(
            sequence_length=10,
            distinct_values=1,
            min_distinct_required=ENERGY_DISTINCT_VALUES_MIN,
        )
        d = f.to_dict()
        assert d["kind"] == "zero_energy_shortcut"
        assert d["sequence_length"] == 10
        assert d["distinct_values"] == 1
        assert d["min_distinct_required"] == ENERGY_DISTINCT_VALUES_MIN

    def test_low_diversity_to_dict(self) -> None:
        """SCENARIO-LEARN-002: LowDiversityFinding serialises correctly."""
        f = LowDiversityFinding(
            gini=0.8,
            n_types=3,
            dominant_type="code",
            dominant_fraction=0.9,
            threshold=GINI_DIVERSITY_THRESHOLD,
        )
        d = f.to_dict()
        assert d["kind"] == "low_diversity"
        assert d["gini"] == pytest.approx(0.8)
        assert d["n_types"] == 3
        assert d["dominant_type"] == "code"
        assert d["dominant_fraction"] == pytest.approx(0.9)
        assert d["threshold"] == GINI_DIVERSITY_THRESHOLD

    def test_train_holdout_divergence_to_dict(self) -> None:
        """SCENARIO-LEARN-002: TrainHoldoutDivergenceFinding serialises correctly."""
        f = TrainHoldoutDivergenceFinding(
            mean_train_energy=0.1,
            mean_holdout_energy=0.5,
            gap=0.4,
            min_gap_threshold=DIVERGENCE_MIN_GAP,
        )
        d = f.to_dict()
        assert d["kind"] == "train_holdout_divergence"
        assert d["mean_train_energy"] == pytest.approx(0.1)
        assert d["mean_holdout_energy"] == pytest.approx(0.5)
        assert d["gap"] == pytest.approx(0.4)
        assert d["min_gap_threshold"] == DIVERGENCE_MIN_GAP


# ---------------------------------------------------------------------------
# REQ-LEARN-002: RewardHackingReport
# ---------------------------------------------------------------------------


class TestRewardHackingReport:
    """REQ-LEARN-002: Report aggregation and serialisation."""

    def test_clean_when_no_findings(self) -> None:
        """SCENARIO-LEARN-002: Report with no findings is clean."""
        report = RewardHackingReport()
        assert report.clean is True

    def test_not_clean_when_findings_present(self) -> None:
        """REQ-LEARN-002: Report with any finding is not clean."""
        report = RewardHackingReport(
            findings=[
                TrivialConstraintFinding(
                    constraint_type="x", fired=10, precision=0.01
                )
            ]
        )
        assert report.clean is False

    def test_to_dict_clean_report(self) -> None:
        """REQ-LEARN-002: to_dict() on clean report has clean=True and n_findings=0."""
        report = RewardHackingReport()
        d = report.to_dict()
        assert d["clean"] is True
        assert d["n_findings"] == 0
        assert d["findings"] == []

    def test_to_dict_with_findings(self) -> None:
        """REQ-LEARN-002: to_dict() includes all serialised findings."""
        f = ZeroEnergyFinding(sequence_length=5, distinct_values=1)
        report = RewardHackingReport(findings=[f])
        d = report.to_dict()
        assert d["clean"] is False
        assert d["n_findings"] == 1
        assert len(d["findings"]) == 1
        assert d["findings"][0]["kind"] == "zero_energy_shortcut"


# ---------------------------------------------------------------------------
# REQ-LEARN-002: audit_tracker()
# ---------------------------------------------------------------------------


class TestAuditTracker:
    """REQ-LEARN-002: Tracker audit detects trivial constraints and low diversity."""

    def _make_tracker(self, **type_fired_caught: tuple[int, int]) -> ConstraintTracker:
        """Helper: build a tracker from {type: (fires, catches)} dict."""
        tracker = ConstraintTracker()
        for ctype, (fires, catches) in type_fired_caught.items():
            for i in range(fires):
                caught = i < catches
                tracker.record(ctype, fired=True, caught_error=caught, any_error_in_batch=True)
        return tracker

    def test_empty_tracker_is_clean(self) -> None:
        """SCENARIO-LEARN-002: Empty tracker produces no findings."""
        tracker = ConstraintTracker()
        report = audit_tracker(tracker)
        assert report.clean is True

    def test_trivial_constraint_flagged(self) -> None:
        """SCENARIO-LEARN-002: High-fire, near-zero-precision type is flagged."""
        # fire 10 times, catch 0 — precision=0.0, well below 0.05 threshold.
        tracker = self._make_tracker(arithmetic=(10, 0))
        report = audit_tracker(tracker)
        finding_types = [type(f).__name__ for f in report.findings]
        assert "TrivialConstraintFinding" in finding_types

    def test_trivial_constraint_finding_details(self) -> None:
        """REQ-LEARN-002: TrivialConstraintFinding has correct type and precision."""
        tracker = self._make_tracker(arithmetic=(10, 0))
        report = audit_tracker(tracker)
        tcf = next(f for f in report.findings if isinstance(f, TrivialConstraintFinding))
        assert tcf.constraint_type == "arithmetic"
        assert tcf.fired == 10
        assert tcf.precision == pytest.approx(0.0)

    def test_below_min_fire_count_not_flagged(self) -> None:
        """REQ-LEARN-002: Fewer than min_fire_count fires — not enough evidence to flag."""
        # fire only 3 times (below default threshold of 5).
        tracker = self._make_tracker(arithmetic=(3, 0))
        report = audit_tracker(tracker)
        trivial = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        assert len(trivial) == 0

    def test_high_precision_not_flagged_as_trivial(self) -> None:
        """REQ-LEARN-002: High-precision constraint is not trivial."""
        tracker = self._make_tracker(arithmetic=(10, 10))
        report = audit_tracker(tracker)
        trivial = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        assert len(trivial) == 0

    def test_precision_at_threshold_not_flagged(self) -> None:
        """REQ-LEARN-002: Precision exactly at threshold is not flagged (strict <)."""
        # 1 catch out of 20 fires = 0.05 precision exactly = threshold; should NOT flag.
        tracker = self._make_tracker(arithmetic=(20, 1))
        report = audit_tracker(tracker, trivial_precision_threshold=0.05)
        trivial = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        assert len(trivial) == 0

    def test_low_diversity_flagged_for_dominant_type(self) -> None:
        """SCENARIO-LEARN-002: Heavily skewed fire distribution triggers low diversity."""
        # 'code' fires 100x, 'logic' fires 1x — extreme concentration.
        tracker = self._make_tracker(code=(100, 10), logic=(1, 0))
        report = audit_tracker(tracker)
        diversity_findings = [f for f in report.findings if isinstance(f, LowDiversityFinding)]
        assert len(diversity_findings) == 1

    def test_low_diversity_dominant_type_identified(self) -> None:
        """REQ-LEARN-002: LowDiversityFinding correctly names the dominant type."""
        tracker = self._make_tracker(code=(100, 10), logic=(1, 0))
        report = audit_tracker(tracker)
        ldf = next(f for f in report.findings if isinstance(f, LowDiversityFinding))
        assert ldf.dominant_type == "code"
        assert ldf.dominant_fraction > 0.9

    def test_single_type_no_diversity_check(self) -> None:
        """REQ-LEARN-002: Diversity check requires at least 2 types; single type skipped."""
        tracker = self._make_tracker(code=(100, 0))
        report = audit_tracker(tracker)
        # May flag trivial but NOT low diversity (can't compute Gini with 1 type).
        diversity_findings = [f for f in report.findings if isinstance(f, LowDiversityFinding)]
        assert len(diversity_findings) == 0

    def test_balanced_two_types_no_diversity_flag(self) -> None:
        """REQ-LEARN-002: Balanced distribution does not trigger low diversity."""
        tracker = self._make_tracker(arithmetic=(50, 5), logic=(50, 5))
        report = audit_tracker(tracker)
        diversity_findings = [f for f in report.findings if isinstance(f, LowDiversityFinding)]
        assert len(diversity_findings) == 0

    def test_custom_thresholds_applied(self) -> None:
        """REQ-LEARN-002: Custom min_fire_count and trivial_precision_threshold respected."""
        # With min_fire_count=20 and only 10 fires, should not flag as trivial.
        tracker = self._make_tracker(arithmetic=(10, 0))
        report = audit_tracker(tracker, min_fire_count=20)
        trivial = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        assert len(trivial) == 0

    def test_multiple_trivial_types_all_flagged(self) -> None:
        """REQ-LEARN-002: Multiple trivially-passing types all appear in report."""
        tracker = self._make_tracker(arithmetic=(10, 0), logic=(10, 0), code=(10, 0))
        report = audit_tracker(tracker)
        trivial = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        flagged_types = {f.constraint_type for f in trivial}
        assert "arithmetic" in flagged_types
        assert "logic" in flagged_types
        assert "code" in flagged_types

    def test_all_zero_fired_for_gini(self) -> None:
        """REQ-LEARN-002: All-zero fire counts don't cause ZeroDivisionError in Gini."""
        # Build tracker with explicit zero-fire types using only 'any_error_in_batch' records.
        tracker = ConstraintTracker()
        # Record with fired=False for both types — they appear in stats but fired=0.
        tracker.record("a", fired=False, caught_error=False, any_error_in_batch=True)
        tracker.record("b", fired=False, caught_error=False, any_error_in_batch=True)
        # Should not raise even though all fired=0.
        report = audit_tracker(tracker)
        assert isinstance(report, RewardHackingReport)


# ---------------------------------------------------------------------------
# REQ-LEARN-002: audit_energy_trajectory()
# ---------------------------------------------------------------------------


class TestAuditEnergyTrajectory:
    """REQ-LEARN-002: Energy trajectory audit detects shortcuts and gaming."""

    def test_too_few_samples_returns_clean(self) -> None:
        """SCENARIO-LEARN-002: Fewer than min_samples — not enough data to judge."""
        report = audit_energy_trajectory([0.1, 0.2], [0.1, 0.2])
        assert report.clean is True

    def test_train_too_short_clean(self) -> None:
        """REQ-LEARN-002: Training sequence too short — clean report."""
        report = audit_energy_trajectory([0.1], [0.1, 0.2, 0.3])
        assert report.clean is True

    def test_holdout_too_short_clean(self) -> None:
        """REQ-LEARN-002: Holdout sequence too short — clean report."""
        report = audit_energy_trajectory([0.1, 0.2, 0.3], [0.1])
        assert report.clean is True

    def test_constant_train_energy_flagged(self) -> None:
        """SCENARIO-LEARN-002: Constant training energy is flagged as zero-energy shortcut."""
        train = [0.0, 0.0, 0.0, 0.0, 0.0]
        holdout = [0.1, 0.2, 0.3, 0.4, 0.5]
        report = audit_energy_trajectory(train, holdout)
        zero_findings = [f for f in report.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_findings) == 1

    def test_constant_holdout_energy_flagged(self) -> None:
        """SCENARIO-LEARN-002: Constant holdout energy is flagged as zero-energy shortcut."""
        train = [0.1, 0.2, 0.3, 0.4, 0.5]
        holdout = [0.5, 0.5, 0.5, 0.5, 0.5]
        report = audit_energy_trajectory(train, holdout)
        zero_findings = [f for f in report.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_findings) == 1

    def test_zero_finding_reports_shorter_sequence(self) -> None:
        """REQ-LEARN-002: ZeroEnergyFinding reports the problematic sequence's length."""
        # Both constant — report uses the one with fewer distinct values (tie: train).
        train = [0.0, 0.0, 0.0]
        holdout = [1.0, 1.0, 1.0]
        report = audit_energy_trajectory(train, holdout)
        zf = next(f for f in report.findings if isinstance(f, ZeroEnergyFinding))
        # Both have distinct=1; both have same length=3. Should be 3.
        assert zf.sequence_length == 3
        assert zf.distinct_values == 1

    def test_holdout_has_fewer_distinct_reports_holdout_length(self) -> None:
        """REQ-LEARN-002: When holdout is more constant, reports holdout's length."""
        # train: 3 distinct values (fine), holdout: 1 distinct value (constant).
        train = [0.1, 0.2, 0.3, 0.4, 0.5]
        holdout = [0.9, 0.9, 0.9, 0.9, 0.9]
        report = audit_energy_trajectory(train, holdout)
        zf = next(f for f in report.findings if isinstance(f, ZeroEnergyFinding))
        assert zf.sequence_length == 5  # holdout length
        assert zf.distinct_values == 1

    def test_diverse_energies_no_shortcut_flag(self) -> None:
        """REQ-LEARN-002: Diverse energy sequences are not flagged as shortcuts."""
        train = [0.1, 0.3, 0.5, 0.2, 0.4]
        holdout = [0.2, 0.4, 0.6, 0.1, 0.3]
        report = audit_energy_trajectory(train, holdout)
        zero_findings = [f for f in report.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_findings) == 0

    def test_train_holdout_divergence_flagged(self) -> None:
        """SCENARIO-LEARN-002: Improving training + flat holdout triggers divergence flag."""
        # Training improves (low energy); holdout stays high.
        train = [0.05, 0.05, 0.05, 0.05, 0.05]
        holdout = [0.8, 0.8, 0.8, 0.8, 0.8]
        report = audit_energy_trajectory(train, holdout)
        div_findings = [f for f in report.findings if isinstance(f, TrainHoldoutDivergenceFinding)]
        assert len(div_findings) == 1

    def test_divergence_finding_details(self) -> None:
        """REQ-LEARN-002: TrainHoldoutDivergenceFinding has correct mean values."""
        train = [0.1, 0.1, 0.1]
        holdout = [0.9, 0.9, 0.9]
        report = audit_energy_trajectory(train, holdout)
        df = next(f for f in report.findings if isinstance(f, TrainHoldoutDivergenceFinding))
        assert df.mean_train_energy == pytest.approx(0.1)
        assert df.mean_holdout_energy == pytest.approx(0.9)
        assert df.gap == pytest.approx(0.8)

    def test_both_improving_no_divergence(self) -> None:
        """SCENARIO-LEARN-002: Both train and holdout with similar means — not diverging."""
        # Small gap (within noise threshold).
        train = [0.2, 0.2, 0.2, 0.2, 0.2]
        holdout = [0.22, 0.22, 0.22, 0.22, 0.22]
        report = audit_energy_trajectory(train, holdout)
        div_findings = [f for f in report.findings if isinstance(f, TrainHoldoutDivergenceFinding)]
        assert len(div_findings) == 0

    def test_holdout_lower_than_train_no_flag(self) -> None:
        """REQ-LEARN-002: Holdout energy lower than train (both learned) — not flagged."""
        train = [0.5, 0.5, 0.5, 0.5, 0.5]
        holdout = [0.1, 0.1, 0.1, 0.1, 0.1]
        report = audit_energy_trajectory(train, holdout)
        div_findings = [f for f in report.findings if isinstance(f, TrainHoldoutDivergenceFinding)]
        assert len(div_findings) == 0

    def test_custom_min_gap_respected(self) -> None:
        """REQ-LEARN-002: Custom min_gap parameter controls divergence sensitivity."""
        # gap = 0.3; default threshold=0.05 would flag, but min_gap=0.5 should not.
        train = [0.1, 0.1, 0.1]
        holdout = [0.4, 0.4, 0.4]
        report = audit_energy_trajectory(train, holdout, min_gap=0.5)
        div_findings = [f for f in report.findings if isinstance(f, TrainHoldoutDivergenceFinding)]
        assert len(div_findings) == 0

    def test_custom_distinct_values_min_respected(self) -> None:
        """REQ-LEARN-002: distinct_values_min=3 flags sequences with only 2 distinct values."""
        # 2 distinct values in each; default threshold=2 would allow it, but 3 flags it.
        train = [0.1, 0.2, 0.1, 0.2, 0.1]
        holdout = [0.3, 0.4, 0.3, 0.4, 0.3]
        report = audit_energy_trajectory(train, holdout, distinct_values_min=3)
        zero_findings = [f for f in report.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_findings) == 1

    def test_fp_noise_not_treated_as_distinct(self) -> None:
        """REQ-LEARN-002: Values differing by <1e-6 collapse to same bucket."""
        # Values differ only at 7th decimal place — should be 1 distinct value.
        v = 0.5000001
        train = [0.5, v, 0.5, v, 0.5]
        holdout = [0.1, 0.2, 0.3, 0.4, 0.5]
        report = audit_energy_trajectory(train, holdout)
        zero_findings = [f for f in report.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_findings) == 1  # train has effectively 1 distinct value

    def test_both_shortcut_and_divergence_reported(self) -> None:
        """REQ-LEARN-002: Multiple findings can coexist in same report."""
        # Constant training (shortcut) AND big divergence.
        train = [0.0, 0.0, 0.0, 0.0, 0.0]
        holdout = [0.9, 0.9, 0.9, 0.9, 0.9]
        report = audit_energy_trajectory(train, holdout)
        kinds = {type(f).__name__ for f in report.findings}
        # shortcut: train is constant; divergence: holdout >> train.
        assert "ZeroEnergyFinding" in kinds
        assert "TrainHoldoutDivergenceFinding" in kinds

    def test_custom_min_samples_respected(self) -> None:
        """REQ-LEARN-002: Custom min_samples allows shorter sequences to be checked."""
        # Only 2 samples per sequence; default min_samples=3 would skip.
        # With min_samples=2, constant sequences should be flagged.
        train = [0.0, 0.0]
        holdout = [0.9, 0.9]
        report = audit_energy_trajectory(train, holdout, min_samples=2)
        zero_findings = [f for f in report.findings if isinstance(f, ZeroEnergyFinding)]
        assert len(zero_findings) == 1


# ---------------------------------------------------------------------------
# REQ-LEARN-002: audit_full()
# ---------------------------------------------------------------------------


class TestAuditFull:
    """REQ-LEARN-002: Combined full audit merges tracker and energy findings."""

    def _clean_tracker(self) -> ConstraintTracker:
        """Build a tracker with no reward hacking signals."""
        tracker = ConstraintTracker()
        for _ in range(10):
            tracker.record("arithmetic", fired=True, caught_error=True, any_error_in_batch=True)
        return tracker

    def test_clean_inputs_produce_clean_report(self) -> None:
        """SCENARIO-LEARN-002: No hacking signals — fully clean report."""
        tracker = self._clean_tracker()
        train = [0.1, 0.2, 0.3, 0.4, 0.5]
        holdout = [0.15, 0.25, 0.35, 0.45, 0.55]
        report = audit_full(tracker, train, holdout)
        assert report.clean is True

    def test_findings_from_both_sources_combined(self) -> None:
        """SCENARIO-LEARN-002: Tracker finding + energy finding both appear in report."""
        # Build hacked tracker (trivial constraint).
        tracker = ConstraintTracker()
        for _ in range(20):
            tracker.record("noise", fired=True, caught_error=False, any_error_in_batch=True)

        # Also add diverging energy.
        train = [0.05, 0.05, 0.05, 0.05, 0.05]
        holdout = [0.9, 0.9, 0.9, 0.9, 0.9]

        report = audit_full(tracker, train, holdout)
        assert report.clean is False
        kinds = {type(f).__name__ for f in report.findings}
        assert "TrivialConstraintFinding" in kinds
        assert "TrainHoldoutDivergenceFinding" in kinds

    def test_tracker_only_finding_reported(self) -> None:
        """REQ-LEARN-002: Tracker finding with clean energy — only tracker finding."""
        tracker = ConstraintTracker()
        for _ in range(20):
            tracker.record("noise", fired=True, caught_error=False, any_error_in_batch=True)

        train = [0.1, 0.2, 0.3, 0.4, 0.5]
        holdout = [0.15, 0.25, 0.35, 0.45, 0.55]

        report = audit_full(tracker, train, holdout)
        assert report.clean is False
        div_findings = [f for f in report.findings if isinstance(f, TrainHoldoutDivergenceFinding)]
        assert len(div_findings) == 0

    def test_energy_only_finding_reported(self) -> None:
        """REQ-LEARN-002: Clean tracker + gaming energy — only energy finding."""
        tracker = self._clean_tracker()
        train = [0.05, 0.05, 0.05, 0.05, 0.05]
        holdout = [0.9, 0.9, 0.9, 0.9, 0.9]

        report = audit_full(tracker, train, holdout)
        assert report.clean is False
        trivial = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        assert len(trivial) == 0

    def test_kwargs_forwarded_to_subfunctions(self) -> None:
        """REQ-LEARN-002: min_fire_count and min_gap kwargs are forwarded."""
        # Trivial tracker with only 4 fires; default threshold=5 would skip.
        tracker = ConstraintTracker()
        for _ in range(4):
            tracker.record("noise", fired=True, caught_error=False, any_error_in_batch=True)

        train = [0.1, 0.2, 0.3]
        holdout = [0.2, 0.3, 0.4]

        # With min_fire_count=4, should flag trivial.
        report = audit_full(tracker, train, holdout, min_fire_count=4)
        trivial = [f for f in report.findings if isinstance(f, TrivialConstraintFinding)]
        assert len(trivial) == 1

    def test_short_energy_sequences_not_flagged(self) -> None:
        """REQ-LEARN-002: Energy sequences below min_samples skip checks."""
        tracker = self._clean_tracker()
        report = audit_full(tracker, [0.0], [1.0])
        # No energy findings — sequences too short.
        energy_findings = [
            f for f in report.findings
            if isinstance(f, (ZeroEnergyFinding, TrainHoldoutDivergenceFinding))
        ]
        assert len(energy_findings) == 0


# ---------------------------------------------------------------------------
# REQ-LEARN-002: Exported constants are accessible
# ---------------------------------------------------------------------------


class TestExportedConstants:
    """REQ-LEARN-002: Public constants have expected types and sensible values."""

    def test_min_fire_count_is_positive_int(self) -> None:
        assert isinstance(MIN_FIRE_COUNT_FOR_TRIVIAL_FLAG, int)
        assert MIN_FIRE_COUNT_FOR_TRIVIAL_FLAG > 0

    def test_trivial_precision_threshold_in_range(self) -> None:
        assert 0.0 < TRIVIAL_PRECISION_THRESHOLD < 1.0

    def test_energy_distinct_values_min_at_least_two(self) -> None:
        assert ENERGY_DISTINCT_VALUES_MIN >= 2

    def test_gini_threshold_in_range(self) -> None:
        assert 0.0 < GINI_DIVERSITY_THRESHOLD < 1.0

    def test_divergence_min_gap_positive(self) -> None:
        assert DIVERGENCE_MIN_GAP > 0.0

    def test_min_energy_samples_at_least_two(self) -> None:
        assert MIN_ENERGY_SAMPLES >= 2
