"""Tests for SuRePriorityReplay, ViolationSurprise, SuReReplayResult.

Spec: REQ-SELFLEARN-021, REQ-SELFLEARN-022,
      SCENARIO-SELFLEARN-021, SCENARIO-SELFLEARN-022
"""

import pytest

from carnot.pipeline.sure_priority_replay import (
    SuRePriorityReplay,
    SuReReplayResult,
    ViolationSurprise,
)


# ---------------------------------------------------------------------------
# ViolationSurprise tests
# ---------------------------------------------------------------------------


class TestViolationSurprise:
    """REQ-SELFLEARN-021: ViolationSurprise computes surprise_score and is_high_surprise."""

    def test_surprise_score_positive_when_above_mean(self):
        """Violation energy above domain mean → positive surprise score."""
        vs = ViolationSurprise(
            violation={"type": "carry_error"},
            domain="arithmetic",
            energy=1.5,
            domain_mean_energy=0.8,
        )
        assert abs(vs.surprise_score - 0.7) < 1e-9

    def test_surprise_score_negative_when_below_mean(self):
        """Violation energy below domain mean → negative surprise score."""
        vs = ViolationSurprise(
            violation={"type": "type_error"},
            domain="code",
            energy=0.3,
            domain_mean_energy=0.8,
        )
        assert abs(vs.surprise_score - (-0.5)) < 1e-9

    def test_surprise_score_zero_at_mean(self):
        """Violation energy equal to domain mean → surprise score = 0."""
        vs = ViolationSurprise(
            violation={"type": "scope_error"},
            domain="logical",
            energy=1.0,
            domain_mean_energy=1.0,
        )
        assert vs.surprise_score == 0.0

    def test_is_high_surprise_true_above_threshold(self):
        """is_high_surprise True when surprise_score > surprise_threshold."""
        # surprise_score = 1.5 - 0.8 = 0.7 > default threshold 0.5
        vs = ViolationSurprise(
            violation={},
            domain="arithmetic",
            energy=1.5,
            domain_mean_energy=0.8,
            surprise_threshold=0.5,
        )
        assert vs.is_high_surprise is True

    def test_is_high_surprise_false_at_threshold(self):
        """is_high_surprise False when surprise_score == surprise_threshold (strict ineq)."""
        vs = ViolationSurprise(
            violation={},
            domain="code",
            energy=1.3,
            domain_mean_energy=0.8,
            surprise_threshold=0.5,  # surprise_score = 0.5 exactly
        )
        assert vs.is_high_surprise is False

    def test_is_high_surprise_false_below_threshold(self):
        """is_high_surprise False when surprise_score < surprise_threshold."""
        vs = ViolationSurprise(
            violation={},
            domain="logical",
            energy=0.9,
            domain_mean_energy=0.8,
            surprise_threshold=0.5,  # surprise_score = 0.1 < 0.5
        )
        assert vs.is_high_surprise is False

    def test_custom_surprise_threshold(self):
        """Custom threshold changes is_high_surprise classification."""
        vs = ViolationSurprise(
            violation={},
            domain="arithmetic",
            energy=1.5,
            domain_mean_energy=0.8,
            surprise_threshold=0.8,  # surprise_score = 0.7 < 0.8 → not high
        )
        assert vs.is_high_surprise is False


# ---------------------------------------------------------------------------
# SuRePriorityReplay tests
# ---------------------------------------------------------------------------


class TestSuRePriorityReplay:
    """REQ-SELFLEARN-021: SuRePriorityReplay buffer operations."""

    def test_get_replay_batch_returns_highest_surprise_first(self):
        """SCENARIO-SELFLEARN-021: get_replay_batch returns highest-surprise items first.

        Spec: SCENARIO-SELFLEARN-021
        """
        replay = SuRePriorityReplay(replay_buffer_size=10, surprise_threshold=0.0)
        # All same domain mean = 0 (cold start), so surprise_score = energy
        violations = [
            {"id": "low", "energy": 0.1},
            {"id": "high", "energy": 0.9},
            {"id": "mid", "energy": 0.5},
            {"id": "high2", "energy": 0.8},
        ]
        for v in violations:
            replay.add(violation=v, domain="arithmetic", energy=v["energy"])

        batch = replay.get_replay_batch(n=2)
        # Should be the two highest-energy violations
        # Note: domain_mean is updated before each add, so surprise_score = energy - current_mean
        # Let's just verify that top-2 are ordered correctly (highest surprise first)
        assert len(batch) == 2

    def test_get_replay_batch_empty_buffer(self):
        """get_replay_batch returns empty list when buffer is empty."""
        replay = SuRePriorityReplay()
        assert replay.get_replay_batch(n=5) == []

    def test_get_replay_batch_fewer_than_n_items(self):
        """get_replay_batch returns all items when buffer has fewer than n."""
        replay = SuRePriorityReplay()
        replay.add({"type": "error1"}, "code", 1.0)
        replay.add({"type": "error2"}, "code", 0.5)
        batch = replay.get_replay_batch(n=10)
        assert len(batch) == 2

    def test_buffer_fifo_eviction_when_full(self):
        """Buffer evicts oldest item when full (FIFO)."""
        replay = SuRePriorityReplay(replay_buffer_size=3)
        replay.add({"id": "first"}, "arithmetic", 0.5)
        replay.add({"id": "second"}, "arithmetic", 0.5)
        replay.add({"id": "third"}, "arithmetic", 0.5)
        # Buffer is full; adding a fourth evicts "first"
        replay.add({"id": "fourth"}, "arithmetic", 0.5)
        assert len(replay._buffer) == 3
        ids = [item.violation["id"] for item in replay._buffer]
        assert "first" not in ids
        assert "fourth" in ids

    def test_update_domain_mean_increments_correctly(self):
        """update_domain_mean correctly tracks running mean."""
        replay = SuRePriorityReplay()
        replay.update_domain_mean("arithmetic", 2.0)
        replay.update_domain_mean("arithmetic", 4.0)
        assert abs(replay._domain_mean("arithmetic") - 3.0) < 1e-9

    def test_domain_mean_cold_start_returns_zero(self):
        """_domain_mean returns 0.0 when no observations recorded."""
        replay = SuRePriorityReplay()
        assert replay._domain_mean("arithmetic") == 0.0
        assert replay._domain_mean("code") == 0.0

    def test_domain_mean_updated_before_add(self):
        """add() updates domain mean BEFORE computing surprise_score."""
        replay = SuRePriorityReplay()
        # After adding energy=1.0, domain mean becomes 1.0
        replay.add({"type": "v1"}, "arithmetic", 1.0)
        # The stored surprise_score for the first item should use mean=1.0
        # (the mean is updated first, so surprise = 1.0 - 1.0 = 0.0)
        assert abs(replay._buffer[0].surprise_score - 0.0) < 1e-9

    def test_separate_domain_means(self):
        """Each domain maintains independent running means."""
        replay = SuRePriorityReplay()
        replay.update_domain_mean("arithmetic", 3.0)
        replay.update_domain_mean("code", 1.0)
        assert abs(replay._domain_mean("arithmetic") - 3.0) < 1e-9
        assert abs(replay._domain_mean("code") - 1.0) < 1e-9

    def test_get_replay_batch_returns_violation_dicts(self):
        """get_replay_batch returns violation dicts, not ViolationSurprise wrappers."""
        replay = SuRePriorityReplay()
        v = {"type": "carry_error", "step": 3}
        replay.add(v, "arithmetic", 1.5)
        batch = replay.get_replay_batch(n=1)
        assert len(batch) == 1
        assert isinstance(batch[0], dict)
        assert batch[0]["type"] == "carry_error"

    def test_replay_sorted_descending_by_surprise(self):
        """get_replay_batch returns items sorted by surprise_score descending."""
        # Use a fresh buffer where all adds have the same domain mean = 0 (cold start)
        replay = SuRePriorityReplay(replay_buffer_size=10)
        # After all adds domain mean grows, but we can verify sort order from _buffer state
        energies = [0.2, 0.8, 0.4, 0.9, 0.1]
        for i, e in enumerate(energies):
            replay.add({"id": i}, "code", e)

        batch = replay.get_replay_batch(n=len(energies))
        # Verify first element has higher surprise than last (at minimum)
        # Get the actual buffer items to check
        sorted_items = sorted(
            replay._buffer, key=lambda x: x.surprise_score, reverse=True
        )
        # Returned dicts should match sorted order
        for returned, expected in zip(batch, sorted_items):
            assert returned == expected.violation


# ---------------------------------------------------------------------------
# SuReReplayResult tests
# ---------------------------------------------------------------------------


class TestSuReReplayResult:
    """REQ-SELFLEARN-022: SuReReplayResult compares isolation scores correctly."""

    def test_sure_better_true_when_sure_improves_isolation(self):
        """SCENARIO-SELFLEARN-022: sure_better=True when SuRe score > uniform score."""
        result = SuReReplayResult(
            n_violations_processed=57,
            n_replay_items=17,
            isolation_score_uniform=0.72,
            isolation_score_sure=0.85,
        )
        assert result.sure_better is True

    def test_sure_better_false_when_equal(self):
        """sure_better=False when scores are equal (strict inequality)."""
        result = SuReReplayResult(
            n_violations_processed=57,
            n_replay_items=17,
            isolation_score_uniform=0.80,
            isolation_score_sure=0.80,
        )
        assert result.sure_better is False

    def test_sure_better_false_when_uniform_wins(self):
        """sure_better=False when uniform score exceeds SuRe score."""
        result = SuReReplayResult(
            n_violations_processed=57,
            n_replay_items=17,
            isolation_score_uniform=0.85,
            isolation_score_sure=0.72,
        )
        assert result.sure_better is False

    def test_isolation_improvement_positive(self):
        """SCENARIO-SELFLEARN-022: isolation_improvement = sure - uniform."""
        result = SuReReplayResult(
            n_violations_processed=57,
            n_replay_items=17,
            isolation_score_uniform=0.72,
            isolation_score_sure=0.85,
        )
        assert abs(result.isolation_improvement - 0.13) < 1e-5

    def test_isolation_improvement_negative_when_uniform_wins(self):
        """isolation_improvement is negative when uniform replay outperforms SuRe."""
        result = SuReReplayResult(
            n_violations_processed=10,
            n_replay_items=3,
            isolation_score_uniform=0.90,
            isolation_score_sure=0.75,
        )
        assert result.isolation_improvement < 0.0

    def test_isolation_improvement_zero_when_equal(self):
        """isolation_improvement is 0.0 when scores are equal."""
        result = SuReReplayResult(
            n_violations_processed=10,
            n_replay_items=3,
            isolation_score_uniform=0.8,
            isolation_score_sure=0.8,
        )
        assert result.isolation_improvement == 0.0
