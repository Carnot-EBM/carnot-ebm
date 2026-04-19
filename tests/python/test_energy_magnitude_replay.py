"""Tests for EnergyMagnitudeBuffer and EnergyMagnitudeReplay (RETRO-050, Exp 509).

Spec: REQ-LEARN-043, REQ-LEARN-044, REQ-LEARN-045,
      SCENARIO-LEARN-071, SCENARIO-LEARN-072, SCENARIO-LEARN-073
"""

from __future__ import annotations

import pytest

from carnot.pipeline.energy_magnitude_replay import (
    EnergyMagnitudeBuffer,
    EnergyMagnitudeReplay,
)


# ---------------------------------------------------------------------------
# EnergyMagnitudeBuffer tests
# ---------------------------------------------------------------------------


class TestEnergyMagnitudeBuffer:
    """REQ-LEARN-044: per-domain sorted buffer ranked by |energy - mean|."""

    def test_empty_buffer_mean_is_zero(self):
        """mean_energy returns 0.0 for empty buffer (SCENARIO-LEARN-071)."""
        buf = EnergyMagnitudeBuffer("arithmetic")
        assert buf.mean_energy == 0.0

    def test_mean_energy_single_item(self):
        """Mean equals the single added energy."""
        buf = EnergyMagnitudeBuffer("arithmetic")
        buf.add({"type": "add"}, 3.0)
        assert buf.mean_energy == pytest.approx(3.0)

    def test_mean_energy_multiple_items(self):
        """Mean is updated correctly with Welford's algorithm."""
        buf = EnergyMagnitudeBuffer("arithmetic")
        buf.add({"a": 1}, 2.0)
        buf.add({"a": 2}, 4.0)
        assert buf.mean_energy == pytest.approx(3.0)

    def test_top_k_returns_highest_deviation_first(self):
        """SCENARIO-LEARN-071: top_k returns highest |energy - mean| violations first."""
        buf = EnergyMagnitudeBuffer("arithmetic", max_size=10)
        v1 = {"id": 1}
        v2 = {"id": 2}
        v3 = {"id": 3}
        v4 = {"id": 4}
        buf.add(v1, 1.0)   # mean after: 1.0,  dev: 0.0
        buf.add(v2, 5.0)   # mean after: 3.0,  dev: 2.0
        buf.add(v3, 2.0)   # mean after: 2.67, dev: 0.67
        buf.add(v4, 4.0)   # mean after: 3.0,  dev: 1.0

        result = buf.top_k(2)
        assert len(result) == 2
        # v2 (energy=5, dev=2.0) and v4 or v1 should be top 2
        assert v2 in result

    def test_top_k_fewer_than_k_items(self):
        """top_k returns all items when buffer has fewer than k entries."""
        buf = EnergyMagnitudeBuffer("code", max_size=10)
        buf.add({"x": 1}, 1.0)
        buf.add({"x": 2}, 2.0)
        result = buf.top_k(5)
        assert len(result) == 2

    def test_top_k_empty_buffer(self):
        """top_k on empty buffer returns empty list."""
        buf = EnergyMagnitudeBuffer("logical")
        assert buf.top_k(3) == []

    def test_buffer_max_size_evicts_lowest_deviation(self):
        """When full, lowest-deviation item is evicted, not oldest (REQ-LEARN-044)."""
        buf = EnergyMagnitudeBuffer("arithmetic", max_size=2)
        v_easy = {"id": "easy"}
        v_hard = {"id": "hard"}
        v_harder = {"id": "harder"}

        buf.add(v_easy, 1.0)   # mean=1.0, dev=0.0
        buf.add(v_hard, 3.0)   # mean=2.0, dev=1.0 (hard)
        # Buffer now has v_easy (dev=?) and v_hard (dev=1.0) - both in
        # Now add a harder item that should evict the easiest:
        buf.add(v_harder, 10.0)  # mean ~4.67, dev large

        result = buf.top_k(2)
        # v_harder should be in result (very high deviation)
        assert v_harder in result

    def test_buffer_size_two_keeps_harder_items(self):
        """Buffer of size 2 retains the 2 highest-deviation items."""
        buf = EnergyMagnitudeBuffer("code", max_size=2)
        v_low = {"id": "low"}
        v_high = {"id": "high"}
        v_mid = {"id": "mid"}

        buf.add(v_low, 1.0)
        buf.add(v_high, 9.0)
        buf.add(v_mid, 5.0)

        result = buf.top_k(10)
        # Should have 2 items, v_high should be in there (highest deviation)
        assert len(result) == 2
        assert v_high in result

    def test_add_updates_count(self):
        """Running mean count increments correctly."""
        buf = EnergyMagnitudeBuffer("logical")
        buf.add({}, 1.0)
        buf.add({}, 3.0)
        buf.add({}, 5.0)
        assert buf.mean_energy == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# EnergyMagnitudeReplay tests
# ---------------------------------------------------------------------------


class TestEnergyMagnitudeReplay:
    """REQ-LEARN-043, REQ-LEARN-045: multi-domain replay with isolation score."""

    def test_construction_with_domains(self):
        """EnergyMagnitudeReplay initializes with given domains."""
        emr = EnergyMagnitudeReplay(domains=["arithmetic", "code", "logical"], k=10)
        assert emr.k == 10

    def test_add_violation_known_domain(self):
        """add_violation for a known domain is recorded."""
        emr = EnergyMagnitudeReplay(domains=["arithmetic"], k=5)
        emr.add_violation("arithmetic", {"type": "add"}, 2.0)
        result = emr.get_replay_batch("arithmetic")
        assert len(result) == 1

    def test_add_violation_unknown_domain_ignored(self):
        """add_violation for unknown domain does not raise."""
        emr = EnergyMagnitudeReplay(domains=["arithmetic"], k=5)
        emr.add_violation("unknown_domain", {"x": 1}, 3.0)  # no error
        assert emr.get_replay_batch("unknown_domain") == []

    def test_get_replay_batch_returns_top_k(self):
        """get_replay_batch returns up to k items."""
        emr = EnergyMagnitudeReplay(domains=["arithmetic"], k=3)
        for i in range(10):
            emr.add_violation("arithmetic", {"id": i}, float(i))
        result = emr.get_replay_batch("arithmetic")
        assert len(result) <= 3

    def test_get_replay_batch_unknown_domain_returns_empty(self):
        """get_replay_batch for unknown domain returns []."""
        emr = EnergyMagnitudeReplay(domains=["arithmetic"], k=5)
        assert emr.get_replay_batch("nope") == []

    def test_isolation_score_in_range(self):
        """SCENARIO-LEARN-072: isolation_score returns float in [-1.0, 1.0]."""
        emr = EnergyMagnitudeReplay(domains=["arithmetic", "code"], k=5)
        for i in range(10):
            emr.add_violation("arithmetic", {"arith_key": i}, float(i))
            emr.add_violation("code", {"code_key": i}, float(i * 2))
        score = emr.isolation_score("arithmetic", "code", n_steps=20)
        assert -1.0 <= score <= 1.0

    def test_isolation_score_disjoint_domains_is_one(self):
        """SCENARIO-LEARN-072: perfectly disjoint domains (no shared keys) → score=1.0."""
        emr = EnergyMagnitudeReplay(domains=["arithmetic", "code"], k=5)
        for i in range(10):
            emr.add_violation("arithmetic", {"arith_only": i}, float(i))
            emr.add_violation("code", {"code_only": i}, float(i))
        score = emr.isolation_score("arithmetic", "code", n_steps=10)
        assert score == pytest.approx(1.0)

    def test_isolation_score_fully_overlapping_domains_is_minus_one(self):
        """Domains with identical key structure → score=-1.0 (complete interference)."""
        emr = EnergyMagnitudeReplay(domains=["a", "b"], k=5)
        for i in range(10):
            emr.add_violation("a", {"shared_key": i}, float(i))
            emr.add_violation("b", {"shared_key": i + 10}, float(i))
        score = emr.isolation_score("a", "b", n_steps=10)
        assert score == pytest.approx(-1.0)

    def test_isolation_score_empty_buffer_returns_zero(self):
        """isolation_score with empty buffers returns 0.0 (neutral)."""
        emr = EnergyMagnitudeReplay(domains=["arithmetic", "code"], k=5)
        score = emr.isolation_score("arithmetic", "code", n_steps=10)
        assert score == pytest.approx(0.0)

    def test_isolation_score_unknown_domain_returns_zero(self):
        """isolation_score with unknown domain returns 0.0."""
        emr = EnergyMagnitudeReplay(domains=["arithmetic"], k=5)
        emr.add_violation("arithmetic", {"x": 1}, 2.0)
        score = emr.isolation_score("arithmetic", "unknown", n_steps=10)
        assert score == pytest.approx(0.0)

    def test_three_domains(self):
        """Multi-domain replay handles three domains correctly."""
        emr = EnergyMagnitudeReplay(
            domains=["arithmetic", "code", "logical"], k=10
        )
        for i in range(20):
            emr.add_violation("arithmetic", {"arith": i}, float(i))
            emr.add_violation("code", {"code": i}, float(i * 1.5))
            emr.add_violation("logical", {"logic": i}, float(i * 0.5))
        # All three domains should have replay batches
        assert len(emr.get_replay_batch("arithmetic")) > 0
        assert len(emr.get_replay_batch("code")) > 0
        assert len(emr.get_replay_batch("logical")) > 0
