"""Tests for MetaJuLSAdapter and ExtractorPolicy.

Every test traces to a specific REQ-* or SCENARIO-* from the autoresearch spec.

Spec coverage:
    REQ-LEARN-078  — policy adapts from live batch feedback
    REQ-LEARN-079  — precision trend is non-negative after adaptation
    SCENARIO-LEARN-121 — low precision → temperature down, threshold up
    SCENARIO-LEARN-122 — high precision → temperature up, threshold down
    SCENARIO-LEARN-123 — trend >= 0 after two improving batches
"""

from __future__ import annotations

import pytest

from carnot.pipeline.metajuls_adapter import ExtractorPolicy, MetaJuLSAdapter


# ---------------------------------------------------------------------------
# ExtractorPolicy tests
# ---------------------------------------------------------------------------


class TestExtractorPolicy:
    """Tests for ExtractorPolicy dataclass defaults and serialisation."""

    def test_default_temperature(self):
        # REQ-LEARN-078: policy has sensible defaults so adapter can start cold.
        policy = ExtractorPolicy()
        assert policy.temperature == 0.1

    def test_default_threshold(self):
        # REQ-LEARN-078
        policy = ExtractorPolicy()
        assert policy.claim_confidence_threshold == 0.5

    def test_default_strategy_weights_keys(self):
        # REQ-LEARN-078: all three strategies must be represented.
        policy = ExtractorPolicy()
        assert set(policy.strategy_weights.keys()) == {"json", "symcode", "chain"}

    def test_to_dict_round_trip(self):
        # REQ-LEARN-078: artifact embedding requires JSON-serialisable dict.
        policy = ExtractorPolicy(
            temperature=0.2,
            claim_confidence_threshold=0.6,
            strategy_weights={"json": 0.5, "symcode": 0.3, "chain": 0.2},
        )
        d = policy.to_dict()
        assert d["temperature"] == 0.2
        assert d["claim_confidence_threshold"] == 0.6
        assert d["strategy_weights"] == {"json": 0.5, "symcode": 0.3, "chain": 0.2}

    def test_to_dict_strategy_weights_is_copy(self):
        # Mutation of the dict returned by to_dict must not affect the policy.
        policy = ExtractorPolicy()
        d = policy.to_dict()
        d["strategy_weights"]["json"] = 999.0
        assert policy.strategy_weights["json"] != 999.0


# ---------------------------------------------------------------------------
# MetaJuLSAdapter initialisation tests
# ---------------------------------------------------------------------------


class TestMetaJuLSAdapterInit:
    """Tests for MetaJuLSAdapter default construction."""

    def test_default_policy_is_extractorpolicy(self):
        # REQ-LEARN-078: adapter starts with a valid policy.
        adapter = MetaJuLSAdapter()
        assert isinstance(adapter.policy, ExtractorPolicy)

    def test_default_experience_empty(self):
        # REQ-LEARN-078: no experience before first batch.
        adapter = MetaJuLSAdapter()
        assert adapter.experience == []

    def test_custom_policy_preserved(self):
        # REQ-LEARN-078: caller can inject an initial policy.
        policy = ExtractorPolicy(temperature=0.3)
        adapter = MetaJuLSAdapter(initial_policy=policy)
        assert adapter.policy.temperature == 0.3

    def test_default_strategy_weights_sum_near_one(self):
        # Sanity check: default weights sum to ~1.0.
        adapter = MetaJuLSAdapter()
        total = sum(adapter.policy.strategy_weights.values())
        assert abs(total - 1.0) < 0.01


# ---------------------------------------------------------------------------
# update_from_batch: low precision path (SCENARIO-LEARN-121)
# ---------------------------------------------------------------------------


class TestUpdateFromBatchLowPrecision:
    """SCENARIO-LEARN-121: low precision → conservative policy update."""

    def _low_precision_batch(self) -> list[dict]:
        # FP-heavy batch: extractor fires but most are correct responses.
        # 1 TP (violation_detected=True, true_label='incorrect')
        # 4 FP (violation_detected=True, true_label='correct')
        # precision = 1 / (1 + 4 + 1e-9) ≈ 0.20  → < 0.5
        return [
            {"response": "r0", "violation_detected": True, "true_label": "incorrect"},
            {"response": "r1", "violation_detected": True, "true_label": "correct"},
            {"response": "r2", "violation_detected": True, "true_label": "correct"},
            {"response": "r3", "violation_detected": True, "true_label": "correct"},
            {"response": "r4", "violation_detected": True, "true_label": "correct"},
            {"response": "r5", "violation_detected": False, "true_label": "correct"},
        ]

    def test_temperature_decreases(self):
        # SCENARIO-LEARN-121
        adapter = MetaJuLSAdapter()
        initial_temp = adapter.policy.temperature
        adapter.update_from_batch(self._low_precision_batch())
        assert adapter.policy.temperature < initial_temp

    def test_temperature_floor_respected(self):
        # Temperature must not go below 0.01.
        adapter = MetaJuLSAdapter(initial_policy=ExtractorPolicy(temperature=0.01))
        adapter.update_from_batch(self._low_precision_batch())
        assert adapter.policy.temperature >= 0.01

    def test_threshold_increases(self):
        # SCENARIO-LEARN-121
        adapter = MetaJuLSAdapter()
        initial_threshold = adapter.policy.claim_confidence_threshold
        adapter.update_from_batch(self._low_precision_batch())
        assert adapter.policy.claim_confidence_threshold > initial_threshold

    def test_threshold_ceiling_respected(self):
        # Threshold must not exceed 0.9.
        adapter = MetaJuLSAdapter(
            initial_policy=ExtractorPolicy(claim_confidence_threshold=0.9)
        )
        adapter.update_from_batch(self._low_precision_batch())
        assert adapter.policy.claim_confidence_threshold <= 0.9

    def test_experience_appended(self):
        # REQ-LEARN-078: experience grows by one entry per batch.
        adapter = MetaJuLSAdapter()
        adapter.update_from_batch(self._low_precision_batch())
        assert len(adapter.experience) == 1
        assert "batch_id" in adapter.experience[0]
        assert "precision" in adapter.experience[0]

    def test_experience_batch_id_is_zero_for_first_batch(self):
        adapter = MetaJuLSAdapter()
        adapter.update_from_batch(self._low_precision_batch())
        assert adapter.experience[0]["batch_id"] == 0

    def test_returns_policy(self):
        # update_from_batch must return the updated policy.
        adapter = MetaJuLSAdapter()
        result = adapter.update_from_batch(self._low_precision_batch())
        assert isinstance(result, ExtractorPolicy)


# ---------------------------------------------------------------------------
# update_from_batch: high precision path (SCENARIO-LEARN-122)
# ---------------------------------------------------------------------------


class TestUpdateFromBatchHighPrecision:
    """SCENARIO-LEARN-122: high precision → relaxed policy update."""

    def _high_precision_batch(self) -> list[dict]:
        # 5 TP, 0 FP → precision ≈ 1.0  → > 0.8
        return [
            {"response": f"r{i}", "violation_detected": True, "true_label": "incorrect"}
            for i in range(5)
        ] + [
            {"response": f"r{i+5}", "violation_detected": False, "true_label": "correct"}
            for i in range(3)
        ]

    def test_temperature_increases(self):
        # SCENARIO-LEARN-122
        adapter = MetaJuLSAdapter()
        initial_temp = adapter.policy.temperature
        adapter.update_from_batch(self._high_precision_batch())
        assert adapter.policy.temperature > initial_temp

    def test_temperature_ceiling_respected(self):
        # Temperature must not exceed 0.5.
        adapter = MetaJuLSAdapter(initial_policy=ExtractorPolicy(temperature=0.5))
        adapter.update_from_batch(self._high_precision_batch())
        assert adapter.policy.temperature <= 0.5

    def test_threshold_decreases(self):
        # SCENARIO-LEARN-122
        adapter = MetaJuLSAdapter()
        initial_threshold = adapter.policy.claim_confidence_threshold
        adapter.update_from_batch(self._high_precision_batch())
        assert adapter.policy.claim_confidence_threshold < initial_threshold

    def test_threshold_floor_respected(self):
        # Threshold must not go below 0.3.
        adapter = MetaJuLSAdapter(
            initial_policy=ExtractorPolicy(claim_confidence_threshold=0.3)
        )
        adapter.update_from_batch(self._high_precision_batch())
        assert adapter.policy.claim_confidence_threshold >= 0.3


# ---------------------------------------------------------------------------
# update_from_batch: neutral precision path (no update expected)
# ---------------------------------------------------------------------------


class TestUpdateFromBatchNeutralPrecision:
    """Precision in [0.5, 0.8] must leave policy unchanged."""

    def _neutral_precision_batch(self) -> list[dict]:
        # 3 TP, 2 FP → precision ≈ 0.6  → in [0.5, 0.8]
        return [
            {"response": f"r{i}", "violation_detected": True, "true_label": "incorrect"}
            for i in range(3)
        ] + [
            {"response": f"r{i+3}", "violation_detected": True, "true_label": "correct"}
            for i in range(2)
        ]

    def test_temperature_unchanged(self):
        adapter = MetaJuLSAdapter()
        initial_temp = adapter.policy.temperature
        adapter.update_from_batch(self._neutral_precision_batch())
        assert adapter.policy.temperature == initial_temp

    def test_threshold_unchanged(self):
        adapter = MetaJuLSAdapter()
        initial_threshold = adapter.policy.claim_confidence_threshold
        adapter.update_from_batch(self._neutral_precision_batch())
        assert adapter.policy.claim_confidence_threshold == initial_threshold

    def test_experience_still_appended(self):
        # Even a hold-steady batch must record the precision.
        adapter = MetaJuLSAdapter()
        adapter.update_from_batch(self._neutral_precision_batch())
        assert len(adapter.experience) == 1


# ---------------------------------------------------------------------------
# update_from_batch: empty batch edge case
# ---------------------------------------------------------------------------


class TestUpdateFromBatchEmpty:
    """Empty batch should not crash; precision will be ~0 → conservative update."""

    def test_empty_batch_does_not_raise(self):
        adapter = MetaJuLSAdapter()
        # Should not raise even with no data.
        adapter.update_from_batch([])

    def test_empty_batch_appends_experience(self):
        adapter = MetaJuLSAdapter()
        adapter.update_from_batch([])
        assert len(adapter.experience) == 1

    def test_empty_batch_precision_near_zero(self):
        adapter = MetaJuLSAdapter()
        adapter.update_from_batch([])
        # tp=0, fp=0 → precision = 0 / (0 + 0 + 1e-9) ≈ 0
        assert adapter.experience[0]["precision"] < 0.5

    def test_empty_batch_temperature_decreases(self):
        # Precision ≈ 0 < 0.5 → conservative update applied.
        adapter = MetaJuLSAdapter()
        initial_temp = adapter.policy.temperature
        adapter.update_from_batch([])
        assert adapter.policy.temperature < initial_temp


# ---------------------------------------------------------------------------
# precision_trend tests (REQ-LEARN-079, SCENARIO-LEARN-123)
# ---------------------------------------------------------------------------


class TestPrecisionTrend:
    """Tests for the precision_trend() method."""

    def test_returns_zero_with_no_experience(self):
        # REQ-LEARN-079: undefined trend before any batch.
        adapter = MetaJuLSAdapter()
        assert adapter.precision_trend() == 0.0

    def test_returns_zero_with_one_batch(self):
        # REQ-LEARN-079: need at least 2 batches for a slope.
        adapter = MetaJuLSAdapter()
        adapter.experience.append({"batch_id": 0, "precision": 0.5})
        assert adapter.precision_trend() == 0.0

    def test_positive_trend_when_improving(self):
        # SCENARIO-LEARN-123: second batch precision > first batch precision.
        adapter = MetaJuLSAdapter()
        adapter.experience.append({"batch_id": 0, "precision": 0.4})
        adapter.experience.append({"batch_id": 1, "precision": 0.7})
        assert adapter.precision_trend() > 0.0

    def test_zero_trend_when_flat(self):
        # SCENARIO-LEARN-123: equal precision batches → trend = 0.
        adapter = MetaJuLSAdapter()
        adapter.experience.append({"batch_id": 0, "precision": 0.6})
        adapter.experience.append({"batch_id": 1, "precision": 0.6})
        assert adapter.precision_trend() == pytest.approx(0.0)

    def test_negative_trend_when_degrading(self):
        adapter = MetaJuLSAdapter()
        adapter.experience.append({"batch_id": 0, "precision": 0.9})
        adapter.experience.append({"batch_id": 1, "precision": 0.4})
        assert adapter.precision_trend() < 0.0

    def test_uses_last_three_batches_only(self):
        # Trend is computed over the most recent 3 batches, not all history.
        adapter = MetaJuLSAdapter()
        # Old batches with decreasing precision.
        for i in range(5):
            adapter.experience.append({"batch_id": i, "precision": 1.0 - i * 0.1})
        # Recent 3: precision[2]=0.8, precision[3]=0.7, precision[4]=0.6 → decreasing
        # last 3 in experience: indices 2,3,4 → first=0.8, last=0.6 → trend < 0
        assert adapter.precision_trend() < 0.0

    def test_trend_with_exactly_three_batches(self):
        # Ensure 3-batch window works correctly end-to-end.
        adapter = MetaJuLSAdapter()
        adapter.experience.append({"batch_id": 0, "precision": 0.3})
        adapter.experience.append({"batch_id": 1, "precision": 0.5})
        adapter.experience.append({"batch_id": 2, "precision": 0.7})
        # trend = 0.7 - 0.3 = 0.4
        assert adapter.precision_trend() == pytest.approx(0.4)

    def test_batch_id_increments_per_batch(self):
        # Batch IDs must be sequential: 0, 1, 2, ...
        adapter = MetaJuLSAdapter()
        batch = [
            {"response": "x", "violation_detected": False, "true_label": "correct"}
        ]
        adapter.update_from_batch(batch)
        adapter.update_from_batch(batch)
        adapter.update_from_batch(batch)
        assert [e["batch_id"] for e in adapter.experience] == [0, 1, 2]
