"""Tests for LSEBMConstraintReplayer and ViolationDistribution.

Spec: REQ-SELFLEARN-013, REQ-SELFLEARN-014, REQ-SELFLEARN-015,
SCENARIO-SELFLEARN-013, SCENARIO-SELFLEARN-014, SCENARIO-SELFLEARN-015
"""

from __future__ import annotations

import pytest

from carnot.pipeline.lsebm_replayer import LSEBMConstraintReplayer, ViolationDistribution


# ---------------------------------------------------------------------------
# ViolationDistribution tests
# ---------------------------------------------------------------------------


class TestViolationDistribution:
    def test_vocab_is_sorted(self):
        # REQ-SELFLEARN-013: vocab order is deterministic for reproducibility.
        dist = ViolationDistribution(counts={"sign": 2, "carry": 5, "unit": 1})
        assert dist.vocab == ["carry", "sign", "unit"]

    def test_to_training_pairs_length(self):
        # Total pairs = sum of counts.
        dist = ViolationDistribution(counts={"carry": 3, "sign": 2})
        pairs = dist.to_training_pairs()
        assert len(pairs) == 5  # 3 + 2

    def test_to_training_pairs_vectors(self):
        # REQ-SELFLEARN-013: each pair is a one-hot vector + energy_target 0.0.
        dist = ViolationDistribution(counts={"carry": 2, "sign": 1})
        pairs = dist.to_training_pairs()
        # vocab = ['carry', 'sign']
        # carry → [1.0, 0.0], sign → [0.0, 1.0]
        carry_pairs = [p for p in pairs if p[0][0] == 1.0]
        sign_pairs = [p for p in pairs if p[0][1] == 1.0]
        assert len(carry_pairs) == 2
        assert len(sign_pairs) == 1
        for vec, energy in pairs:
            assert energy == 0.0
            assert sum(vec) == 1.0  # one-hot

    def test_to_training_pairs_empty(self):
        # Empty counts → empty pairs.
        dist = ViolationDistribution(counts={})
        assert dist.to_training_pairs() == []

    def test_most_common_basic(self):
        # REQ-SELFLEARN-013: most_common returns top-n by count.
        dist = ViolationDistribution(counts={"carry": 10, "sign": 3, "unit": 7})
        top2 = dist.most_common(2)
        assert top2[0] == ("carry", 10)
        assert top2[1] == ("unit", 7)

    def test_most_common_n_larger_than_vocab(self):
        # When n > len(vocab), return all items.
        dist = ViolationDistribution(counts={"carry": 1})
        result = dist.most_common(10)
        assert result == [("carry", 1)]

    def test_most_common_empty(self):
        dist = ViolationDistribution(counts={})
        assert dist.most_common(5) == []

    def test_to_training_pairs_single_type(self):
        # Single violation type → all pairs are [1.0] with energy 0.0.
        dist = ViolationDistribution(counts={"carry": 4})
        pairs = dist.to_training_pairs()
        assert len(pairs) == 4
        for vec, energy in pairs:
            assert vec == [1.0]
            assert energy == 0.0


# ---------------------------------------------------------------------------
# LSEBMConstraintReplayer tests
# ---------------------------------------------------------------------------


class TestLSEBMConstraintReplayer:
    def test_fit_and_generate_returns_known_types(self):
        # SCENARIO-SELFLEARN-013: generate only returns types seen during fit.
        replayer = LSEBMConstraintReplayer(n_replay=20, ebm_n_iter=100)
        violations = ["carry"] * 30 + ["sign"] * 10
        replayer.fit(violations)
        results = replayer.generate_replay(10)
        assert len(results) == 10
        known = {"carry", "sign"}
        for r in results:
            assert r in known, f"Unexpected violation type: {r!r}"

    def test_generate_without_fit_returns_empty(self):
        # generate_replay before fit should return empty list.
        replayer = LSEBMConstraintReplayer()
        assert replayer.generate_replay(10) == []

    def test_fit_empty_violations_then_generate_empty(self):
        # Fitting on empty list → no vocab → generate returns empty.
        replayer = LSEBMConstraintReplayer()
        replayer.fit([])
        assert replayer.generate_replay(5) == []

    def test_warm_start_returns_positive_count(self):
        # SCENARIO-SELFLEARN-014: warm_start returns count > 0 after fitting on carry session.
        replayer = LSEBMConstraintReplayer(n_replay=20, ebm_n_iter=50)
        violations = ["carry"] * 50
        replayer.fit(violations)

        class FakeMemory:
            pass

        memory = FakeMemory()
        count = replayer.warm_start(memory)
        assert count > 0
        assert hasattr(memory, "_warm_start_counts")
        assert isinstance(memory._warm_start_counts, dict)

    def test_warm_start_populates_counts(self):
        # Warm-start counts should only contain violation types from the EBM vocab.
        replayer = LSEBMConstraintReplayer(n_replay=20, ebm_n_iter=50)
        violations = ["carry"] * 40 + ["sign"] * 10
        replayer.fit(violations)

        class FakeMemory:
            pass

        memory = FakeMemory()
        replayer.warm_start(memory)
        for vtype in memory._warm_start_counts:
            assert vtype in {"carry", "sign"}
        total = sum(memory._warm_start_counts.values())
        assert total == 20  # n_replay samples

    def test_generate_correct_length(self):
        # generate_replay(n) always returns exactly n items.
        replayer = LSEBMConstraintReplayer(n_replay=5, ebm_n_iter=20)
        replayer.fit(["carry"] * 10)
        results = replayer.generate_replay(15)
        assert len(results) == 15

    def test_fit_single_type_generates_that_type(self):
        # When only one type is seen, all generated samples must be that type.
        replayer = LSEBMConstraintReplayer(n_replay=20, ebm_n_iter=100)
        replayer.fit(["carry"] * 50)
        results = replayer.generate_replay(20)
        assert len(results) == 20
        assert all(r == "carry" for r in results)

    def test_warm_start_without_fit_returns_zero(self):
        # warm_start before fit → empty replay → zero count.
        replayer = LSEBMConstraintReplayer()

        class FakeMemory:
            pass

        memory = FakeMemory()
        count = replayer.warm_start(memory)
        assert count == 0
        assert memory._warm_start_counts == {}

    def test_violation_distribution_integration(self):
        # ViolationDistribution.to_training_pairs matches what fit would use.
        dist = ViolationDistribution(counts={"carry": 3, "sign": 1})
        pairs = dist.to_training_pairs()
        assert len(pairs) == 4

        replayer = LSEBMConstraintReplayer(n_replay=10, ebm_n_iter=10)
        replayer.fit(["carry", "carry", "carry", "sign"])
        results = replayer.generate_replay(5)
        assert len(results) == 5
        for r in results:
            assert r in {"carry", "sign"}
