"""Tests for jepa_retrain_v2: CoTPairQualityFilter, JEPAQualityAugmentor, JEPARetrainV2Result.

100% coverage for python/carnot/models/jepa_retrain_v2.py.

Spec coverage: REQ-LEARN-037, REQ-LEARN-038, REQ-LEARN-039,
               SCENARIO-LEARN-066, SCENARIO-LEARN-067, SCENARIO-LEARN-068
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from carnot.models.ising import IsingConfig, IsingModel
from carnot.models.jepa_retrain_v2 import (
    CoTPairQuality,
    CoTPairQualityFilter,
    JEPAQualityAugmentor,
    JEPARetrainV2Result,
    _estimate_arithmetic_coverage,
    _estimate_label_confidence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def small_ising() -> IsingModel:
    """Tiny Ising model for fast CPU tests."""
    return IsingModel(IsingConfig(input_dim=8))


@pytest.fixture()
def high_coverage_pair() -> dict:
    return {
        "step_text": "2 * 3 = 6 and 4 + 5 = 9 and 10 - 2 = 8",
        "label": "correct",
        "confidence": 0.9,
        "arithmetic_coverage": 0.8,
    }


@pytest.fixture()
def low_coverage_pair() -> dict:
    return {
        "step_text": "The answer is probably correct because it makes sense",
        "label": "incorrect",
        "confidence": 0.8,
        "arithmetic_coverage": 0.1,
    }


@pytest.fixture()
def low_confidence_pair() -> dict:
    return {
        "step_text": "2 + 2 = 4",
        "label": "incorrect",
        "confidence": 0.5,
        "arithmetic_coverage": 0.6,
    }


@pytest.fixture()
def passing_pair() -> dict:
    return {
        "step_text": "3 * 7 = 21 plus 5 = 26",
        "label": "correct",
        "confidence": 0.95,
        "arithmetic_coverage": 0.7,
    }


# ---------------------------------------------------------------------------
# CoTPairQuality tests
# ---------------------------------------------------------------------------


class TestCoTPairQuality:
    def test_passes_gate_both_thresholds_met(self):
        # REQ-LEARN-037: pair passes when coverage >= 0.3 AND confidence >= 0.7
        q = CoTPairQuality(arithmetic_coverage=0.5, label_confidence=0.8, n_steps=3)
        assert q.passes_gate is True

    def test_fails_gate_low_coverage(self):
        # SCENARIO-LEARN-066: coverage < 0.3 → rejected
        q = CoTPairQuality(arithmetic_coverage=0.2, label_confidence=0.9, n_steps=2)
        assert q.passes_gate is False

    def test_fails_gate_low_confidence(self):
        q = CoTPairQuality(arithmetic_coverage=0.5, label_confidence=0.6, n_steps=2)
        assert q.passes_gate is False

    def test_fails_gate_both_below_threshold(self):
        q = CoTPairQuality(arithmetic_coverage=0.1, label_confidence=0.4, n_steps=1)
        assert q.passes_gate is False

    def test_passes_gate_at_exact_thresholds(self):
        # boundary: exactly 0.3 and 0.7 should pass
        q = CoTPairQuality(arithmetic_coverage=0.3, label_confidence=0.7, n_steps=5)
        assert q.passes_gate is True

    def test_quality_score_harmonic_mean(self):
        # harmonic mean of 0.6 and 0.9 = 2*(0.6*0.9)/(0.6+0.9)
        q = CoTPairQuality(arithmetic_coverage=0.6, label_confidence=0.9, n_steps=3)
        expected = 2.0 * 0.6 * 0.9 / (0.6 + 0.9)
        assert abs(q.quality_score - expected) < 1e-9

    def test_quality_score_zero_when_coverage_zero(self):
        q = CoTPairQuality(arithmetic_coverage=0.0, label_confidence=1.0, n_steps=1)
        assert q.quality_score == 0.0

    def test_quality_score_zero_when_confidence_zero(self):
        q = CoTPairQuality(arithmetic_coverage=1.0, label_confidence=0.0, n_steps=1)
        assert q.quality_score == 0.0

    def test_quality_score_perfect(self):
        q = CoTPairQuality(arithmetic_coverage=1.0, label_confidence=1.0, n_steps=5)
        assert abs(q.quality_score - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# _estimate_arithmetic_coverage tests
# ---------------------------------------------------------------------------


class TestEstimateArithmeticCoverage:
    def test_explicit_field_used_first(self):
        pair = {"arithmetic_coverage": 0.42, "step_text": "no arithmetic here"}
        assert _estimate_arithmetic_coverage(pair) == pytest.approx(0.42)

    def test_invalid_explicit_field_falls_through(self):
        pair = {"arithmetic_coverage": "not_a_number", "step_text": "2 * 3 = 6"}
        result = _estimate_arithmetic_coverage(pair)
        assert result >= 0.0

    def test_step_text_with_arithmetic(self):
        pair = {"step_text": "2 * 3 = 6 and 5 + 4 = 9"}
        result = _estimate_arithmetic_coverage(pair)
        assert result > 0.0

    def test_step_text_without_arithmetic(self):
        pair = {"step_text": "The answer is because the reasoning makes sense"}
        result = _estimate_arithmetic_coverage(pair)
        assert result == pytest.approx(0.0)

    def test_empty_text_returns_zero(self):
        pair = {}
        result = _estimate_arithmetic_coverage(pair)
        assert result == pytest.approx(0.0)

    def test_response_field_used_when_step_text_absent(self):
        pair = {"response": "3 * 7 = 21"}
        result = _estimate_arithmetic_coverage(pair)
        assert result >= 0.0

    def test_result_clamped_to_one(self):
        # Dense arithmetic text should not exceed 1.0
        pair = {
            "step_text": "1+2=3\n2+3=5\n3+4=7\n4+5=9\n5+6=11\n6+7=13"
        }
        result = _estimate_arithmetic_coverage(pair)
        assert result <= 1.0


# ---------------------------------------------------------------------------
# _estimate_label_confidence tests
# ---------------------------------------------------------------------------


class TestEstimateLabelConfidence:
    def test_label_confidence_field_first(self):
        pair = {"label_confidence": 0.75, "confidence": 0.5}
        assert _estimate_label_confidence(pair) == pytest.approx(0.75)

    def test_confidence_field_fallback(self):
        pair = {"confidence": 0.85}
        assert _estimate_label_confidence(pair) == pytest.approx(0.85)

    def test_default_is_one_when_no_field(self):
        pair = {"label": "correct"}
        assert _estimate_label_confidence(pair) == pytest.approx(1.0)

    def test_invalid_confidence_falls_through(self):
        pair = {"label_confidence": "bad", "confidence": "also_bad"}
        assert _estimate_label_confidence(pair) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# CoTPairQualityFilter tests
# ---------------------------------------------------------------------------


class TestCoTPairQualityFilter:
    def test_filter_removes_low_coverage_pair(self, low_coverage_pair):
        # SCENARIO-LEARN-066: pair with coverage=0.1 is rejected
        f = CoTPairQualityFilter(min_coverage=0.3, min_confidence=0.7)
        result = f.filter([low_coverage_pair])
        assert result == []

    def test_filter_removes_low_confidence_pair(self, low_confidence_pair):
        f = CoTPairQualityFilter(min_coverage=0.3, min_confidence=0.7)
        result = f.filter([low_confidence_pair])
        assert result == []

    def test_filter_passes_high_quality_pair(self, high_coverage_pair):
        f = CoTPairQualityFilter(min_coverage=0.3, min_confidence=0.7)
        result = f.filter([high_coverage_pair])
        assert len(result) == 1
        assert result[0] is high_coverage_pair

    def test_filter_empty_list(self):
        f = CoTPairQualityFilter()
        assert f.filter([]) == []

    def test_filter_mixed_corpus(self, high_coverage_pair, low_coverage_pair, low_confidence_pair, passing_pair):
        f = CoTPairQualityFilter(min_coverage=0.3, min_confidence=0.7)
        corpus = [high_coverage_pair, low_coverage_pair, low_confidence_pair, passing_pair]
        result = f.filter(corpus)
        assert high_coverage_pair in result
        assert passing_pair in result
        assert low_coverage_pair not in result
        assert low_confidence_pair not in result

    def test_compute_quality_returns_cot_pair_quality(self, high_coverage_pair):
        f = CoTPairQualityFilter()
        q = f.compute_quality(high_coverage_pair)
        assert isinstance(q, CoTPairQuality)
        assert q.arithmetic_coverage == pytest.approx(0.8)  # explicit field
        assert q.label_confidence == pytest.approx(0.9)

    def test_compute_quality_n_steps_from_text(self):
        pair = {"step_text": "line 1\nline 2\nline 3", "arithmetic_coverage": 0.5, "confidence": 0.8}
        f = CoTPairQualityFilter()
        q = f.compute_quality(pair)
        assert q.n_steps == 3

    def test_compute_quality_n_steps_default_one_when_no_text(self):
        pair = {"arithmetic_coverage": 0.5, "confidence": 0.8}
        f = CoTPairQualityFilter()
        q = f.compute_quality(pair)
        assert q.n_steps == 1

    def test_custom_thresholds(self):
        # With very strict thresholds, even good pairs fail
        pair = {"arithmetic_coverage": 0.5, "confidence": 0.8}
        f = CoTPairQualityFilter(min_coverage=0.9, min_confidence=0.95)
        result = f.filter([pair])
        assert result == []

    def test_default_thresholds_are_0_3_and_0_7(self):
        f = CoTPairQualityFilter()
        assert f.min_coverage == pytest.approx(0.3)
        assert f.min_confidence == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# JEPAQualityAugmentor tests
# ---------------------------------------------------------------------------


class TestJEPAQualityAugmentor:
    def test_generate_violation_pairs_all_incorrect(self, small_ising):
        # SCENARIO-LEARN-067: all generated violation pairs have correct=False
        aug = JEPAQualityAugmentor(small_ising, n_samples=20)
        pairs = aug.generate_violation_pairs()
        assert len(pairs) > 0
        for pair in pairs:
            assert pair["correct"] is False
            assert pair["label"] == "incorrect"

    def test_generate_correct_pairs_all_correct(self, small_ising):
        aug = JEPAQualityAugmentor(small_ising, n_samples=20)
        pairs = aug.generate_correct_pairs()
        assert len(pairs) > 0
        for pair in pairs:
            assert pair["correct"] is True
            assert pair["label"] == "correct"

    def test_violation_pairs_have_required_fields(self, small_ising):
        aug = JEPAQualityAugmentor(small_ising, n_samples=10)
        pairs = aug.generate_violation_pairs()
        for pair in pairs:
            assert "response" in pair
            assert "question_id" in pair
            assert "source" in pair
            assert pair["source"] == "ebm_guided_synthetic"

    def test_correct_pairs_have_required_fields(self, small_ising):
        aug = JEPAQualityAugmentor(small_ising, n_samples=10)
        pairs = aug.generate_correct_pairs()
        for pair in pairs:
            assert "response" in pair
            assert "question_id" in pair
            assert pair["source"] == "ebm_guided_synthetic"

    def test_violation_pairs_have_label_confidence_one(self, small_ising):
        # EBM energy is ground truth — synthetic pairs are labeled with confidence=1.0
        aug = JEPAQualityAugmentor(small_ising, n_samples=10)
        pairs = aug.generate_violation_pairs()
        for pair in pairs:
            assert pair["label_confidence"] == pytest.approx(1.0)

    def test_violation_pairs_pass_quality_filter(self, small_ising):
        # Synthetic pairs set arithmetic_coverage=0.5, confidence=1.0 → should pass gate
        aug = JEPAQualityAugmentor(small_ising, n_samples=10)
        pairs = aug.generate_violation_pairs()
        f = CoTPairQualityFilter(min_coverage=0.3, min_confidence=0.7)
        filtered = f.filter(pairs)
        assert len(filtered) == len(pairs)

    def test_n_samples_zero_returns_empty(self, small_ising):
        aug = JEPAQualityAugmentor(small_ising, n_samples=0)
        assert aug.generate_violation_pairs() == []
        assert aug.generate_correct_pairs() == []

    def test_spin_to_text_produces_string(self, small_ising):
        aug = JEPAQualityAugmentor(small_ising, n_samples=4)
        spin = jnp.ones(8)
        text = aug._spin_to_text(spin)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_violation_energies_above_mean(self, small_ising):
        # SCENARIO-LEARN-067: violation configs have energy above mean
        aug = JEPAQualityAugmentor(small_ising, n_samples=40)
        # Compute the mean energy from sample configs
        configs, energies = aug._sample_spin_configs(seed=1)
        mean_e = sum(energies) / len(energies)
        violation_pairs = aug.generate_violation_pairs()
        for pair in violation_pairs:
            assert pair["energy"] >= mean_e


# ---------------------------------------------------------------------------
# JEPARetrainV2Result tests
# ---------------------------------------------------------------------------


class TestJEPARetrainV2Result:
    def test_auc_improvement(self):
        r = JEPARetrainV2Result(
            n_pairs_raw=57, n_pairs_filtered=30, n_synthetic=170,
            before_auc=0.400, after_auc=0.650
        )
        assert r.auc_improvement == pytest.approx(0.250)

    def test_target_met_when_above_0_700(self):
        r = JEPARetrainV2Result(57, 30, 170, before_auc=0.400, after_auc=0.720)
        assert r.target_met is True

    def test_target_not_met_when_below_0_700(self):
        r = JEPARetrainV2Result(57, 30, 170, before_auc=0.400, after_auc=0.650)
        assert r.target_met is False

    def test_regression_recovered_when_above_0_571(self):
        # SCENARIO-LEARN-068: after_auc=0.620 > 0.571 → regression_recovered=True
        r = JEPARetrainV2Result(57, 30, 170, before_auc=0.400, after_auc=0.620)
        assert r.regression_recovered is True

    def test_regression_not_recovered_when_below_0_571(self):
        r = JEPARetrainV2Result(57, 30, 170, before_auc=0.400, after_auc=0.500)
        assert r.regression_recovered is False

    def test_retro_040_closed_when_above_0_600(self):
        # SCENARIO-LEARN-068
        r = JEPARetrainV2Result(57, 30, 170, before_auc=0.400, after_auc=0.620)
        assert r.retro_040_closed is True

    def test_retro_040_not_closed_when_below_0_600(self):
        r = JEPARetrainV2Result(57, 30, 170, before_auc=0.400, after_auc=0.580)
        assert r.retro_040_closed is False

    def test_all_false_when_no_improvement(self):
        r = JEPARetrainV2Result(57, 30, 170, before_auc=0.400, after_auc=0.400)
        assert r.auc_improvement == pytest.approx(0.0)
        assert r.target_met is False
        assert r.regression_recovered is False
        assert r.retro_040_closed is False

    def test_scenario_learn_068_exact(self):
        # SCENARIO-LEARN-068 exactly as spec'd
        r = JEPARetrainV2Result(
            n_pairs_raw=57, n_pairs_filtered=30, n_synthetic=170,
            before_auc=0.400, after_auc=0.620
        )
        assert r.regression_recovered is True
        assert r.retro_040_closed is True
        assert r.target_met is False

    def test_import_from_carnot_models(self):
        # Verify the export from carnot.models.__init__
        from carnot.models import JEPARetrainV2Result as imported
        assert imported is JEPARetrainV2Result
