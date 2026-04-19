"""Tests for NUPProbe — Neural Uncertainty Principle Tier 0c pre-filter.

Spec: REQ-VERIFY-096, REQ-VERIFY-097,
      SCENARIO-VERIFY-129, SCENARIO-VERIFY-130
"""

from __future__ import annotations

import math

import pytest

from carnot.pipeline.nup_probe import (
    ContinuationEntropy,
    NUPProbe,
    NUPProbeResult,
    score_with_latency,
)


# ---------------------------------------------------------------------------
# ContinuationEntropy
# ---------------------------------------------------------------------------


class TestContinuationEntropy:
    def test_uniform_logprobs_returns_max_entropy(self):
        """Uniform distribution over N tokens has entropy = ln(N)."""
        # SCENARIO-VERIFY-129: uniform logprobs → maximum entropy
        n = 8
        logprobs = [0.0] * n  # log(1) = 0 for all, uniform after normalisation
        ce = ContinuationEntropy.from_logprobs(logprobs, threshold=1.5)
        expected = math.log(n)
        assert abs(ce.entropy - expected) < 1e-9

    def test_peaked_logprobs_returns_near_zero_entropy(self):
        """Distribution heavily peaked on one token → entropy near 0."""
        # SCENARIO-VERIFY-129: peaked logprobs → near-zero entropy
        # One very high logprob, rest extremely low
        logprobs = [0.0] + [-1000.0] * 9
        ce = ContinuationEntropy.from_logprobs(logprobs, threshold=1.5)
        assert ce.entropy < 0.01

    def test_is_high_entropy_true_when_above_threshold(self):
        n = 20  # ln(20) ≈ 3.0 > 1.5
        logprobs = [0.0] * n
        ce = ContinuationEntropy.from_logprobs(logprobs, threshold=1.5)
        assert ce.is_high_entropy is True

    def test_is_high_entropy_false_when_below_threshold(self):
        # Two tokens equally probable → ln(2) ≈ 0.693 < 1.5
        logprobs = [0.0, 0.0]
        ce = ContinuationEntropy.from_logprobs(logprobs, threshold=1.5)
        assert ce.is_high_entropy is False

    def test_empty_logprobs_returns_zero_entropy(self):
        ce = ContinuationEntropy.from_logprobs([], threshold=1.5)
        assert ce.entropy == 0.0
        assert ce.is_high_entropy is False

    def test_single_logprob_returns_zero_entropy(self):
        ce = ContinuationEntropy.from_logprobs([-0.5], threshold=1.5)
        assert ce.entropy == 0.0
        assert ce.is_high_entropy is False

    def test_entropy_is_non_negative(self):
        logprobs = [-1.0, -2.0, -3.0, -0.5]
        ce = ContinuationEntropy.from_logprobs(logprobs)
        assert ce.entropy >= 0.0

    def test_threshold_stored_on_instance(self):
        ce = ContinuationEntropy.from_logprobs([0.0, 0.0], threshold=2.5)
        assert ce.threshold == 2.5

    def test_logprobs_stored_on_instance(self):
        lp = [-0.1, -0.9]
        ce = ContinuationEntropy.from_logprobs(lp)
        assert ce.logprobs == lp

    def test_unnormalised_logprobs_still_computes_correct_entropy(self):
        """Logprobs that don't sum to 1 in prob-space are renormalised internally."""
        # log-probs offset by a constant — should produce same entropy as normalised
        logprobs_a = [0.0, 0.0]
        logprobs_b = [-5.0, -5.0]
        ce_a = ContinuationEntropy.from_logprobs(logprobs_a)
        ce_b = ContinuationEntropy.from_logprobs(logprobs_b)
        assert abs(ce_a.entropy - ce_b.entropy) < 1e-9


# ---------------------------------------------------------------------------
# NUPProbeResult
# ---------------------------------------------------------------------------


class TestNUPProbeResult:
    def test_is_viable_tier_0c_true_when_auc_above_threshold(self):
        result = NUPProbeResult(n_pairs=50, auc=0.75, threshold_used=1.5, probe_latency_ms=0.01)
        assert result.is_viable_tier_0c is True

    def test_is_viable_tier_0c_false_when_auc_below_threshold(self):
        result = NUPProbeResult(n_pairs=50, auc=0.65, threshold_used=1.5, probe_latency_ms=0.01)
        assert result.is_viable_tier_0c is False

    def test_is_viable_tier_0c_false_at_exact_threshold(self):
        # Strictly greater than 0.700 required
        result = NUPProbeResult(n_pairs=50, auc=0.700, threshold_used=1.5, probe_latency_ms=0.01)
        assert result.is_viable_tier_0c is False

    def test_fields_stored(self):
        result = NUPProbeResult(n_pairs=57, auc=0.82, threshold_used=1.8, probe_latency_ms=0.003)
        assert result.n_pairs == 57
        assert result.auc == 0.82
        assert result.threshold_used == 1.8
        assert result.probe_latency_ms == 0.003


# ---------------------------------------------------------------------------
# NUPProbe.score
# ---------------------------------------------------------------------------


class TestNUPProbeScore:
    def test_score_with_logprobs_returns_entropy(self):
        probe = NUPProbe(entropy_threshold=1.5)
        logprobs = [0.0] * 4  # ln(4) ≈ 1.386
        s = probe.score("some text", logprobs=logprobs)
        assert abs(s - math.log(4)) < 1e-9

    def test_score_without_logprobs_uses_char_entropy_fallback(self):
        probe = NUPProbe(entropy_threshold=1.5)
        # Character entropy of "aaaa" = 0 (all same char)
        s = probe.score("aaaa", logprobs=None)
        assert s == 0.0

    def test_score_empty_logprobs_falls_back_to_char_entropy(self):
        probe = NUPProbe(entropy_threshold=1.5)
        # Empty logprobs list should fall through to char entropy
        s_empty = probe.score("hello", logprobs=[])
        s_none = probe.score("hello", logprobs=None)
        assert s_empty == s_none

    def test_score_returns_float(self):
        probe = NUPProbe()
        assert isinstance(probe.score("test"), float)

    def test_score_is_non_negative(self):
        probe = NUPProbe()
        assert probe.score("2 + 2 = 4") >= 0.0
        assert probe.score("some long text with many characters") >= 0.0


# ---------------------------------------------------------------------------
# NUPProbe.predict_violation
# ---------------------------------------------------------------------------


class TestNUPProbePredictViolation:
    def test_high_entropy_predicts_violation(self):
        # SCENARIO-VERIFY-129: high entropy → violation predicted
        probe = NUPProbe(entropy_threshold=1.5)
        # 20 equal tokens → ln(20) ≈ 3.0 > 1.5
        logprobs = [0.0] * 20
        assert probe.predict_violation("text", logprobs=logprobs) is True

    def test_low_entropy_predicts_no_violation(self):
        probe = NUPProbe(entropy_threshold=1.5)
        # 2 equal tokens → ln(2) ≈ 0.693 < 1.5
        logprobs = [0.0, 0.0]
        assert probe.predict_violation("text", logprobs=logprobs) is False

    def test_predict_violation_without_logprobs(self):
        probe = NUPProbe(entropy_threshold=1.5)
        result = probe.predict_violation("aaaa")
        assert isinstance(result, bool)

    def test_exactly_at_threshold_is_not_violation(self):
        # Must be STRICTLY greater than threshold
        probe = NUPProbe(entropy_threshold=math.log(4))
        logprobs = [0.0] * 4  # entropy = exactly ln(4)
        assert probe.predict_violation("text", logprobs=logprobs) is False


# ---------------------------------------------------------------------------
# NUPProbe.evaluate_auc
# ---------------------------------------------------------------------------


class TestNUPProbeEvaluateAuc:
    def test_evaluate_auc_returns_float_in_range(self):
        # SCENARIO-VERIFY-130: AUC in [0, 1]
        probe = NUPProbe(entropy_threshold=1.5)
        pairs = [
            {"step_text": "2+2=4", "label": "correct"},
            {"step_text": "2+2=5", "label": "incorrect"},
            {"step_text": "abc def ghi", "label": "incorrect"},
            {"step_text": "xyz", "label": "correct"},
        ]
        auc = probe.evaluate_auc(pairs)
        assert isinstance(auc, float)
        assert 0.0 <= auc <= 1.0

    def test_evaluate_auc_fewer_than_2_pairs_returns_chance(self):
        probe = NUPProbe()
        assert probe.evaluate_auc([]) == 0.5
        assert probe.evaluate_auc([{"step_text": "x", "label": "correct"}]) == 0.5

    def test_evaluate_auc_all_same_label_returns_chance(self):
        probe = NUPProbe()
        pairs = [{"step_text": f"step{i}", "label": "correct"} for i in range(5)]
        auc = probe.evaluate_auc(pairs)
        assert auc == 0.5

    def test_evaluate_auc_accepts_cot_text_key(self):
        probe = NUPProbe()
        pairs = [
            {"cot_text": "2+2=4", "label": "correct"},
            {"cot_text": "hallucination", "label": "incorrect"},
        ]
        auc = probe.evaluate_auc(pairs)
        assert 0.0 <= auc <= 1.0

    def test_evaluate_auc_accepts_bool_labels(self):
        probe = NUPProbe()
        pairs = [
            {"step_text": "correct step", "label": True},
            {"step_text": "bad step", "label": False},
        ]
        auc = probe.evaluate_auc(pairs)
        assert 0.0 <= auc <= 1.0

    def test_evaluate_auc_with_logprobs(self):
        probe = NUPProbe(entropy_threshold=1.5)
        # High-entropy step labeled incorrect, low-entropy labeled correct
        pairs = [
            {"step_text": "x", "logprobs": [0.0] * 20, "label": "incorrect"},
            {"step_text": "x", "logprobs": [0.0, -100.0], "label": "correct"},
        ]
        auc = probe.evaluate_auc(pairs)
        # Should be > 0.5 since high entropy correlates with violation
        assert auc > 0.5

    def test_evaluate_auc_perfect_discrimination(self):
        """When high-entropy steps are always violations, AUC should be 1.0."""
        probe = NUPProbe(entropy_threshold=1.5)
        pairs = [
            # violations have high logprob entropy
            {"step_text": "v1", "logprobs": [0.0] * 50, "label": "incorrect"},
            {"step_text": "v2", "logprobs": [0.0] * 50, "label": "incorrect"},
            # correct steps have low logprob entropy
            {"step_text": "c1", "logprobs": [0.0, -100.0], "label": "correct"},
            {"step_text": "c2", "logprobs": [0.0, -100.0], "label": "correct"},
        ]
        auc = probe.evaluate_auc(pairs)
        assert auc >= 0.99


# ---------------------------------------------------------------------------
# NUPProbe._char_entropy (internal, tested via score)
# ---------------------------------------------------------------------------


class TestCharEntropy:
    def test_single_char_repeated_entropy_is_zero(self):
        probe = NUPProbe()
        assert probe._char_entropy("aaaa") == 0.0

    def test_empty_string_entropy_is_zero(self):
        probe = NUPProbe()
        assert probe._char_entropy("") == 0.0

    def test_single_char_entropy_is_zero(self):
        probe = NUPProbe()
        assert probe._char_entropy("x") == 0.0

    def test_two_distinct_chars_entropy_is_ln2(self):
        probe = NUPProbe()
        # "ab" → p(a)=0.5, p(b)=0.5 → H = ln(2)
        h = probe._char_entropy("ab")
        assert abs(h - math.log(2)) < 1e-9

    def test_more_distinct_chars_higher_entropy(self):
        probe = NUPProbe()
        h_low = probe._char_entropy("aabb")
        h_high = probe._char_entropy("abcd")
        assert h_high > h_low


# ---------------------------------------------------------------------------
# score_with_latency convenience function
# ---------------------------------------------------------------------------


class TestScoreWithLatency:
    def test_returns_score_and_latency(self):
        probe = NUPProbe()
        score, latency_ms = score_with_latency(probe, "test text")
        assert isinstance(score, float)
        assert isinstance(latency_ms, float)
        assert latency_ms >= 0.0

    def test_score_matches_probe_score(self):
        probe = NUPProbe()
        logprobs = [0.0] * 5
        score, _ = score_with_latency(probe, "text", logprobs=logprobs)
        assert abs(score - probe.score("text", logprobs=logprobs)) < 1e-12


# ---------------------------------------------------------------------------
# Integration: NUPProbe on live CoT data format
# ---------------------------------------------------------------------------


class TestNUPProbeIntegration:
    """Tests against the fover_labeled_steps_live.json data format."""

    def _make_pairs(self) -> list[dict]:
        """Build representative pairs matching the live data format."""
        return [
            {
                "question_id": "156",
                "step_text": "3. S = 20\nTotal = T + C + S = 160 + 80 + 20 = 260",
                "label": "incorrect",
                "confidence": 1.0,
            },
            {
                "question_id": "159",
                "step_text": "Overtime rate is $10 × 1.2 = $12 per hour.",
                "label": "correct",
                "confidence": 1.0,
            },
            {
                "question_id": "160",
                "step_text": "60 + 180 + 126 = 366",
                "label": "incorrect",
                "confidence": 1.0,
            },
            {
                "question_id": "165",
                "step_text": "Initial Investment: $5,000. Profit: 5000 × 2.5% = 125",
                "label": "correct",
                "confidence": 1.0,
            },
        ]

    def test_probe_scores_all_pairs(self):
        probe = NUPProbe(entropy_threshold=1.5)
        for pair in self._make_pairs():
            s = probe.score(pair["step_text"])
            assert s >= 0.0

    def test_evaluate_auc_on_live_format_returns_float(self):
        probe = NUPProbe(entropy_threshold=1.5)
        pairs = self._make_pairs()
        auc = probe.evaluate_auc(pairs)
        assert isinstance(auc, float)
        assert 0.0 <= auc <= 1.0
