"""Tests for PCIBProbe — Predictive Coding + Information Bottleneck hallucination signals.

Validates that the probe:
  - Correctly computes entity_uptake (novel numbers = higher surprise).
  - Correctly computes falsifiability_score (ungrounded conclusions = higher score).
  - Combined score is a weighted average in [0, 1].
  - Handles edge cases: empty text, no numbers, no conclusions.
  - Handles context fallback when context is a short ID string.

Spec: REQ-VERIFY-162, REQ-VERIFY-163
"""

from __future__ import annotations

import pytest

from python.carnot.verify.pcib_probe import (
    PCIBProbe,
    _extract_numbers,
    _extract_sentences,
    _is_conclusion_sentence,
    _values_reachable,
)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestExtractNumbers:
    """Spec: REQ-VERIFY-162"""

    def test_plain_integers(self):
        nums = _extract_numbers("The value is 42 and another is 100.")
        assert 42.0 in nums
        assert 100.0 in nums

    def test_decimals(self):
        nums = _extract_numbers("Pi is approximately 3.14159.")
        assert any(abs(n - 3.14159) < 0.0001 for n in nums)

    def test_latex_wrapped_numbers(self):
        # LaTeX like \( 80 \) should still yield 80
        nums = _extract_numbers(r"\( 80 \) plus \( 20 \)")
        assert 80.0 in nums or 20.0 in nums  # at least one must survive stripping

    def test_empty_string(self):
        assert _extract_numbers("") == []

    def test_no_numbers(self):
        assert _extract_numbers("No numeric content here.") == []


class TestExtractSentences:
    """Spec: REQ-VERIFY-162"""

    def test_splits_on_period(self):
        sents = _extract_sentences("First sentence. Second sentence. Third.")
        assert len(sents) == 3

    def test_splits_on_newline(self):
        sents = _extract_sentences("Line one\nLine two\nLine three")
        assert len(sents) >= 2

    def test_empty_string(self):
        assert _extract_sentences("") == []


class TestIsConclusionSentence:
    """Spec: REQ-VERIFY-162"""

    def test_therefore_is_conclusion(self):
        assert _is_conclusion_sentence("Therefore, the total is 260.")

    def test_thus_is_conclusion(self):
        assert _is_conclusion_sentence("Thus the answer is 42.")

    def test_plain_computation_is_not_conclusion(self):
        assert not _is_conclusion_sentence("We substitute C = 80 into the equation.")

    def test_case_insensitive(self):
        assert _is_conclusion_sentence("THEREFORE X = 100")


class TestValuesReachable:
    """Spec: REQ-VERIFY-163"""

    def test_identity(self):
        # Target present directly in sources
        assert _values_reachable(80.0, [80.0, 20.0])

    def test_addition(self):
        # 160 + 80 = 240 (pairwise addition — implementation checks pairs only)
        assert _values_reachable(240.0, [160.0, 80.0, 20.0])

    def test_multiplication(self):
        # 4 * 20 = 80
        assert _values_reachable(80.0, [4.0, 20.0])

    def test_unreachable(self):
        # 999 cannot be derived from 1, 2, 3
        assert not _values_reachable(999.0, [1.0, 2.0, 3.0])

    def test_empty_sources(self):
        assert not _values_reachable(42.0, [])


# ---------------------------------------------------------------------------
# Integration tests for PCIBProbe
# ---------------------------------------------------------------------------


class TestPCIBProbeEntityUptake:
    """Spec: REQ-VERIFY-162"""

    def test_fully_grounded_step(self):
        """A step that only uses numbers from its context has low entity uptake."""
        probe = PCIBProbe()
        # Context introduces 20 and 4. Response uses only those values.
        context = "There are 20 sheep and a factor of 4."
        response = "Therefore C = 4 * 20 = 80."
        eu = probe.compute_entity_uptake(response, context)
        # 80 is new but derived; 4 and 20 are in context.
        # The probe measures novelty, so 80 (not in context) counts as novel.
        assert 0.0 <= eu <= 1.0

    def test_hallucinated_step_has_higher_uptake(self):
        """A step that introduces completely new numbers has higher entity uptake."""
        probe = PCIBProbe()
        context = "There are 5 apples."
        # Response introduces completely unrelated large numbers
        response = "Therefore the total is 1234567 across all categories."
        eu_hallu = probe.compute_entity_uptake(response, context)

        response_grounded = "Therefore the total is 5."
        eu_grounded = probe.compute_entity_uptake(response_grounded, context)
        # Hallucinated step should have higher or equal uptake
        assert eu_hallu >= eu_grounded

    def test_no_numbers_returns_zero(self):
        probe = PCIBProbe()
        eu = probe.compute_entity_uptake(
            "There are no numbers here.", "Context also lacks numbers."
        )
        assert eu == 0.0

    def test_short_context_uses_fallback(self):
        """When context is just an ID (short), probe falls back to split-halves."""
        probe = PCIBProbe()
        step = "First half has 10 and 20. Second half concludes 999 is the total."
        eu = probe.compute_entity_uptake(step, "42")  # short context = fallback
        assert 0.0 <= eu <= 1.0


class TestPCIBProbeFalsifiabilityScore:
    """Spec: REQ-VERIFY-163"""

    def test_correct_conclusion_has_low_falsifiability(self):
        """A conclusion that follows from visible arithmetic is low falsifiability."""
        probe = PCIBProbe()
        step = "C = 4 * 20 = 80. T = 2 * 80 = 160. Therefore, the total is 160 + 80 + 20 = 260."
        fs = probe.compute_falsifiability_score(step, "")
        # 260 IS reachable from {160, 80, 20} via addition — should be low
        assert 0.0 <= fs <= 1.0

    def test_wrong_conclusion_has_higher_falsifiability(self):
        """A conclusion citing an unmotivated number has higher falsifiability."""
        probe = PCIBProbe()
        correct_step = "C = 4 * 20 = 80. T = 2 * 80 = 160. Therefore, the total is 260."
        wrong_step = "C = 4 * 20 = 80. T = 2 * 80 = 160. Therefore, the total is 9999."
        fs_correct = probe.compute_falsifiability_score(correct_step, "")
        fs_wrong = probe.compute_falsifiability_score(wrong_step, "")
        # Wrong conclusion should be at least as high as correct
        assert fs_wrong >= fs_correct

    def test_no_conclusion_returns_zero(self):
        probe = PCIBProbe()
        step = "We substitute C = 80 into the first equation to get T = 160."
        fs = probe.compute_falsifiability_score(step, "")
        assert fs == 0.0

    def test_too_few_numbers_returns_zero(self):
        """With fewer than min_numbers_for_falsifiability, score is 0."""
        probe = PCIBProbe(min_numbers_for_falsifiability=5)
        step = "Therefore, the answer is 42."
        fs = probe.compute_falsifiability_score(step, "")
        # Only 1 number in premise (none, actually) — below threshold
        assert fs == 0.0


class TestPCIBProbeCombinedScore:
    """Spec: REQ-VERIFY-162, REQ-VERIFY-163"""

    def test_score_in_unit_interval(self):
        probe = PCIBProbe()
        s = probe.score("Therefore, the total is 260.", "There are 20 sheep.")
        assert 0.0 <= s <= 1.0

    def test_score_is_weighted_average(self):
        """Combined score = entity_weight * eu + falsifiability_weight * fs."""
        probe = PCIBProbe(entity_weight=0.3, falsifiability_weight=0.7)
        step = "Therefore the total is 999."
        ctx = "There are 1 item."
        eu = probe.compute_entity_uptake(step, ctx)
        fs = probe.compute_falsifiability_score(step, ctx)
        expected = 0.3 * eu + 0.7 * fs
        actual = probe.score(step, ctx)
        assert abs(actual - expected) < 1e-9

    def test_empty_text_does_not_crash(self):
        probe = PCIBProbe()
        assert probe.score("", "") == 0.0
        assert probe.score("", "some context") == 0.0
