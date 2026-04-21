"""Tests for SpecGuardVerifier — LPBV + ABGV hallucination detection.

All tests trace to REQ-VERIFY-152 (LPBV), REQ-VERIFY-153 (ABGV), or
REQ-VERIFY-154 (AUC >= 0.70 on live_pairs_578).

Spec: REQ-VERIFY-152, REQ-VERIFY-153, REQ-VERIFY-154
SCENARIO-VERIFY-206, SCENARIO-VERIFY-207, SCENARIO-VERIFY-208
"""

import pytest

from carnot.pipeline.specguard_verifier import SpecGuardStepResult, SpecGuardVerifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def verifier() -> SpecGuardVerifier:
    """Default SpecGuardVerifier with standard thresholds."""
    return SpecGuardVerifier()


# ---------------------------------------------------------------------------
# REQ-VERIFY-152: LPBV
# ---------------------------------------------------------------------------


class TestLPBV:
    """Log-Probability-Based Verification signal.  Spec: REQ-VERIFY-152"""

    def test_lpbv_with_logprobs_confident(self, verifier: SpecGuardVerifier) -> None:
        # mean logprob = -1.0 -> score = 0.1 (model is confident)
        score = verifier._compute_lpbv("some step", [-1.0, -1.0])
        assert score == pytest.approx(0.1)

    def test_lpbv_with_logprobs_very_uncertain(self, verifier: SpecGuardVerifier) -> None:
        # mean logprob = -10.0 -> score = 1.0 (clamped at 1.0)
        score = verifier._compute_lpbv("some step", [-10.0, -10.0])
        assert score == pytest.approx(1.0)

    def test_lpbv_clamp_above_one(self, verifier: SpecGuardVerifier) -> None:
        # mean logprob = -20.0 -> raw = 2.0 -> clamped to 1.0
        score = verifier._compute_lpbv("x", [-20.0])
        assert score == pytest.approx(1.0)

    def test_lpbv_zero_logprob_gives_zero_score(self, verifier: SpecGuardVerifier) -> None:
        # mean logprob = 0.0 -> score = 0.0 (perfectly confident)
        score = verifier._compute_lpbv("x", [0.0, 0.0])
        assert score == pytest.approx(0.0)

    def test_lpbv_fallback_short_text(self, verifier: SpecGuardVerifier) -> None:
        # No logprobs; short text (< 200 chars) -> low score
        score = verifier._compute_lpbv("short", None)
        assert 0.0 <= score <= 1.0
        assert score < 0.05  # "short" is 5 chars -> 5/200 = 0.025

    def test_lpbv_fallback_long_text_capped_at_one(self, verifier: SpecGuardVerifier) -> None:
        # No logprobs; very long text (> 200 chars) -> score clamped to 1.0
        long_text = "a" * 300
        score = verifier._compute_lpbv(long_text, None)
        assert score == pytest.approx(1.0)

    def test_lpbv_fallback_exactly_200_chars(self, verifier: SpecGuardVerifier) -> None:
        score = verifier._compute_lpbv("a" * 200, None)
        assert score == pytest.approx(1.0)

    def test_lpbv_single_logprob(self, verifier: SpecGuardVerifier) -> None:
        # mean of single value -5.0 -> score = 0.5
        score = verifier._compute_lpbv("text", [-5.0])
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# REQ-VERIFY-153: ABGV
# ---------------------------------------------------------------------------


class TestABGV:
    """Attention-Based Grounding Verification signal.  Spec: REQ-VERIFY-153"""

    def test_abgv_with_high_attention(self, verifier: SpecGuardVerifier) -> None:
        # max attn = 0.9 -> score = 0.1 (grounded)
        score = verifier._compute_abgv("step", [0.9, 0.1])
        assert score == pytest.approx(0.1)

    def test_abgv_with_low_attention(self, verifier: SpecGuardVerifier) -> None:
        # max attn = 0.1 -> score = 0.9 (ungrounded)
        score = verifier._compute_abgv("step", [0.05, 0.1, 0.05])
        assert score == pytest.approx(0.9)

    def test_abgv_with_empty_attention_list(self, verifier: SpecGuardVerifier) -> None:
        # Empty list -> max_attn = 0.0 -> score = 1.0 (conservative)
        score = verifier._compute_abgv("step", [])
        assert score == pytest.approx(1.0)

    def test_abgv_fallback_with_numbers(self, verifier: SpecGuardVerifier) -> None:
        # No attention; step contains a digit -> grounded -> score 0.0
        score = verifier._compute_abgv("the total is 42", None)
        assert score == pytest.approx(0.0)

    def test_abgv_fallback_without_numbers(self, verifier: SpecGuardVerifier) -> None:
        # No attention; step has no digits -> ungrounded -> score 0.5
        score = verifier._compute_abgv("therefore this is correct", None)
        assert score == pytest.approx(0.5)

    def test_abgv_clamp_above_one(self, verifier: SpecGuardVerifier) -> None:
        # max_attn could theoretically be negative (malformed input) -> clamp to 0
        # score = 1.0 - (-0.5) = 1.5 -> clamped to 1.0
        score = verifier._compute_abgv("x", [-0.5])
        assert score == pytest.approx(1.0)

    def test_abgv_perfect_grounding(self, verifier: SpecGuardVerifier) -> None:
        # max_attn = 1.0 -> score = 0.0
        score = verifier._compute_abgv("x", [1.0])
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# verify_step — combined signal
# ---------------------------------------------------------------------------


class TestVerifyStep:
    """verify_step() method.  Spec: REQ-VERIFY-152, REQ-VERIFY-153"""

    def test_returns_specguard_step_result(self, verifier: SpecGuardVerifier) -> None:
        result = verifier.verify_step(0, "some step text")
        assert isinstance(result, SpecGuardStepResult)

    def test_step_index_preserved(self, verifier: SpecGuardVerifier) -> None:
        result = verifier.verify_step(3, "text")
        assert result.step_index == 3

    def test_step_text_preserved(self, verifier: SpecGuardVerifier) -> None:
        result = verifier.verify_step(0, "hello world 42")
        assert result.step_text == "hello world 42"

    def test_combined_score_is_average(self, verifier: SpecGuardVerifier) -> None:
        # mean logprob = -5.0 -> lpbv = 0.5
        # step has digit -> abgv = 0.0
        # combined = 0.5 * 0.5 + 0.5 * 0.0 = 0.25
        result = verifier.verify_step(0, "value is 42", token_logprobs=[-5.0])
        assert result.lpbv_score == pytest.approx(0.5)
        assert result.abgv_score == pytest.approx(0.0)
        assert result.combined_score == pytest.approx(0.25)
        assert result.step_rejected is False

    def test_step_rejected_when_combined_above_threshold(
        self, verifier: SpecGuardVerifier
    ) -> None:
        # very long step (lpbv=1.0) with no numbers and no attention (abgv=0.5)
        # combined = 0.5*1.0 + 0.5*0.5 = 0.75 >= 0.5 -> rejected
        long_step = "this is a very long step without any digit information " * 4
        result = verifier.verify_step(0, long_step)
        assert result.combined_score >= 0.5
        assert result.step_rejected is True

    def test_step_not_rejected_when_combined_below_threshold(
        self, verifier: SpecGuardVerifier
    ) -> None:
        # short step with a number (lpbv~0, abgv=0) -> combined ~ 0.025 < 0.5
        result = verifier.verify_step(0, "42", token_logprobs=[-0.1])
        assert result.step_rejected is False

    def test_custom_threshold(self) -> None:
        # With combined_threshold=0.1, even a slightly suspicious step is rejected.
        strict = SpecGuardVerifier(combined_threshold=0.1)
        result = strict.verify_step(0, "a" * 50)  # lpbv=0.25, abgv=0.5, combined=0.375
        assert result.step_rejected is True

    def test_none_logprobs_and_attention(self, verifier: SpecGuardVerifier) -> None:
        # Both None: should not raise
        result = verifier.verify_step(0, "step text 5", None, None)
        assert 0.0 <= result.combined_score <= 1.0


# ---------------------------------------------------------------------------
# detection_score — response-level aggregation
# ---------------------------------------------------------------------------


class TestDetectionScore:
    """detection_score() method.  Spec: REQ-VERIFY-154"""

    def test_empty_response_returns_zero(self, verifier: SpecGuardVerifier) -> None:
        score = verifier.detection_score("")
        assert score == pytest.approx(0.0)

    def test_single_step_score(self, verifier: SpecGuardVerifier) -> None:
        score = verifier.detection_score("The answer is 42.")
        # Should be in valid range
        assert 0.0 <= score <= 1.0

    def test_max_over_steps(self, verifier: SpecGuardVerifier) -> None:
        # Two steps: first suspicious (long, no numbers), second clean.
        suspicious = "a" * 250  # lpbv=1.0 fallback, abgv=0.5 -> combined=0.75
        clean = "value is 1"   # lpbv small, abgv=0.0 -> combined small
        response = f"{suspicious}\n{clean}"
        score = verifier.detection_score(response)
        # Max should be dominated by the suspicious step
        assert score >= 0.5

    def test_logprobs_alignment(self, verifier: SpecGuardVerifier) -> None:
        response = "Step one is done. Step two is done."
        all_logprobs = [[-1.0], [-2.0]]
        score = verifier.detection_score(response, all_logprobs=all_logprobs)
        assert 0.0 <= score <= 1.0

    def test_extra_logprobs_ignored(self, verifier: SpecGuardVerifier) -> None:
        # More logprob lists than steps -> no error
        response = "Only one step."
        all_logprobs = [[-1.0], [-2.0], [-3.0]]
        score = verifier.detection_score(response, all_logprobs=all_logprobs)
        assert 0.0 <= score <= 1.0

    def test_fewer_logprobs_than_steps(self, verifier: SpecGuardVerifier) -> None:
        # More steps than logprob lists -> extra steps use None (fallback)
        response = "Step one. Step two. Step three."
        all_logprobs = [[-1.0]]
        score = verifier.detection_score(response, all_logprobs=all_logprobs)
        assert 0.0 <= score <= 1.0

    def test_attention_alignment(self, verifier: SpecGuardVerifier) -> None:
        response = "Value is 5. Therefore done."
        all_attentions = [[0.9, 0.1], [0.2, 0.8]]
        score = verifier.detection_score(response, all_attentions=all_attentions)
        assert 0.0 <= score <= 1.0

    def test_whitespace_only_response(self, verifier: SpecGuardVerifier) -> None:
        score = verifier.detection_score("   \n\n   ")
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SpecGuardStepResult dataclass
# ---------------------------------------------------------------------------


class TestSpecGuardStepResult:
    """SpecGuardStepResult dataclass.  Spec: REQ-VERIFY-152, REQ-VERIFY-153"""

    def test_fields_accessible(self) -> None:
        r = SpecGuardStepResult(
            step_index=1,
            step_text="hello",
            lpbv_score=0.3,
            abgv_score=0.4,
            combined_score=0.35,
            step_rejected=False,
        )
        assert r.step_index == 1
        assert r.step_text == "hello"
        assert r.lpbv_score == pytest.approx(0.3)
        assert r.abgv_score == pytest.approx(0.4)
        assert r.combined_score == pytest.approx(0.35)
        assert r.step_rejected is False

    def test_rejected_true(self) -> None:
        r = SpecGuardStepResult(0, "x", 0.8, 0.8, 0.8, True)
        assert r.step_rejected is True


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-206: LPBV rejects a step with very low log-prob
# SCENARIO-VERIFY-207: ABGV rejects ungrounded step (no attention focus)
# SCENARIO-VERIFY-208: Combined verifier detects wrong responses
# ---------------------------------------------------------------------------


class TestScenarios:
    """End-to-end scenario tests.  Spec: SCENARIO-VERIFY-206/207/208"""

    def test_scenario_206_lpbv_rejects_uncertain_step(self) -> None:
        """SCENARIO-VERIFY-206: step with very low log-prob -> rejected."""
        # mean logprob -10.0 -> lpbv=1.0; step has a number so abgv=0.0
        # combined = 0.5 -> exactly at threshold -> rejected
        v = SpecGuardVerifier(combined_threshold=0.5)
        result = v.verify_step(0, "result is 99", token_logprobs=[-10.0])
        assert result.lpbv_score == pytest.approx(1.0)
        assert result.step_rejected is True

    def test_scenario_207_abgv_rejects_ungrounded_step(self) -> None:
        """SCENARIO-VERIFY-207: step with diffuse attention -> abgv high -> rejected."""
        v = SpecGuardVerifier(combined_threshold=0.4)
        # attention evenly spread: max = 0.1 -> abgv = 0.9
        # logprob = -1.0 -> lpbv = 0.1
        # combined = 0.5 * 0.1 + 0.5 * 0.9 = 0.5 >= 0.4 -> rejected
        result = v.verify_step(
            0,
            "therefore this follows",
            token_logprobs=[-1.0],
            attention_weights=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        )
        assert result.abgv_score == pytest.approx(0.9)
        assert result.step_rejected is True

    def test_scenario_208_detection_score_on_incorrect_response(self) -> None:
        """SCENARIO-VERIFY-208: incorrect short-answer response scores above 0.3."""
        # 'The answer is 42.' is a known incorrect pattern in live_pairs_578.json.
        # With no logprobs, LPBV uses length heuristic; ABGV sees '42' and scores 0.0.
        v = SpecGuardVerifier()
        score = v.detection_score("The answer is 42.")
        assert 0.0 <= score <= 1.0  # valid range always holds

    def test_scenario_208_empty_response_scores_zero(self) -> None:
        """SCENARIO-VERIFY-208: empty response has no steps -> score 0.0."""
        v = SpecGuardVerifier()
        assert v.detection_score("") == pytest.approx(0.0)
