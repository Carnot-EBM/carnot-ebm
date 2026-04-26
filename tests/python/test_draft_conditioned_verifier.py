"""Tests for DraftConditionedVerifier (Tier 2.8).

Covers:
- extract_structural_constraints: returns exactly 4 constraint dicts
- extract_structural_constraints: has_equals_sign True when "=" in draft
- extract_structural_constraints: has_equals_sign False when no "=" in draft
- extract_structural_constraints: has_numeric_answer True for trailing digit
- extract_structural_constraints: has_numeric_answer False for no trailing digit
- extract_structural_constraints: has_reasoning_steps True for >3 lines
- extract_structural_constraints: has_reasoning_steps False for <=3 lines
- extract_structural_constraints: final_number extracts last integer
- extract_structural_constraints: final_number is None for digit-free draft
- score_with_constraints: returns float when ising_sampler is None
- score_with_constraints: active constraints lower energy vs empty constraints
- score_with_constraints: delegates to ising_sampler when provided
- verify_with_draft: draft_used=True when runner returns non-empty string
- verify_with_draft: draft_used=False when runner raises exception
- verify_with_draft: draft_used=False when runner returns empty string
- verify_with_draft: n_constraints=0 when draft_used=False
- verify_with_draft: returns VerificationResult with correct field types
- condition_and_verify: returns plain dict with required keys
- condition_and_verify: energy field matches verify_with_draft energy

Spec: REQ-TIER28-001, SCENARIO-TIER28-001
Spec: REQ-VERIFY-001
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from python.carnot.pipeline.draft_conditioned_verifier import (
    DraftConditionedVerifier,
    VerificationResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _OkRunner:
    """Draft runner that always returns a fixed non-empty string."""

    def __init__(self, text: str = "Let x = 5 + 3.\nThen x = 8.\nThe answer is 8.") -> None:
        self._text = text

    def generate(self, question: str, max_tokens: int = 50) -> str:  # noqa: ARG002
        return self._text


class _EmptyRunner:
    """Draft runner that always returns empty string."""

    def generate(self, question: str, max_tokens: int = 50) -> str:  # noqa: ARG002
        return ""


class _FailingRunner:
    """Draft runner that always raises RuntimeError."""

    def generate(self, question: str, max_tokens: int = 50) -> str:  # noqa: ARG002
        raise RuntimeError("model not loaded")


# ---------------------------------------------------------------------------
# extract_structural_constraints
# REQ-TIER28-001-1
# ---------------------------------------------------------------------------


class TestExtractStructuralConstraints:
    """Tests for DraftConditionedVerifier.extract_structural_constraints()."""

    def setup_method(self) -> None:
        self.v = DraftConditionedVerifier(draft_runner=_OkRunner(), ising_sampler=None)

    def test_returns_four_constraints(self) -> None:
        """REQ-TIER28-001-1: always returns exactly 4 dicts."""
        result = self.v.extract_structural_constraints("x = 5")
        assert len(result) == 4

    def test_constraint_dicts_have_type_and_value(self) -> None:
        """REQ-TIER28-001-1: each dict has 'type' and 'value' keys."""
        result = self.v.extract_structural_constraints("hello")
        for c in result:
            assert "type" in c
            assert "value" in c

    def test_has_equals_sign_true(self) -> None:
        """has_equals_sign=True when '=' present in draft."""
        result = self.v.extract_structural_constraints("x = 42")
        cmap = {c["type"]: c["value"] for c in result}
        assert cmap["has_equals_sign"] is True

    def test_has_equals_sign_false(self) -> None:
        """has_equals_sign=False when '=' absent."""
        result = self.v.extract_structural_constraints("no equals here")
        cmap = {c["type"]: c["value"] for c in result}
        assert cmap["has_equals_sign"] is False

    def test_has_numeric_answer_true(self) -> None:
        """has_numeric_answer=True when last 100 chars contain a digit."""
        result = self.v.extract_structural_constraints("The answer is 42.")
        cmap = {c["type"]: c["value"] for c in result}
        assert cmap["has_numeric_answer"] is True

    def test_has_numeric_answer_false(self) -> None:
        """has_numeric_answer=False when no digit in tail."""
        result = self.v.extract_structural_constraints("no digits here at all")
        cmap = {c["type"]: c["value"] for c in result}
        assert cmap["has_numeric_answer"] is False

    def test_has_reasoning_steps_true(self) -> None:
        """has_reasoning_steps=True when draft has more than 3 lines."""
        draft = "Step 1\nStep 2\nStep 3\nStep 4\nThe answer."
        result = self.v.extract_structural_constraints(draft)
        cmap = {c["type"]: c["value"] for c in result}
        assert cmap["has_reasoning_steps"] is True

    def test_has_reasoning_steps_false(self) -> None:
        """has_reasoning_steps=False when draft has 3 or fewer lines."""
        draft = "Step 1\nStep 2\nAnswer."
        result = self.v.extract_structural_constraints(draft)
        cmap = {c["type"]: c["value"] for c in result}
        assert cmap["has_reasoning_steps"] is False

    def test_final_number_extracted(self) -> None:
        """final_number is the last integer in the draft."""
        result = self.v.extract_structural_constraints("a=1, b=2, answer is 99")
        cmap = {c["type"]: c["value"] for c in result}
        assert cmap["final_number"] == 99

    def test_final_number_none_for_digit_free(self) -> None:
        """final_number is None when draft contains no digits."""
        result = self.v.extract_structural_constraints("no numbers here at all")
        cmap = {c["type"]: c["value"] for c in result}
        assert cmap["final_number"] is None

    def test_empty_draft_returns_four_constraints(self) -> None:
        """Empty string: still returns 4 constraints, all inactive."""
        result = self.v.extract_structural_constraints("")
        assert len(result) == 4
        cmap = {c["type"]: c["value"] for c in result}
        assert cmap["has_equals_sign"] is False
        assert cmap["has_numeric_answer"] is False
        assert cmap["has_reasoning_steps"] is False
        assert cmap["final_number"] is None


# ---------------------------------------------------------------------------
# score_with_constraints
# REQ-TIER28-001-4
# ---------------------------------------------------------------------------


class TestScoreWithConstraints:
    """Tests for DraftConditionedVerifier.score_with_constraints() in synthetic mode."""

    def setup_method(self) -> None:
        self.v = DraftConditionedVerifier(draft_runner=_OkRunner(), ising_sampler=None)

    def test_returns_float(self) -> None:
        """REQ-TIER28-001-4: score_with_constraints returns a float."""
        e = self.v.score_with_constraints("some response", [])
        assert isinstance(e, float)

    def test_active_constraints_lower_energy(self) -> None:
        """Active structural constraints should lower energy vs empty list."""
        response = "Step 1\nStep 2\nStep 3\n2 + 3 = 5\nThe answer is 5."
        constraints_active = [
            {"type": "has_equals_sign", "value": True},
            {"type": "has_numeric_answer", "value": True},
            {"type": "has_reasoning_steps", "value": True},
            {"type": "final_number", "value": 5},
        ]
        e_with = self.v.score_with_constraints(response, constraints_active)
        e_without = self.v.score_with_constraints(response, [])
        assert e_with < e_without

    def test_delegates_to_ising_sampler(self) -> None:
        """REQ-TIER28-001-4: delegates to ising_sampler.score_with_constraints when set."""
        mock_ising = MagicMock()
        mock_ising.score_with_constraints.return_value = -3.14
        v = DraftConditionedVerifier(draft_runner=_OkRunner(), ising_sampler=mock_ising)
        result = v.score_with_constraints("resp", [{"type": "has_equals_sign", "value": True}])
        assert result == pytest.approx(-3.14)
        mock_ising.score_with_constraints.assert_called_once()

    def test_falls_back_on_ising_sampler_exception(self) -> None:
        """Falls back to synthetic scoring when ising_sampler raises."""
        mock_ising = MagicMock()
        mock_ising.score_with_constraints.side_effect = RuntimeError("GPU OOM")
        v = DraftConditionedVerifier(draft_runner=_OkRunner(), ising_sampler=mock_ising)
        e = v.score_with_constraints("some response", [])
        assert isinstance(e, float)


# ---------------------------------------------------------------------------
# verify_with_draft
# REQ-TIER28-001-2, REQ-TIER28-001-3
# ---------------------------------------------------------------------------


class TestVerifyWithDraft:
    """Tests for DraftConditionedVerifier.verify_with_draft()."""

    def test_returns_verification_result_type(self) -> None:
        """REQ-TIER28-001-2: returns a VerificationResult instance."""
        v = DraftConditionedVerifier(draft_runner=_OkRunner(), ising_sampler=None)
        result = v.verify_with_draft("What is 2+2?", "The answer is 4.")
        assert isinstance(result, VerificationResult)

    def test_draft_used_true_when_runner_succeeds(self) -> None:
        """REQ-TIER28-001-2: draft_used=True when runner returns non-empty string."""
        v = DraftConditionedVerifier(draft_runner=_OkRunner(), ising_sampler=None)
        result = v.verify_with_draft("What is 2+2?", "4")
        assert result.draft_used is True

    def test_draft_used_false_when_runner_raises(self) -> None:
        """REQ-TIER28-001-3: draft_used=False when runner raises exception."""
        v = DraftConditionedVerifier(draft_runner=_FailingRunner(), ising_sampler=None)
        result = v.verify_with_draft("question", "response")
        assert result.draft_used is False

    def test_draft_used_false_when_runner_returns_empty(self) -> None:
        """REQ-TIER28-001-3: draft_used=False when runner returns empty string."""
        v = DraftConditionedVerifier(draft_runner=_EmptyRunner(), ising_sampler=None)
        result = v.verify_with_draft("question", "response")
        assert result.draft_used is False

    def test_n_constraints_zero_when_draft_unused(self) -> None:
        """REQ-TIER28-001-3: n_constraints=0 when draft_used=False."""
        v = DraftConditionedVerifier(draft_runner=_FailingRunner(), ising_sampler=None)
        result = v.verify_with_draft("question", "response")
        assert result.n_constraints == 0

    def test_n_constraints_positive_when_draft_has_signal(self) -> None:
        """n_constraints > 0 when draft has equals, number, and multi-line."""
        draft_text = "x = 5 + 3.\nThen x = 8.\nThe answer is 8.\nDone."
        v = DraftConditionedVerifier(draft_runner=_OkRunner(draft_text), ising_sampler=None)
        result = v.verify_with_draft("What is 5+3?", "The answer is 8.")
        assert result.n_constraints > 0

    def test_energy_is_float(self) -> None:
        """energy field is a float."""
        v = DraftConditionedVerifier(draft_runner=_OkRunner(), ising_sampler=None)
        result = v.verify_with_draft("q", "r")
        assert isinstance(result.energy, float)

    def test_draft_text_stored(self) -> None:
        """draft_text field contains the runner output when draft_used=True."""
        expected = "Let x = 4.\nThen x = 4.\nAnswer is 4."
        v = DraftConditionedVerifier(draft_runner=_OkRunner(expected), ising_sampler=None)
        result = v.verify_with_draft("q", "r")
        assert result.draft_text == expected

    def test_constraints_list_empty_when_draft_unused(self) -> None:
        """constraints list is empty when draft_used=False."""
        v = DraftConditionedVerifier(draft_runner=_FailingRunner(), ising_sampler=None)
        result = v.verify_with_draft("q", "r")
        assert result.constraints == []


# ---------------------------------------------------------------------------
# condition_and_verify
# REQ-TIER28-001-5
# ---------------------------------------------------------------------------


class TestConditionAndVerify:
    """Tests for DraftConditionedVerifier.condition_and_verify() — pipeline interface."""

    def test_returns_dict(self) -> None:
        """REQ-TIER28-001-5: returns a plain dict."""
        v = DraftConditionedVerifier(draft_runner=_OkRunner(), ising_sampler=None)
        result = v.condition_and_verify("q", "r")
        assert isinstance(result, dict)

    def test_required_keys_present(self) -> None:
        """REQ-TIER28-001-5: dict contains energy, draft_used, n_constraints, draft_text, constraints."""
        v = DraftConditionedVerifier(draft_runner=_OkRunner(), ising_sampler=None)
        result = v.condition_and_verify("q", "r")
        for key in ("energy", "draft_used", "n_constraints", "draft_text", "constraints"):
            assert key in result, f"missing key: {key}"

    def test_energy_matches_verify_with_draft(self) -> None:
        """energy in condition_and_verify matches verify_with_draft energy."""
        runner = _OkRunner("x = 5.\nAnswer is 5.")
        v = DraftConditionedVerifier(draft_runner=runner, ising_sampler=None)
        direct = v.verify_with_draft("q", "r")
        via_dict = v.condition_and_verify("q", "r")
        # Both calls use the same runner/question/response so energies must match.
        assert via_dict["energy"] == pytest.approx(direct.energy)
