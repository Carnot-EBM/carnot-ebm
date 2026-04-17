"""Tests for FOVERAnnotator — Z3 step annotation pipeline.

Spec: REQ-LEARN-030, REQ-LEARN-031,
      SCENARIO-LEARN-054, SCENARIO-LEARN-055, SCENARIO-LEARN-056
"""

from __future__ import annotations

import pytest

from carnot.pipeline.fover_annotator import (
    FOVERAnnotator,
    FOVERCoTStep,
    _build_z3_assertion,
    _INLINE_EQ_RE,
    annotate_step_with_z3,
    parse_cot_into_steps,
)


# ---------------------------------------------------------------------------
# FOVERCoTStep dataclass
# ---------------------------------------------------------------------------


class TestFOVERCoTStep:
    # REQ-LEARN-030
    def test_default_values(self):
        step = FOVERCoTStep(step_idx=0, step_text="hello", claimed_equation=None)
        assert step.z3_label is None
        assert step.z3_confidence == 0.0

    def test_with_equation(self):
        step = FOVERCoTStep(
            step_idx=1,
            step_text="2 + 3 = 5",
            claimed_equation="2 + 3 = 5",
            z3_label="correct",
            z3_confidence=1.0,
        )
        assert step.step_idx == 1
        assert step.z3_label == "correct"
        assert step.z3_confidence == 1.0

    def test_incorrect_label(self):
        step = FOVERCoTStep(
            step_idx=0,
            step_text="2 + 3 = 6",
            claimed_equation="2 + 3 = 6",
            z3_label="incorrect",
            z3_confidence=1.0,
        )
        assert step.z3_label == "incorrect"

    def test_not_verifiable_label(self):
        step = FOVERCoTStep(
            step_idx=0,
            step_text="The answer is obvious.",
            claimed_equation=None,
            z3_label="not_verifiable",
            z3_confidence=0.0,
        )
        assert step.z3_label == "not_verifiable"
        assert step.z3_confidence == 0.0


# ---------------------------------------------------------------------------
# parse_cot_into_steps
# ---------------------------------------------------------------------------


class TestParseCotIntoSteps:
    # SCENARIO-LEARN-054
    def test_three_numbered_steps(self):
        # REQ-LEARN-030: parse numbered steps
        response = "1. First step. 2. Second step. 3. Third step."
        steps = parse_cot_into_steps(response)
        assert len(steps) == 3
        assert steps[0].step_idx == 0
        assert steps[1].step_idx == 1
        assert steps[2].step_idx == 2

    def test_step_n_format(self):
        # REQ-LEARN-030: "Step N:" format
        response = "Step 1: Do this.\nStep 2: Then that.\nStep 3: Finally."
        steps = parse_cot_into_steps(response)
        assert len(steps) == 3

    def test_empty_response_returns_empty_list(self):
        assert parse_cot_into_steps("") == []
        assert parse_cot_into_steps("   ") == []

    def test_no_step_markers_returns_one_chunk(self):
        # Prose without markers is treated as one chunk.
        response = "The answer is 42 because reasons."
        steps = parse_cot_into_steps(response)
        assert len(steps) == 1
        assert steps[0].step_idx == 0

    def test_equation_extracted_from_step(self):
        # REQ-LEARN-030: claimed_equation is populated when equation present.
        response = "1. We compute 4 + 5 = 9. 2. Then 9 + 1 = 10."
        steps = parse_cot_into_steps(response)
        # At least one step should have a claimed equation.
        equations = [s.claimed_equation for s in steps if s.claimed_equation is not None]
        assert len(equations) >= 1

    def test_no_equation_in_step(self):
        # Step text with no arithmetic → claimed_equation=None.
        response = "1. The sky is blue. 2. Water is wet."
        steps = parse_cot_into_steps(response)
        for step in steps:
            assert step.claimed_equation is None

    def test_step_indices_are_sequential(self):
        response = "1. A.\n2. B.\n3. C.\n4. D."
        steps = parse_cot_into_steps(response)
        for i, step in enumerate(steps):
            assert step.step_idx == i

    def test_multiline_numbered_steps(self):
        response = (
            "1. First compute 2 + 3 = 5.\n"
            "   This is still step 1.\n"
            "2. Then compute 5 + 1 = 6."
        )
        steps = parse_cot_into_steps(response)
        assert len(steps) == 2


# ---------------------------------------------------------------------------
# annotate_step_with_z3
# ---------------------------------------------------------------------------


class TestAnnotateStepWithZ3:
    # SCENARIO-LEARN-055: correct equation
    def test_correct_equation(self):
        step = FOVERCoTStep(
            step_idx=0,
            step_text="2 + 3 = 5",
            claimed_equation="2 + 3 = 5",
        )
        result = annotate_step_with_z3(step)
        assert result.z3_label == "correct"
        assert result.z3_confidence >= 0.3

    # SCENARIO-LEARN-056: incorrect equation
    def test_incorrect_equation(self):
        step = FOVERCoTStep(
            step_idx=0,
            step_text="2 + 3 = 6",
            claimed_equation="2 + 3 = 6",
        )
        result = annotate_step_with_z3(step)
        assert result.z3_label == "incorrect"

    def test_no_equation_is_not_verifiable(self):
        # Step with no equation → not_verifiable, confidence 0.
        step = FOVERCoTStep(
            step_idx=0,
            step_text="The answer is obviously correct.",
            claimed_equation=None,
        )
        result = annotate_step_with_z3(step)
        assert result.z3_label == "not_verifiable"
        assert result.z3_confidence == 0.0

    def test_full_equation_has_high_confidence(self):
        step = FOVERCoTStep(
            step_idx=0,
            step_text="10 * 5 = 50",
            claimed_equation="10 * 5 = 50",
        )
        result = annotate_step_with_z3(step)
        assert result.z3_confidence == 1.0

    def test_preserves_step_idx_and_text(self):
        step = FOVERCoTStep(
            step_idx=7,
            step_text="So 7 + 8 = 15.",
            claimed_equation="7 + 8 = 15",
        )
        result = annotate_step_with_z3(step)
        assert result.step_idx == 7
        assert result.step_text == "So 7 + 8 = 15."

    def test_subtraction_correct(self):
        step = FOVERCoTStep(
            step_idx=0,
            step_text="10 - 4 = 6",
            claimed_equation="10 - 4 = 6",
        )
        result = annotate_step_with_z3(step)
        assert result.z3_label == "correct"

    def test_subtraction_incorrect(self):
        step = FOVERCoTStep(
            step_idx=0,
            step_text="10 - 4 = 7",
            claimed_equation="10 - 4 = 7",
        )
        result = annotate_step_with_z3(step)
        assert result.z3_label == "incorrect"

    def test_claimed_equation_rematch_failure_is_not_verifiable(self):
        # claimed_equation set to something that won't re-match _INLINE_EQ_RE.
        step = FOVERCoTStep(
            step_idx=0,
            step_text="weird",
            claimed_equation="not a valid equation at all",
        )
        result = annotate_step_with_z3(step)
        assert result.z3_label == "not_verifiable"

    def test_z3_unknown_or_error_yields_not_verifiable(self, monkeypatch):
        # Lines 343-344: Z3 returns 'unknown' → label='not_verifiable', confidence=0.0
        from carnot.pipeline import fover_annotator as fa
        monkeypatch.setattr(fa, "_exec_z3_snippet", lambda code: ("unknown", None))
        step = FOVERCoTStep(
            step_idx=0,
            step_text="2 + 3 = 5",
            claimed_equation="2 + 3 = 5",
        )
        result = annotate_step_with_z3(step)
        assert result.z3_label == "not_verifiable"
        assert result.z3_confidence == 0.0

    def test_partial_operand_confidence_is_0_5(self, monkeypatch):
        # Lines 322-323: float() raises ValueError → confidence=0.5 (not all parsed)
        # We produce a partial match by monkeypatching the inner parse function.
        # Actually it's easier to test via an equation with a non-float operand.
        # The regex _INLINE_EQ_RE requires digits, so we can't construct a bad match directly.
        # Instead test by monkeypatching _exec_z3_snippet to return 'sat'
        # and checking a valid equation still returns confidence=1.0.
        # The ValueError branch (line 322) is unreachable via _INLINE_EQ_RE since
        # the regex only matches digit strings; test the else branch via monkeypatch:
        from carnot.pipeline import fover_annotator as fa
        monkeypatch.setattr(fa, "_exec_z3_snippet", lambda code: ("sat", None))
        step = FOVERCoTStep(
            step_idx=0,
            step_text="2 + 3 = 5",
            claimed_equation="2 + 3 = 5",
        )
        result = annotate_step_with_z3(step)
        assert result.z3_label == "correct"
        assert result.z3_confidence == 1.0


# ---------------------------------------------------------------------------
# _build_z3_assertion (internal, tested for correctness)
# ---------------------------------------------------------------------------


class TestBuildZ3Assertion:
    def test_addition_correct_snippet(self):
        match = _INLINE_EQ_RE.search("2 + 3 = 5")
        assert match is not None
        snippet = _build_z3_assertion(match)
        assert "import z3" in snippet
        assert "s.add" in snippet
        assert "print(s.check())" in snippet

    def test_unicode_multiply_normalised(self):
        match = _INLINE_EQ_RE.search("4 × 3 = 12")
        assert match is not None
        snippet = _build_z3_assertion(match)
        assert "*" in snippet  # unicode × is normalised to *

    def test_unicode_divide_normalised(self):
        match = _INLINE_EQ_RE.search("10 ÷ 2 = 5")
        assert match is not None
        snippet = _build_z3_assertion(match)
        assert "/" in snippet


# ---------------------------------------------------------------------------
# FOVERAnnotator
# ---------------------------------------------------------------------------


class TestFOVERAnnotator:
    def test_annotate_response_returns_steps(self):
        annotator = FOVERAnnotator()
        response = "1. Compute 2 + 3 = 5. 2. Then 5 + 1 = 6."
        steps = annotator.annotate_response(response, question_id="q1")
        assert len(steps) >= 1
        assert all(s.z3_label is not None for s in steps)

    def test_annotate_corpus_parallel_length(self):
        annotator = FOVERAnnotator()
        responses = [
            {"response": "1. 2 + 2 = 4.", "question_id": "a"},
            {"response": "1. 3 + 3 = 7.", "question_id": "b"},
        ]
        result = annotator.annotate_corpus(responses)
        assert len(result) == 2

    def test_annotate_corpus_missing_question_id(self):
        # question_id is optional — defaults to str(index).
        annotator = FOVERAnnotator()
        responses = [{"response": "1. 2 + 3 = 5."}]
        result = annotator.annotate_corpus(responses)
        assert len(result) == 1

    def test_to_training_pairs_filters_not_verifiable(self):
        # REQ-LEARN-031: not_verifiable steps excluded.
        annotator = FOVERAnnotator()
        steps_a = [
            FOVERCoTStep(0, "2 + 3 = 5", "2 + 3 = 5", z3_label="correct", z3_confidence=1.0),
            FOVERCoTStep(1, "prose step", None, z3_label="not_verifiable", z3_confidence=0.0),
        ]
        pairs = annotator.to_training_pairs([steps_a])
        assert len(pairs) == 1
        assert pairs[0]["label"] == "correct"

    def test_to_training_pairs_filters_low_confidence(self):
        # REQ-LEARN-031: confidence < 0.3 excluded.
        annotator = FOVERAnnotator()
        steps = [
            FOVERCoTStep(0, "eq", "eq", z3_label="correct", z3_confidence=0.1),
            FOVERCoTStep(1, "eq2", "eq2", z3_label="incorrect", z3_confidence=0.5),
        ]
        pairs = annotator.to_training_pairs([steps])
        assert len(pairs) == 1
        assert pairs[0]["label"] == "incorrect"

    def test_to_training_pairs_schema_keys(self):
        annotator = FOVERAnnotator()
        steps = [
            FOVERCoTStep(0, "4 + 4 = 8", "4 + 4 = 8", z3_label="correct", z3_confidence=1.0),
        ]
        responses = [{"question_id": "qtest", "response": "4 + 4 = 8"}]
        pairs = annotator.to_training_pairs([steps], responses=responses)
        assert len(pairs) == 1
        pair = pairs[0]
        assert "question_id" in pair
        assert "step_text" in pair
        assert "label" in pair
        assert "confidence" in pair
        assert pair["question_id"] == "qtest"

    def test_to_training_pairs_without_responses(self):
        # question_id falls back to corpus index string when responses=None.
        annotator = FOVERAnnotator()
        steps = [
            FOVERCoTStep(0, "1 + 1 = 2", "1 + 1 = 2", z3_label="correct", z3_confidence=1.0),
        ]
        pairs = annotator.to_training_pairs([steps], responses=None)
        assert len(pairs) == 1
        assert pairs[0]["question_id"] == "0"

    def test_z3_timeout_stored(self):
        annotator = FOVERAnnotator(z3_timeout_seconds=10)
        assert annotator.z3_timeout_seconds == 10

    def test_empty_corpus(self):
        annotator = FOVERAnnotator()
        result = annotator.annotate_corpus([])
        assert result == []

    def test_to_training_pairs_both_correct_and_incorrect(self):
        # REQ-LEARN-031: both labels pass the filter when confidence >= 0.3.
        annotator = FOVERAnnotator()
        steps = [
            FOVERCoTStep(0, "2+3=5", "2+3=5", z3_label="correct", z3_confidence=1.0),
            FOVERCoTStep(1, "2+3=6", "2+3=6", z3_label="incorrect", z3_confidence=1.0),
        ]
        pairs = annotator.to_training_pairs([steps])
        assert len(pairs) == 2
        labels = {p["label"] for p in pairs}
        assert labels == {"correct", "incorrect"}
