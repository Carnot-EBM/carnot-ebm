"""Tests for carnot.pipeline.constraint_extractor.

Verifies zero false accepts for all 10 canonical instruction types and
confirms CI/live mode behaviour.

Spec: REQ-EXTRACT-055,
      SCENARIO-EXTRACT-094, SCENARIO-EXTRACT-095
"""

from __future__ import annotations

import json
import os

import pytest

from carnot.pipeline.constraint_extractor import (
    DynamicConstraint,
    PromptConstraintExtractor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_extractor() -> PromptConstraintExtractor:
    """Return a PromptConstraintExtractor in CI mode (no LLM calls)."""
    return PromptConstraintExtractor()


# ---------------------------------------------------------------------------
# DynamicConstraint.check() — unit tests per type
# ---------------------------------------------------------------------------


class TestMustContain:
    """REQ-EXTRACT-055-2: must_contain checker.  Zero false accepts."""

    def test_satisfied_when_term_present(self):
        # SCENARIO-EXTRACT-094: term appears in response → satisfied
        c = DynamicConstraint("must_contain", "desc", {"term": "hello"})
        assert c.check("hello world") is True

    def test_violated_when_term_absent(self):
        # SCENARIO-EXTRACT-095: term absent → violated (zero false accept)
        c = DynamicConstraint("must_contain", "desc", {"term": "hello"})
        assert c.check("goodbye world") is False

    def test_case_insensitive(self):
        c = DynamicConstraint("must_contain", "desc", {"term": "Hello"})
        assert c.check("This HELLO you.") is True

    def test_empty_response_is_violated(self):
        c = DynamicConstraint("must_contain", "desc", {"term": "hello"})
        assert c.check("") is False


class TestMustNotContain:
    """REQ-EXTRACT-055-2: must_not_contain checker.  Zero false accepts."""

    def test_satisfied_when_term_absent(self):
        c = DynamicConstraint("must_not_contain", "desc", {"term": "secret"})
        assert c.check("This is fine.") is True

    def test_violated_when_term_present(self):
        # SCENARIO-EXTRACT-095: forbidden term appears → violated
        c = DynamicConstraint("must_not_contain", "desc", {"term": "secret"})
        assert c.check("The secret is out.") is False

    def test_empty_response_satisfied(self):
        c = DynamicConstraint("must_not_contain", "desc", {"term": "secret"})
        assert c.check("") is True

    def test_empty_term_always_satisfied(self):
        # Line 141: empty term → no forbidden content → True
        c = DynamicConstraint("must_not_contain", "desc", {"term": ""})
        assert c.check("anything") is True


class TestFormatJson:
    """REQ-EXTRACT-055-2: format_json checker.  Zero false accepts."""

    def test_satisfied_valid_json_object(self):
        c = DynamicConstraint("format_json", "desc")
        assert c.check('{"key": "value"}') is True

    def test_satisfied_valid_json_array(self):
        c = DynamicConstraint("format_json", "desc")
        assert c.check('[1, 2, 3]') is True

    def test_violated_plain_text(self):
        # SCENARIO-EXTRACT-095: plain prose is not JSON → violated
        c = DynamicConstraint("format_json", "desc")
        assert c.check("Here is some plain text answer.") is False

    def test_satisfied_fenced_json(self):
        c = DynamicConstraint("format_json", "desc")
        fenced = '```json\n{"a": 1}\n```'
        assert c.check(fenced) is True

    def test_violated_empty_response(self):
        c = DynamicConstraint("format_json", "desc")
        assert c.check("") is False


class TestFormatList:
    """REQ-EXTRACT-055-2: format_list checker.  Zero false accepts."""

    def test_satisfied_numbered_list(self):
        c = DynamicConstraint("format_list", "desc")
        assert c.check("1. First item\n2. Second item") is True

    def test_satisfied_bullet_list(self):
        c = DynamicConstraint("format_list", "desc")
        assert c.check("- Apple\n- Banana\n- Cherry") is True

    def test_violated_plain_paragraph(self):
        # SCENARIO-EXTRACT-095: no list markers → violated
        c = DynamicConstraint("format_list", "desc")
        assert c.check("Here is my answer with no list formatting.") is False


class TestMaxWords:
    """REQ-EXTRACT-055-2: max_words checker.  Zero false accepts."""

    def test_satisfied_within_limit(self):
        c = DynamicConstraint("max_words", "desc", {"limit": 10})
        assert c.check("This is a short response.") is True  # 5 words

    def test_violated_over_limit(self):
        # SCENARIO-EXTRACT-095: too many words → violated
        c = DynamicConstraint("max_words", "desc", {"limit": 3})
        assert c.check("This response has more than three words in total.") is False

    def test_exactly_at_limit_satisfied(self):
        c = DynamicConstraint("max_words", "desc", {"limit": 4})
        assert c.check("one two three four") is True


class TestMinWords:
    """REQ-EXTRACT-055-2: min_words checker.  Zero false accepts."""

    def test_satisfied_above_minimum(self):
        c = DynamicConstraint("min_words", "desc", {"limit": 3})
        assert c.check("one two three four") is True

    def test_violated_below_minimum(self):
        # SCENARIO-EXTRACT-095: too few words → violated
        c = DynamicConstraint("min_words", "desc", {"limit": 50})
        assert c.check("Short answer.") is False

    def test_exactly_at_minimum_satisfied(self):
        c = DynamicConstraint("min_words", "desc", {"limit": 3})
        assert c.check("one two three") is True


class TestNumericRange:
    """REQ-EXTRACT-055-2: numeric_range checker.  Zero false accepts."""

    def test_satisfied_number_in_range(self):
        c = DynamicConstraint("numeric_range", "desc", {"low": 1.0, "high": 10.0})
        assert c.check("The answer is 5.") is True

    def test_violated_number_out_of_range(self):
        # SCENARIO-EXTRACT-095: number outside [1, 10] → violated
        c = DynamicConstraint("numeric_range", "desc", {"low": 1.0, "high": 10.0})
        assert c.check("The answer is 99.") is False

    def test_violated_no_number_in_response(self):
        # No number present → cannot satisfy a numeric range constraint
        c = DynamicConstraint("numeric_range", "desc", {"low": 1.0, "high": 10.0})
        assert c.check("There are no numbers here.") is False

    def test_satisfied_boundary_value(self):
        c = DynamicConstraint("numeric_range", "desc", {"low": 0.0, "high": 100.0})
        assert c.check("Exactly 0") is True


class TestStartsWith:
    """REQ-EXTRACT-055-2: starts_with checker.  Zero false accepts."""

    def test_satisfied_correct_prefix(self):
        c = DynamicConstraint("starts_with", "desc", {"prefix": "Dear"})
        assert c.check("Dear reader, welcome.") is True

    def test_violated_wrong_prefix(self):
        # SCENARIO-EXTRACT-095: missing required prefix → violated
        c = DynamicConstraint("starts_with", "desc", {"prefix": "Dear"})
        assert c.check("Hello reader, welcome.") is False

    def test_case_insensitive(self):
        c = DynamicConstraint("starts_with", "desc", {"prefix": "DEAR"})
        assert c.check("dear reader") is True


class TestEndsWith:
    """REQ-EXTRACT-055-2: ends_with checker.  Zero false accepts."""

    def test_satisfied_correct_suffix(self):
        c = DynamicConstraint("ends_with", "desc", {"suffix": "goodbye"})
        assert c.check("Hello and goodbye") is True

    def test_violated_wrong_suffix(self):
        # SCENARIO-EXTRACT-095: missing required suffix → violated
        c = DynamicConstraint("ends_with", "desc", {"suffix": "goodbye"})
        assert c.check("Hello and farewell") is False

    def test_case_insensitive(self):
        c = DynamicConstraint("ends_with", "desc", {"suffix": "GOODBYE"})
        assert c.check("Hello and GOODBYE") is True


class TestNoRepetition:
    """REQ-EXTRACT-055-2: no_repetition checker.  Zero false accepts."""

    def test_satisfied_unique_list_items(self):
        c = DynamicConstraint("no_repetition", "desc")
        assert c.check("1. Apple\n2. Banana\n3. Cherry") is True

    def test_violated_repeated_list_items(self):
        # SCENARIO-EXTRACT-095: same item appears twice → violated
        c = DynamicConstraint("no_repetition", "desc")
        assert c.check("1. Apple\n2. Banana\n3. Apple") is False

    def test_satisfied_prose_no_repeated_sentences(self):
        c = DynamicConstraint("no_repetition", "desc")
        assert c.check("The sky is blue. Water is wet.") is True

    def test_violated_repeated_sentences(self):
        c = DynamicConstraint("no_repetition", "desc")
        assert c.check("The sky is blue. The sky is blue.") is False


class TestLLMExtractedAndUnknownType:
    """Lines 207-217: llm_extracted → True; unknown type → True."""

    def test_llm_extracted_check_always_true(self):
        c = DynamicConstraint("llm_extracted", "avoid passive voice", {"raw": "avoid passive voice"})
        assert c.check("This was done by us.") is True

    def test_unknown_instruction_type_returns_true(self):
        # Conservative pass for unrecognised types — no blocking without evidence.
        c = DynamicConstraint("future_type_not_yet_implemented", "desc")
        assert c.check("any response") is True


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-095: zero-false-accepts table test across all 10 types
# ---------------------------------------------------------------------------


_ZERO_FA_CASES: list[tuple[str, DynamicConstraint, str]] = [
    (
        "must_contain",
        DynamicConstraint("must_contain", "d", {"term": "summary"}),
        "This is the end.",  # no "summary" → violated
    ),
    (
        "must_not_contain",
        DynamicConstraint("must_not_contain", "d", {"term": "secret"}),
        "The secret is revealed.",  # contains "secret" → violated
    ),
    (
        "format_json",
        DynamicConstraint("format_json", "d"),
        "Here is a plain text answer.",  # not JSON → violated
    ),
    (
        "format_list",
        DynamicConstraint("format_list", "d"),
        "No list here, just prose.",  # no list markers → violated
    ),
    (
        "max_words",
        DynamicConstraint("max_words", "d", {"limit": 3}),
        "This response is definitely too long to satisfy the constraint.",  # >3 words
    ),
    (
        "min_words",
        DynamicConstraint("min_words", "d", {"limit": 100}),
        "Too short.",  # <100 words → violated
    ),
    (
        "numeric_range",
        DynamicConstraint("numeric_range", "d", {"low": 1.0, "high": 10.0}),
        "The answer is 999.",  # 999 out of [1, 10] → violated
    ),
    (
        "starts_with",
        DynamicConstraint("starts_with", "d", {"prefix": "Dear"}),
        "Hello, this is wrong.",  # doesn't start with "Dear" → violated
    ),
    (
        "ends_with",
        DynamicConstraint("ends_with", "d", {"suffix": "goodbye"}),
        "Hello and farewell to all.",  # doesn't end with "goodbye" → violated
    ),
    (
        "no_repetition",
        DynamicConstraint("no_repetition", "d"),
        "1. Apple\n2. Banana\n3. Apple",  # Apple repeated → violated
    ),
]


@pytest.mark.parametrize("itype,constraint,violating_response", _ZERO_FA_CASES, ids=[c[0] for c in _ZERO_FA_CASES])
def test_zero_false_accepts(itype: str, constraint: DynamicConstraint, violating_response: str):
    """SCENARIO-EXTRACT-095: all 10 instruction types detect clear violations."""
    # check() must return False for a clearly violating response.
    result = constraint.check(violating_response)
    assert result is False, (
        f"False accept for type='{itype}': check() returned True "
        f"for a response that should be rejected.\n"
        f"  response: {violating_response!r}"
    )


# ---------------------------------------------------------------------------
# PromptConstraintExtractor.extract_from_prompt() tests
# ---------------------------------------------------------------------------


class TestPromptConstraintExtractorCI:
    """REQ-EXTRACT-055-4: CI mode — no LLM calls, only rule extractor.

    SCENARIO-EXTRACT-094: extract_from_prompt detects must_contain correctly.
    """

    def test_extracts_must_contain(self):
        # SCENARIO-EXTRACT-094
        extractor = make_extractor()
        prompt = "Your response must include the word 'summary'."
        constraints = extractor.extract_from_prompt(prompt)
        types = [c.instruction_type for c in constraints]
        assert "must_contain" in types
        mc = next(c for c in constraints if c.instruction_type == "must_contain")
        assert mc.metadata.get("term", "").lower() == "summary"

    def test_extracts_must_not_contain(self):
        extractor = make_extractor()
        prompt = "Do not include the word 'profanity' in your response."
        constraints = extractor.extract_from_prompt(prompt)
        types = [c.instruction_type for c in constraints]
        assert "must_not_contain" in types

    def test_extracts_format_json(self):
        extractor = make_extractor()
        prompt = "Respond in JSON format with keys 'name' and 'age'."
        constraints = extractor.extract_from_prompt(prompt)
        types = [c.instruction_type for c in constraints]
        assert "format_json" in types

    def test_extracts_format_list(self):
        extractor = make_extractor()
        prompt = "Give me a numbered list of five items."
        constraints = extractor.extract_from_prompt(prompt)
        types = [c.instruction_type for c in constraints]
        assert "format_list" in types

    def test_extracts_max_words(self):
        extractor = make_extractor()
        prompt = "Keep your answer under 50 words."
        constraints = extractor.extract_from_prompt(prompt)
        types = [c.instruction_type for c in constraints]
        assert "max_words" in types
        mw = next(c for c in constraints if c.instruction_type == "max_words")
        assert mw.metadata["limit"] == 50

    def test_extracts_min_words(self):
        extractor = make_extractor()
        prompt = "Write at least 100 words."
        constraints = extractor.extract_from_prompt(prompt)
        types = [c.instruction_type for c in constraints]
        assert "min_words" in types
        mw = next(c for c in constraints if c.instruction_type == "min_words")
        assert mw.metadata["limit"] == 100

    def test_extracts_numeric_range(self):
        extractor = make_extractor()
        prompt = "The answer must be between 1 and 10."
        constraints = extractor.extract_from_prompt(prompt)
        types = [c.instruction_type for c in constraints]
        assert "numeric_range" in types
        nr = next(c for c in constraints if c.instruction_type == "numeric_range")
        assert nr.metadata["low"] == 1.0
        assert nr.metadata["high"] == 10.0

    def test_extracts_starts_with(self):
        extractor = make_extractor()
        prompt = "Start your response with 'Dear reader'."
        constraints = extractor.extract_from_prompt(prompt)
        types = [c.instruction_type for c in constraints]
        assert "starts_with" in types
        sw = next(c for c in constraints if c.instruction_type == "starts_with")
        assert sw.metadata["prefix"].lower() == "dear reader"

    def test_extracts_ends_with(self):
        extractor = make_extractor()
        prompt = "End your response with 'goodbye'."
        constraints = extractor.extract_from_prompt(prompt)
        types = [c.instruction_type for c in constraints]
        assert "ends_with" in types
        ew = next(c for c in constraints if c.instruction_type == "ends_with")
        assert ew.metadata["suffix"].lower() == "goodbye"

    def test_extracts_no_repetition(self):
        extractor = make_extractor()
        prompt = "Do not repeat yourself. Each item must appear only once."
        constraints = extractor.extract_from_prompt(prompt)
        types = [c.instruction_type for c in constraints]
        assert "no_repetition" in types

    def test_empty_prompt_returns_no_constraints(self):
        extractor = make_extractor()
        constraints = extractor.extract_from_prompt("")
        assert constraints == []

    def test_no_llm_call_in_ci_mode(self, monkeypatch):
        """REQ-EXTRACT-055-4: generate_fn must NOT be called in CI mode."""
        called = []

        def mock_generate(prompt: str) -> str:
            called.append(prompt)
            return "[]"

        monkeypatch.delenv("CARNOT_FORCE_LIVE", raising=False)
        extractor = PromptConstraintExtractor(generate_fn=mock_generate)
        extractor.extract_from_prompt("Keep your answer under 50 words.")
        assert called == [], "generate_fn was called in CI mode — must not happen"


class TestPromptConstraintExtractorLiveMode:
    """REQ-EXTRACT-055-3: live mode appends LLM-extracted constraints."""

    def test_llm_results_appended(self, monkeypatch):
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

        llm_response = json.dumps(
            [{"instruction_type": "llm_extracted", "description": "No passive voice", "raw": "avoid passive voice"}]
        )

        def mock_generate(prompt: str) -> str:
            return llm_response

        extractor = PromptConstraintExtractor(generate_fn=mock_generate)
        constraints = extractor.extract_from_prompt("Please avoid passive voice.")
        types = [c.instruction_type for c in constraints]
        assert "llm_extracted" in types

    def test_llm_failure_does_not_break_extraction(self, monkeypatch):
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

        def mock_generate(prompt: str) -> str:
            raise RuntimeError("model not loaded")

        extractor = PromptConstraintExtractor(generate_fn=mock_generate)
        # Should not raise; rule-extractor results still returned.
        constraints = extractor.extract_from_prompt("Keep your answer under 10 words.")
        types = [c.instruction_type for c in constraints]
        assert "max_words" in types

    def test_llm_malformed_json_silently_skipped(self, monkeypatch):
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

        def mock_generate(prompt: str) -> str:
            return "not valid json {{{"

        extractor = PromptConstraintExtractor(generate_fn=mock_generate)
        constraints = extractor.extract_from_prompt("Keep your answer under 10 words.")
        # Rule extractor result should still be there; no crash.
        assert any(c.instruction_type == "max_words" for c in constraints)

    def test_llm_returns_non_list_json_silently_skipped(self, monkeypatch):
        """Line 668: JSON object (not array) → _parse_llm_output returns []."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

        def mock_generate(prompt: str) -> str:
            return json.dumps({"error": "not a list"})

        extractor = PromptConstraintExtractor(generate_fn=mock_generate)
        constraints = extractor.extract_from_prompt("Keep your answer under 10 words.")
        # Still has rule-extractor results; no crash.
        assert any(c.instruction_type == "max_words" for c in constraints)

    def test_llm_returns_array_with_non_dict_items_skipped(self, monkeypatch):
        """Line 673: array items that are not dicts are skipped."""
        monkeypatch.setenv("CARNOT_FORCE_LIVE", "1")

        def mock_generate(prompt: str) -> str:
            # Mix of valid dict and invalid non-dict items.
            return json.dumps(
                ["not a dict", 42, {"description": "valid constraint", "raw": "x"}]
            )

        extractor = PromptConstraintExtractor(generate_fn=mock_generate)
        constraints = extractor.extract_from_prompt("Keep your answer under 10 words.")
        llm_types = [c for c in constraints if c.instruction_type == "llm_extracted"]
        # Only the dict item should produce a constraint.
        assert len(llm_types) == 1
        assert llm_types[0].description == "valid constraint"


# ---------------------------------------------------------------------------
# PromptConstraintExtractor.check_response() tests
# ---------------------------------------------------------------------------


class TestCheckResponse:
    """REQ-EXTRACT-055-5: check_response returns violated constraints."""

    def test_all_satisfied_returns_empty(self):
        extractor = make_extractor()
        constraints = [
            DynamicConstraint("must_contain", "d", {"term": "hello"}),
        ]
        violated = extractor.check_response("hello world", constraints)
        assert violated == []

    def test_violated_constraint_returned(self):
        extractor = make_extractor()
        constraints = [
            DynamicConstraint("must_contain", "d", {"term": "hello"}),
            DynamicConstraint("format_json", "d"),
        ]
        violated = extractor.check_response("hello world", constraints)
        # must_contain is satisfied (hello present), format_json is violated (not JSON)
        assert len(violated) == 1
        assert violated[0].instruction_type == "format_json"

    def test_all_violated_all_returned(self):
        extractor = make_extractor()
        constraints = [
            DynamicConstraint("must_contain", "d", {"term": "missing_term"}),
            DynamicConstraint("format_json", "d"),
        ]
        violated = extractor.check_response("plain text that omits the required keyword", constraints)
        assert len(violated) == 2

    def test_empty_constraints_no_violations(self):
        extractor = make_extractor()
        violated = extractor.check_response("any response", [])
        assert violated == []
