"""Tests for the Exp 1763 ROCE extractor.

Spec: REQ-EXTRACT-1763,
      SCENARIO-EXTRACT-1763, SCENARIO-EXTRACT-1764
"""

from __future__ import annotations

from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.roce_extractor import (
    ROCEConstraint,
    ROCEExtractor,
    _dedupe_constraints,
    extract_roce_constraints,
)


def _by_predicate(results: list[ConstraintResult]) -> dict[str, ConstraintResult]:
    return {result.metadata["predicate"]: result for result in results}


def _predicates(results: list[ConstraintResult]) -> set[str]:
    return {result.metadata["predicate"] for result in results}


def test_supported_domains_and_domain_filter() -> None:
    """REQ-EXTRACT-1763-1/5: ROCE supports prompt domains and skips others."""
    extractor = ROCEExtractor()

    assert extractor.supported_domains == ["roce", "prompt", "open_world"]
    assert extractor.extract("Return JSON.", domain="logic") == []

    results = extractor.extract("Return JSON.", domain="roce")
    assert _predicates(results) == {"format_json"}


def test_open_prompt_becomes_formal_constraints() -> None:
    """SCENARIO-EXTRACT-1763: Open prompt instructions become formal metadata."""
    prompt = (
        "Return a single-line JSON object only. "
        'Use strict key order {"final_answer": ..., "claims": [...]} '
        "and no other top-level keys. "
        'Do not mention "draft".'
    )

    results = ROCEExtractor().extract(prompt, domain="roce")
    by_predicate = _by_predicate(results)

    assert {"format_json", "json_required_keys", "forbidden_text", "single_line"} <= set(
        by_predicate
    )
    assert by_predicate["json_required_keys"].metadata["arguments"] == {
        "keys": ["final_answer", "claims"],
        "ordered": True,
        "no_extra_keys": True,
    }
    assert by_predicate["forbidden_text"].metadata["arguments"] == {"term": "draft"}

    for result in results:
        assert result.constraint_type.startswith("roce_")
        assert result.metadata["source"] == "roce"
        assert result.metadata["raw_phrase"]
        assert result.metadata["confidence"] == 1.0
        assert "REQ-EXTRACT-1763" in result.metadata["spec_refs"]


def test_numeric_list_word_and_boundary_constraints() -> None:
    """REQ-EXTRACT-1763-3: ROCE covers numeric, count, format, and boundary rules."""
    prompt = (
        "Give exactly three bullet points. "
        "Keep your answer under 50 words and write at least 10 words. "
        "The score must be between 0 and 1. "
        "Begin with 'Score:' and end with 'done'."
    )

    by_predicate = _by_predicate(ROCEExtractor().extract(prompt))

    assert by_predicate["format_list"].metadata["arguments"] == {"style": "bullet"}
    assert by_predicate["exact_item_count"].metadata["arguments"] == {"count": 3}
    assert by_predicate["word_count_at_most"].metadata["arguments"] == {"limit": 50}
    assert by_predicate["word_count_at_least"].metadata["arguments"] == {"limit": 10}
    assert by_predicate["numeric_range"].metadata["arguments"] == {
        "low": 0.0,
        "high": 1.0,
    }
    assert by_predicate["starts_with"].metadata["arguments"] == {"text": "Score:"}
    assert by_predicate["ends_with"].metadata["arguments"] == {"text": "done"}


def test_required_text_is_deduplicated() -> None:
    """SCENARIO-EXTRACT-1764: Repeated required text instructions dedupe."""
    prompt = "Your response must include the word 'summary'. It must contain the word summary."

    results = ROCEExtractor().extract(prompt)
    required = [r for r in results if r.metadata["predicate"] == "required_text"]

    assert len(required) == 1
    assert required[0].metadata["arguments"] == {"term": "summary"}


def test_final_answer_only_and_answer_type_constraints() -> None:
    """REQ-EXTRACT-1763-3: ROCE captures final-answer-only and typed output."""
    prompt = "Return only the final answer. Answer with a single number."

    by_predicate = _by_predicate(ROCEExtractor().extract(prompt, domain="open_world"))

    assert by_predicate["final_answer_only"].metadata["arguments"] == {}
    assert by_predicate["answer_type"].metadata["arguments"] == {"type": "number"}


def test_formal_constraint_conversion_and_helper() -> None:
    """REQ-EXTRACT-1763-2: ROCEConstraint converts to ConstraintResult schema."""
    formal = ROCEConstraint(
        kind="content",
        predicate="required_text",
        arguments={"term": "x"},
        description="Response must contain x",
        raw_phrase="must contain x",
    )

    result = formal.to_constraint_result()
    assert result.constraint_type == "roce_content"
    assert result.description == "Response must contain x"
    assert result.metadata["predicate"] == "required_text"
    assert result.metadata["arguments"] == {"term": "x"}

    prompt_results = extract_roce_constraints("Use a numbered list with exactly 2 items.")
    assert _predicates(prompt_results) == {"format_list", "exact_item_count"}

    key_results = extract_roce_constraints("Respond in JSON with keys 'name' and 'age'.")
    key_constraint = _by_predicate(key_results)["json_required_keys"]
    assert key_constraint.metadata["arguments"] == {
        "keys": ["name", "age"],
        "ordered": False,
        "no_extra_keys": False,
    }


def test_dedupe_normalizes_nested_arguments() -> None:
    """REQ-EXTRACT-1763-4: Deduplication normalizes nested argument values."""
    first = ROCEConstraint(
        kind="schema",
        predicate="nested",
        arguments={"schema": {"Key": "Value"}},
        description="first",
        raw_phrase="first",
    )
    second = ROCEConstraint(
        kind="schema",
        predicate="nested",
        arguments={"schema": {"key": "value"}},
        description="second",
        raw_phrase="second",
    )

    assert _dedupe_constraints([first, second]) == [first]


def test_empty_prompt_returns_no_constraints() -> None:
    """REQ-EXTRACT-1763-5: Empty prompts produce no open-world constraints."""
    assert ROCEExtractor().extract("") == []
    assert extract_roce_constraints("   ") == []
