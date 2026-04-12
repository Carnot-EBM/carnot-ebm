"""Tests for `carnot.pipeline.typed_reasoning`.

Spec: REQ-VERIFY-015, REQ-VERIFY-016, REQ-VERIFY-017,
SCENARIO-VERIFY-015, SCENARIO-VERIFY-016, SCENARIO-VERIFY-017
"""

from __future__ import annotations

from unittest.mock import patch

import carnot.pipeline.typed_reasoning as typed_reasoning
import pytest
from carnot.pipeline.typed_reasoning import (
    AtomicClaim,
    ExtractionProvenance,
    FinalAnswer,
    ReasoningStep,
    TypedReasoningExtractor,
    TypedReasoningIR,
    UserConstraint,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline

# ---------------------------------------------------------------------------
# Direct JSON path -- REQ-VERIFY-015, REQ-VERIFY-016, SCENARIO-VERIFY-015
# ---------------------------------------------------------------------------


def test_direct_json_payload_parses_and_round_trips_deterministically() -> None:
    """SCENARIO-VERIFY-015: Direct JSON preserves typed sections and round-trips."""
    extractor = TypedReasoningExtractor(parser_version="20260412")
    response = """```json
{
  "user_constraints": [
    "Return JSON only.",
    {"constraint_id": "uc2", "kind": "content_requirement", "text": "Include confidence."}
  ],
  "reasoning_steps": [
    {"step_id": "s1", "kind": "arithmetic", "text": "47 + 28 = 75", "claim_ids": ["cl1"]},
    {"step_id": "s2", "kind": "finalization", "text": "Return 75", "claim_ids": ["cl2"]}
  ],
  "atomic_claims": [
    {"claim_id": "cl1", "kind": "equation", "text": "47 + 28 = 75", "value": 75, "step_id": "s1"},
    {
      "claim_id": "cl2",
      "kind": "answer_binding",
      "text": "answer = 75",
      "value": 75,
      "step_id": "s2"
    }
  ],
  "final_answer": {"text": "75", "normalized": 75, "answer_type": "number", "source_step_id": "s2"}
}
```"""

    ir = extractor.extract(
        question="Return JSON only with answer and confidence.",
        response=response,
    )

    assert ir.provenance.extraction_method == "direct_json"
    assert ir.provenance.parser_version == "20260412"
    assert [constraint.constraint_id for constraint in ir.user_constraints] == [
        "uc1",
        "uc2",
    ]
    assert ir.reasoning_steps[0].claim_ids == ["cl1"]
    assert ir.atomic_claims[0].step_id == "s1"
    assert ir.final_answer is not None
    assert ir.final_answer.normalized == 75

    encoded = ir.to_json()
    assert encoded == ir.to_json()

    restored = TypedReasoningIR.from_json(encoded)
    assert restored.to_dict() == ir.to_dict()


def test_direct_json_uses_prompt_constraints_when_payload_omits_them() -> None:
    """REQ-VERIFY-016: Direct JSON falls back to prompt constraints when needed."""
    extractor = TypedReasoningExtractor(parser_version="20260412")
    response = """{
      "steps": ["7 + 8 = 15"],
      "claims": ["7 + 8 = 15"],
      "answer": "15"
    }"""

    ir = extractor.extract(
        question="Return JSON only and do not use markdown fences.",
        response=response,
    )

    assert ir.provenance.extraction_method == "direct_json"
    assert len(ir.user_constraints) >= 1
    assert any("Return JSON only" in constraint.text for constraint in ir.user_constraints)
    assert ir.final_answer is not None
    assert ir.final_answer.normalized == 15


# ---------------------------------------------------------------------------
# Fallback path -- REQ-VERIFY-015, REQ-VERIFY-016, SCENARIO-VERIFY-016
# ---------------------------------------------------------------------------


def test_fallback_text_extracts_constraints_steps_claims_and_final_answer() -> None:
    """SCENARIO-VERIFY-016: Plain text falls back to post-hoc parsing."""
    extractor = TypedReasoningExtractor(parser_version="20260412")
    question = "Give exactly 3 bullet points and do not use urgent."
    response = "REASONING:\n1. 27 - 15 = 12.\n2. 12 / 3 = 4.\n3. 4 / 2 = 2.\nFINAL: 2"

    ir = extractor.extract(question=question, response=response)

    assert ir.provenance.extraction_method == "fallback_text"
    assert any("exactly 3 bullet points" in constraint.text for constraint in ir.user_constraints)
    assert any("do not use urgent" in constraint.text for constraint in ir.user_constraints)
    assert [step.step_id for step in ir.reasoning_steps] == ["s1", "s2", "s3"]
    assert [claim.step_id for claim in ir.atomic_claims] == ["s1", "s2", "s3"]
    assert ir.final_answer is not None
    assert ir.final_answer.text == "2"
    assert ir.final_answer.normalized == 2
    assert ir.final_answer.answer_type == "number"


def test_fallback_without_markers_becomes_single_step() -> None:
    """REQ-VERIFY-016: Plain responses without markers still produce a valid IR."""
    extractor = TypedReasoningExtractor(parser_version="20260412")

    ir = extractor.extract(
        question="Answer in one sentence.",
        response="The answer is 9.",
    )

    assert ir.provenance.extraction_method == "fallback_text"
    assert len(ir.reasoning_steps) == 1
    assert len(ir.atomic_claims) == 1
    assert ir.final_answer is not None
    assert ir.final_answer.normalized == 9


# ---------------------------------------------------------------------------
# Validation and pipeline integration -- REQ-VERIFY-017, SCENARIO-VERIFY-017
# ---------------------------------------------------------------------------


def test_validation_rejects_duplicate_ids_and_bad_references() -> None:
    """REQ-VERIFY-017: Validation rejects duplicate identifiers and bad links."""
    duplicate_steps = TypedReasoningIR(
        question="q",
        user_constraints=[UserConstraint("uc1", "prompt_constraint", "Do the task.")],
        reasoning_steps=[
            ReasoningStep("s1", "analysis", "first"),
            ReasoningStep("s1", "analysis", "duplicate"),
        ],
        atomic_claims=[],
        final_answer=None,
        provenance=ExtractionProvenance("fallback_text", "plain_text", "20260412"),
    )
    with pytest.raises(ValueError, match="duplicate step_id"):
        duplicate_steps.validate()

    bad_reference = TypedReasoningIR(
        question="q",
        user_constraints=[UserConstraint("uc1", "prompt_constraint", "Do the task.")],
        reasoning_steps=[ReasoningStep("s1", "analysis", "first")],
        atomic_claims=[
            AtomicClaim("cl1", "equation", "1 + 1 = 2", value=2, step_id="missing"),
        ],
        final_answer=FinalAnswer(
            text="2",
            normalized=2,
            answer_type="number",
            source_step_id="missing",
        ),
        provenance=ExtractionProvenance("direct_json", "json", "20260412"),
    )
    with pytest.raises(ValueError, match="unknown step_id"):
        bad_reference.validate()


def test_validation_and_parser_helpers_cover_error_paths() -> None:
    """REQ-VERIFY-017: Helper paths fail deterministically on invalid IR."""
    invalid_method = TypedReasoningIR(
        question="q",
        user_constraints=[],
        reasoning_steps=[],
        atomic_claims=[],
        final_answer=None,
        provenance=ExtractionProvenance("unknown", "plain_text", "20260412"),
    )
    with pytest.raises(ValueError, match="unknown extraction_method"):
        invalid_method.validate()

    duplicate_claims = TypedReasoningIR(
        question="q",
        user_constraints=[],
        reasoning_steps=[ReasoningStep("s1", "analysis", "step", ["cl1"])],
        atomic_claims=[
            AtomicClaim("cl1", "statement", "first", step_id="s1"),
            AtomicClaim("cl1", "statement", "duplicate", step_id="s1"),
        ],
        final_answer=None,
        provenance=ExtractionProvenance("fallback_text", "plain_text", "20260412"),
    )
    with pytest.raises(ValueError, match="duplicate claim_id"):
        duplicate_claims.validate()

    missing_claim = TypedReasoningIR(
        question="q",
        user_constraints=[],
        reasoning_steps=[ReasoningStep("s1", "analysis", "step", ["missing"])],
        atomic_claims=[],
        final_answer=None,
        provenance=ExtractionProvenance("fallback_text", "plain_text", "20260412"),
    )
    with pytest.raises(ValueError, match="unknown claim_id"):
        missing_claim.validate()

    bad_final = TypedReasoningIR(
        question="q",
        user_constraints=[],
        reasoning_steps=[ReasoningStep("s1", "analysis", "step")],
        atomic_claims=[],
        final_answer=FinalAnswer(
            text="2",
            normalized=2,
            answer_type="number",
            source_step_id="missing",
        ),
        provenance=ExtractionProvenance("fallback_text", "plain_text", "20260412"),
    )
    with pytest.raises(ValueError, match="unknown step_id"):
        bad_final.validate()

    with pytest.raises(ValueError, match="typed reasoning JSON payload not found"):
        TypedReasoningIR.from_json("not json")


def test_helper_branches_cover_direct_json_fallbacks_and_normalization() -> None:
    """REQ-VERIFY-016: Helper paths cover direct JSON aliases and text fallbacks."""
    extractor = TypedReasoningExtractor(parser_version="20260412")
    no_claims_or_answer = extractor.extract(
        question=".",
        response='{"checks": [{"constraint": "scope", "evidence": "use total amount"}]}',
    )

    assert no_claims_or_answer.provenance.extraction_method == "direct_json"
    assert len(no_claims_or_answer.user_constraints) == 1
    assert no_claims_or_answer.user_constraints[0].text == "."
    assert no_claims_or_answer.reasoning_steps[0].text == "scope: use total amount"
    assert no_claims_or_answer.atomic_claims[0].text == "scope: use total amount"
    assert no_claims_or_answer.final_answer is None

    empty_ir = extractor.extract(question="", response="")
    assert empty_ir.user_constraints == []
    assert empty_ir.reasoning_steps == []
    assert empty_ir.atomic_claims == []
    assert empty_ir.final_answer is None

    assert typed_reasoning._coerce_steps(None) == []
    assert typed_reasoning._coerce_claims(None, []) == []
    assert typed_reasoning._coerce_final_answer(None, []) is None
    assert len(typed_reasoning._infer_prompt_constraints("First.\n\nSecond.")) == 2
    assert typed_reasoning._extract_json_dict('prefix {"answer": 1} suffix') == {"answer": 1}
    assert typed_reasoning._extract_json_dict("prefix {not json") is None
    assert typed_reasoning._extract_json_dict("") is None
    assert typed_reasoning._strip_fence("plain text") == "plain text"

    assert typed_reasoning._normalized_value(True) is True
    assert typed_reasoning._normalized_value(3) == 3
    assert typed_reasoning._normalized_value(3.0) == 3
    assert typed_reasoning._normalized_value([1, 2]) == [1, 2]
    assert typed_reasoning._normalized_value({"a": 1}) == {"a": 1}
    assert typed_reasoning._normalized_value("") is None
    assert typed_reasoning._normalized_value("7") == 7
    assert typed_reasoning._normalized_value("3.5") == 3.5
    assert typed_reasoning._normalized_value('{"a": 1}') == {"a": 1}
    assert typed_reasoning._normalized_value("plain text") == "plain text"

    assert typed_reasoning._answer_type(None) == "unknown"
    assert typed_reasoning._answer_type(True) == "boolean"
    assert typed_reasoning._answer_type([1]) == "list"
    assert typed_reasoning._answer_type({"a": 1}) == "json_object"
    assert typed_reasoning._answer_type("plain text") == "text"

    assert typed_reasoning._constraint_kind("Include owner.") == "content_requirement"
    assert typed_reasoning._step_kind("Return 5") == "finalization"
    assert typed_reasoning._claim_kind("A implies B") == "statement"
    assert typed_reasoning._step_text_from_dict({"description": "desc"}) == "desc"
    assert (
        typed_reasoning._step_text_from_dict({"constraint": "scope", "evidence": "total"})
        == "scope: total"
    )
    assert typed_reasoning._step_text_from_dict({"evidence": "total"}) == "total"
    assert typed_reasoning._step_text_from_dict({"constraint": "scope"}) == "scope"
    assert typed_reasoning._step_text_from_dict({}) == ""
    assert typed_reasoning._clean_reasoning_line("2) value") == "value"

    with pytest.raises(ValueError, match="not JSON-serializable"):
        typed_reasoning._ensure_jsonable(object())

    assert typed_reasoning._as_text(None) == ""
    assert typed_reasoning._optional_text(None) is None
    assert typed_reasoning._as_dict(1) == {}
    assert typed_reasoning._as_dict_list("bad") == []
    assert typed_reasoning._string_list("bad") == []


def test_verify_pipeline_surfaces_typed_reasoning_without_breaking_extractors() -> None:
    """SCENARIO-VERIFY-017: VerifyRepairPipeline adds IR without changing verdicts."""
    pipeline = VerifyRepairPipeline()
    response = """{
      "steps": ["47 + 28 = 75", "Return 75"],
      "claims": ["47 + 28 = 75"],
      "answer": "75"
    }"""

    ir = pipeline.extract_typed_reasoning(
        question="Return JSON only with the answer.",
        response=response,
    )
    assert ir is not None
    assert ir.provenance.extraction_method == "direct_json"

    result = pipeline.verify(
        question="What is 47 + 28?",
        response=response,
        domain="arithmetic",
    )

    assert result.verified is True
    assert result.typed_reasoning is not None
    assert result.typed_reasoning.provenance.extraction_method == "direct_json"
    assert result.typed_reasoning.final_answer is not None
    assert result.typed_reasoning.final_answer.normalized == 75


def test_verify_pipeline_degrades_when_typed_reasoning_extraction_fails() -> None:
    """REQ-VERIFY-017: Typed IR extraction failures degrade to None, not crashes."""
    pipeline = VerifyRepairPipeline()

    with patch(
        "carnot.pipeline.verify_repair.build_typed_reasoning_ir",
        side_effect=ValueError("boom"),
    ):
        assert pipeline.extract_typed_reasoning("q", "r") is None
