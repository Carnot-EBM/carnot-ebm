"""Tests for `carnot.pipeline.semantic_grounding`.

Spec: REQ-VERIFY-020, REQ-VERIFY-021,
SCENARIO-VERIFY-020, SCENARIO-VERIFY-021
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import carnot.pipeline.semantic_grounding as semantic_grounding
from carnot.pipeline.semantic_grounding import (
    PromptClause,
    QuestionProfile,
    SemanticClaim,
    SemanticGroundingResult,
    SemanticGroundingVerifier,
    SemanticGroundingViolation,
    verify_semantic_grounding,
)
from carnot.pipeline.typed_reasoning import (
    AtomicClaim,
    ExtractionProvenance,
    FinalAnswer,
    TypedReasoningIR,
)
from carnot.pipeline.verify_repair import VerificationResult, VerifyRepairPipeline

_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "research" / "semantic_failure_corpus_214.jsonl"
)


def _corpus_example(example_id: str) -> dict[str, Any]:
    for line in _CORPUS_PATH.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["example_id"] == example_id:
            return record
    raise AssertionError(f"missing corpus example: {example_id}")


def test_exp214_omitted_premise_emits_structured_missing_coverage() -> None:
    """SCENARIO-VERIFY-020: Missing prompt premises become structured violations."""
    example = _corpus_example("exp214-followup-omitted-premise-1")

    result = verify_semantic_grounding(
        question=example["prompt"],
        response=example["response"],
    )

    assert result.verified is False
    assert any(
        violation.violation_type == "missing_quantity_coverage"
        and "blueberry" in violation.description.lower()
        for violation in result.violations
    )
    assert any(
        violation.metadata["taxonomy_hint"] == "omitted_premises" for violation in result.violations
    )

    constraints = result.to_constraint_results()
    assert len(constraints) == len(result.violations)
    assert all(constraint.constraint_type == "semantic_grounding" for constraint in constraints)


def test_exp214_profit_case_flags_answer_target_mismatch() -> None:
    """REQ-VERIFY-020: Wrong-target numeric answers are flagged as semantic failures."""
    example = _corpus_example("exp214-followup-question-grounding-2")

    result = verify_semantic_grounding(
        question=example["prompt"],
        response=example["response"],
    )

    assert result.verified is False
    assert any(
        violation.violation_type == "answer_target_mismatch" for violation in result.violations
    )
    assert any(
        "profit" in str(violation.metadata["target_keywords"]) for violation in result.violations
    )
    assert any(
        violation.metadata["taxonomy_hint"] == "question_grounding_failures"
        for violation in result.violations
    )


def test_exp214_entity_binding_assumption_flags_unsupported_reference() -> None:
    """REQ-VERIFY-020: Unsupported assumptions are surfaced deterministically."""
    example = _corpus_example("exp214-live-506")

    result = verify_semantic_grounding(
        question=example["prompt"],
        response=example["response"],
    )

    assert result.verified is False
    assert any(
        violation.violation_type == "unsupported_reference" for violation in result.violations
    )
    assert any(
        violation.metadata["taxonomy_hint"] == "entity_quantity_binding_errors"
        for violation in result.violations
    )


def test_optional_refiner_receives_structured_payload_and_can_filter_violations() -> None:
    """REQ-VERIFY-021: Optional refinement operates on structured summaries."""
    example = _corpus_example("exp214-live-506")
    seen: dict[str, object] = {}

    def refiner(payload: dict[str, object], violations: list[object]) -> list[object]:
        seen["payload"] = payload
        return [
            violation
            for violation in violations
            if violation.violation_type != "unsupported_reference"
        ]

    verifier = SemanticGroundingVerifier(refiner=refiner)
    result = verifier.verify(
        question=example["prompt"],
        response=example["response"],
    )

    assert result.refinement_applied is True
    assert isinstance(seen["payload"], dict)
    payload = seen["payload"]
    assert "claims" in payload
    assert "prompt_clauses" in payload
    assert all(
        violation.violation_type != "unsupported_reference" for violation in result.violations
    )


def test_fully_grounded_answer_stays_violation_free() -> None:
    """REQ-VERIFY-021: Conservative checks stay quiet on grounded correct answers."""
    example = _corpus_example("exp214-followup-omitted-premise-1")
    response = (
        "After setting aside 12 muffins, 48 - 12 = 36 remain. "
        "Split across 3 baskets gives 12 muffins per basket. "
        "Half of each basket is blueberry, so 6 blueberry muffins are in each basket. "
        "Answer: 6"
    )

    result = verify_semantic_grounding(
        question=example["prompt"],
        response=response,
    )

    assert result.verified is True
    assert result.violations == []


def test_code_like_prompt_is_skipped_conservatively() -> None:
    """REQ-VERIFY-021: Code prompts are not forced through the word-problem checker."""
    result = verify_semantic_grounding(
        question=(
            "Write `def dedupe_keep_order(items: list[str]) -> list[str]` that "
            "returns the first occurrence of each string in its original order."
        ),
        response="def dedupe_keep_order(items: list[str]) -> list[str]: return sorted(set(items))",
    )

    assert result.verified is True
    assert result.violations == []


def test_pipeline_verify_fails_on_wrong_question_answered_correctly() -> None:
    """SCENARIO-VERIFY-021: Pipeline fails semantic wrong-target answers."""
    example = _corpus_example("exp214-followup-question-grounding-2")

    result = VerifyRepairPipeline().verify(
        question=example["prompt"],
        response=example["response"],
    )

    assert result.verified is False
    assert any(violation.constraint_type == "semantic_grounding" for violation in result.violations)
    assert all(violation.constraint_type != "arithmetic" for violation in result.violations)


def test_pipeline_verify_preserves_grounded_success_path() -> None:
    """REQ-VERIFY-021: Pipeline remains verified for grounded answers."""
    question = (
        "A baker has 48 muffins. She sets 12 aside for display. The rest are "
        "split equally into 3 baskets, and half of each basket is blueberry. "
        "How many blueberry muffins are in each basket?"
    )
    response = (
        "48 - 12 = 36. 36 / 3 = 12. Half of 12 is 6. "
        "So there are 6 blueberry muffins in each basket. Answer: 6"
    )

    result = VerifyRepairPipeline().verify(
        question=question,
        response=response,
    )

    assert result.verified is True
    assert all(violation.constraint_type != "semantic_grounding" for violation in result.violations)


def test_helper_paths_cover_serialization_and_deduplication() -> None:
    """REQ-VERIFY-021: Helper serialization paths remain deterministic."""
    violation = SemanticGroundingViolation(
        violation_type="missing_entity_coverage",
        description="Prompt clause missing",
        claim_id="cl1",
        clause_id="p1",
        metadata={"taxonomy_hint": "omitted_premises"},
    )
    result = SemanticGroundingResult(
        question_profile=QuestionProfile(
            question="How many apples?",
            prompt_clauses=[PromptClause("p1", "There are 3 apples", ["apple"], ["3"], "premise")],
            target_clause=PromptClause("target", "How many apples?", ["apple"], [], "target"),
            target_keywords=["apple"],
            target_cues=[],
        ),
        claims=[SemanticClaim("cl1", "Answer: 3", ["apple"], ["3"], True, "3")],
        violations=[violation],
    )

    assert violation.to_dict()["violation_type"] == "missing_entity_coverage"
    payload = result.to_refinement_payload()
    assert "question" in payload
    assert "violations" in payload
    deduped = semantic_grounding._dedupe_violations([violation, violation])
    assert len(deduped) == 1


def test_helper_branches_cover_empty_inputs_percentages_and_target_fallbacks() -> None:
    """REQ-VERIFY-020: Helper branches cover conservative fallback behavior."""
    profile = semantic_grounding._build_question_profile(
        "One fee, , extra charge, how much total? Start, , finish."
    )
    assert profile.target_clause.text.lower().startswith("how much")
    assert semantic_grounding._target_text([], "fallback") == "fallback"
    assert semantic_grounding._extract_quantities("20% half") == ["20", "0.2", "0.5"]
    assert semantic_grounding._extract_claims("***", None)[0].text == ""
    assert semantic_grounding._deterministic_violations(profile, []) == []
    assert semantic_grounding._claim_covers_clause(
        PromptClause("p1", "blue apples", ["blue"], [], "premise", ["blue"]),
        SemanticClaim("cl1", "blue apples", ["blue"], []),
    )
    assert semantic_grounding._question_is_groundable(profile) is True
    assert (
        semantic_grounding._question_is_groundable(
            QuestionProfile(
                question="How many more are left?",
                prompt_clauses=[],
                target_clause=PromptClause(
                    "target", "How many more are left?", ["left"], [], "target"
                ),
                target_keywords=["left"],
                target_cues=["difference"],
            )
        )
        is True
    )
    assert (
        semantic_grounding._question_is_groundable(
            QuestionProfile(
                question="Write `def foo()`",
                prompt_clauses=[],
                target_clause=PromptClause("target", "Write `def foo()`", [], [], "target"),
                target_keywords=[],
                target_cues=[],
                is_code_like=True,
            )
        )
        is False
    )
    assert (
        semantic_grounding._clause_requires_grounding(
            PromptClause(
                "p2", "half the basket is blueberry", ["blueberry"], [], "premise", ["blueberry"]
            ),
            profile,
        )
        is True
    )
    assert (
        semantic_grounding._clause_requires_grounding(
            PromptClause("p3", "90 minutes", ["minute"], [], "premise", ["minute"]),
            profile,
        )
        is True
    )
    assert (
        semantic_grounding._clause_requires_grounding(
            PromptClause("p4", "afternoon guests", ["afternoon"], [], "premise", ["afternoon"]),
            QuestionProfile(
                question="How many guests checked in during the afternoon?",
                prompt_clauses=[],
                target_clause=PromptClause(
                    "target",
                    "How many guests checked in during the afternoon?",
                    ["afternoon"],
                    [],
                    "target",
                ),
                target_keywords=["afternoon"],
                target_cues=[],
            ),
        )
        is True
    )

    empty_target_profile = QuestionProfile(
        question="q",
        prompt_clauses=[],
        target_clause=PromptClause("target", "q", [], [], "target"),
        target_keywords=[],
        target_cues=[],
    )
    assert (
        semantic_grounding._answer_target_violation(
            empty_target_profile,
            [SemanticClaim("cl1", "Answer: 2", [], ["2"], True, "2")],
            [],
        )
        is None
    )

    no_issue_profile = QuestionProfile(
        question="Assume 2 dogs. How many dogs are there?",
        prompt_clauses=[PromptClause("p1", "Assume 2 dogs.", ["dog"], ["2"], "premise")],
        target_clause=PromptClause("target", "How many dogs are there?", ["dog"], [], "target"),
        target_keywords=["dog"],
        target_cues=[],
    )
    assert (
        semantic_grounding._unsupported_reference_violation(
            no_issue_profile,
            [SemanticClaim("cl1", "Assume 2 dogs remain.", ["dog", "assume"], ["2"])],
            [],
        )
        is None
    )
    assert semantic_grounding._extract_keywords("guppies fishes") == ["guppy", "fish"]


def test_typed_reasoning_and_taxonomy_helpers_cover_remaining_paths() -> None:
    """REQ-VERIFY-020: Typed reasoning and taxonomy helper branches stay stable."""
    typed_ir = TypedReasoningIR(
        question="How many apples are left?",
        user_constraints=[],
        reasoning_steps=[],
        atomic_claims=[
            AtomicClaim("cl1", "statement", "", value=None, step_id=None),
        ],
        final_answer=FinalAnswer("2", 2, "number"),
        provenance=ExtractionProvenance("fallback_text", "plain_text", "20260412"),
    )
    claims = semantic_grounding._extract_claims("FINAL: 2", typed_ir)
    assert [claim.claim_id for claim in claims] == ["final_answer"]

    afternoon_question = (
        "By how many points did the Tigers win in total after they checked in during the afternoon?"
    )
    base_profile = QuestionProfile(
        question=afternoon_question,
        prompt_clauses=[],
        target_clause=PromptClause(
            "target",
            afternoon_question,
            ["point"],
            [],
            "target",
        ),
        target_keywords=["point"],
        target_cues=["difference", "aggregation", "event_specific"],
    )
    clause = PromptClause("p1", "guest hours", ["guest"], [], "premise", ["guest"])
    assert (
        semantic_grounding._taxonomy_hint(base_profile, clause, "answer_target_mismatch")
        == "question_grounding_failures"
    )
    non_nested_profile = QuestionProfile(
        question="How many guests are there now?",
        prompt_clauses=[],
        target_clause=PromptClause(
            "target",
            "How many guests are there now?",
            ["guest"],
            [],
            "target",
        ),
        target_keywords=["guest"],
        target_cues=["final_state"],
    )
    assert (
        semantic_grounding._taxonomy_hint(
            non_nested_profile,
            clause,
            "unsupported_reference",
        )
        == "question_grounding_failures"
    )

    unit_profile = QuestionProfile(
        question="How much total time?",
        prompt_clauses=[],
        target_clause=PromptClause("target", "How much total time?", ["time"], [], "target"),
        target_keywords=["time"],
        target_cues=["aggregation"],
    )
    assert (
        semantic_grounding._taxonomy_hint(
            unit_profile,
            PromptClause("p2", "90 minutes", ["minute"], ["90"], "premise", ["minute"]),
            "missing_quantity_coverage",
        )
        == "unit_aggregation_errors"
    )

    nested_profile = QuestionProfile(
        question="Five horses each need shoes. How much will it cost?",
        prompt_clauses=[],
        target_clause=PromptClause("target", "How much will it cost?", ["cost"], [], "target"),
        target_keywords=["cost"],
        target_cues=[],
    )
    assert (
        semantic_grounding._taxonomy_hint(
            nested_profile,
            PromptClause("p3", "each horse", ["horse"], [], "premise", ["horse"]),
            "missing_quantity_coverage",
        )
        == "entity_quantity_binding_errors"
    )

    target_overlap_profile = QuestionProfile(
        question="How many guests checked in during the afternoon?",
        prompt_clauses=[],
        target_clause=PromptClause(
            "target",
            "How many guests checked in during the afternoon?",
            ["guest", "afternoon"],
            [],
            "target",
        ),
        target_keywords=["afternoon"],
        target_cues=[],
    )
    assert (
        semantic_grounding._taxonomy_hint(
            target_overlap_profile,
            PromptClause(
                "p4",
                "18 guests check in during the afternoon",
                ["guest", "afternoon"],
                ["18"],
                "premise",
                ["afternoon"],
            ),
            "missing_quantity_coverage",
        )
        == "question_grounding_failures"
    )

    default_profile = QuestionProfile(
        question="How much money did Eli spend?",
        prompt_clauses=[],
        target_clause=PromptClause(
            "target", "How much money did Eli spend?", ["money", "spend"], [], "target"
        ),
        target_keywords=["money"],
        target_cues=[],
    )
    assert (
        semantic_grounding._taxonomy_hint(
            default_profile,
            PromptClause(
                "p5",
                "He buys a notebook for 9 dollars",
                ["notebook"],
                ["9"],
                "premise",
                ["notebook"],
            ),
            "missing_quantity_coverage",
        )
        == "omitted_premises"
    )

    cues = semantic_grounding._target_cues(
        "By how many points in total checked in during the afternoon now?"
    )
    assert "difference" in cues
    assert "aggregation" in cues
    assert "event_specific" in cues
    assert "final_state" in cues


def test_pipeline_semantic_grounding_degrades_cleanly_on_verifier_error() -> None:
    """REQ-VERIFY-021: Pipeline degrades if semantic grounding raises unexpectedly."""

    class BrokenVerifier:
        def verify(
            self, question: str, response: str, typed_reasoning: object | None = None
        ) -> object:
            raise RuntimeError("boom")

    pipeline = VerifyRepairPipeline(semantic_grounding_verifier=BrokenVerifier())  # type: ignore[arg-type]
    assert pipeline.verify_semantic_grounding("What is 1 + 1?", "2") is None


def test_merge_semantic_grounding_adds_pipeline_compatible_violations() -> None:
    """REQ-VERIFY-021: Merge helper appends semantic violations to verification results."""
    base = VerificationResult(
        verified=True,
        constraints=[],
        energy=0.0,
        violations=[],
        certificate={"n_constraints": 0, "n_violations": 0},
    )
    semantic = SemanticGroundingResult(
        question_profile=QuestionProfile(
            question="How much profit?",
            prompt_clauses=[],
            target_clause=PromptClause("target", "How much profit?", ["profit"], [], "target"),
            target_keywords=["profit"],
            target_cues=["difference"],
        ),
        claims=[],
        violations=[
            SemanticGroundingViolation(
                violation_type="answer_target_mismatch",
                description="Wrong target",
                metadata={"taxonomy_hint": "question_grounding_failures"},
            )
        ],
    )

    merged = VerifyRepairPipeline._merge_semantic_grounding(base, semantic)

    assert merged.verified is False
    assert len(merged.constraints) == 1
    assert len(merged.violations) == 1
    assert merged.certificate["n_constraints"] == 1
    assert merged.certificate["n_violations"] == 1
