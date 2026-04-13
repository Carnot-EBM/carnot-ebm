"""Tests for `carnot.pipeline.semantic_verifier_v2`.

Spec: REQ-VERIFY-046, REQ-VERIFY-047,
SCENARIO-VERIFY-047, SCENARIO-VERIFY-048, SCENARIO-VERIFY-049
"""

from __future__ import annotations

import json
from unittest.mock import patch

import carnot.pipeline.semantic_verifier_v2 as semantic_verifier_v2
from carnot.pipeline.semantic_grounding import (
    PromptClause,
    QuestionProfile,
    SemanticClaim,
    SemanticGroundingResult,
    SemanticGroundingViolation,
)
from carnot.pipeline.semantic_verifier_v2 import SemanticVerifierV2
from carnot.pipeline.verify_repair import VerifyRepairPipeline

_QUESTION = (
    "A baker has 48 muffins. She sets 12 aside for display. The rest are split equally "
    "into 3 baskets, and half of each basket is blueberry. How many blueberry muffins "
    "are in each basket?"
)
_WRONG_TARGET_RESPONSE = json.dumps(
    {
        "steps": [
            "48 - 12 = 36",
            "36 / 3 = 12",
            "Each basket has 12 muffins.",
        ],
        "claims": [
            "Each basket has 12 muffins.",
        ],
        "answer": "12",
    }
)
_SUPPORTED_RESPONSE = json.dumps(
    {
        "steps": [
            "48 - 12 = 36",
            "36 / 3 = 12",
            "Half of each basket is blueberry, so each basket has 6 blueberry muffins.",
        ],
        "claims": [
            "Each basket has 12 muffins.",
            "Each basket has 6 blueberry muffins.",
        ],
        "answer": "6",
    }
)


def test_thresholds_load_from_exp232_and_exp233_artifacts() -> None:
    """REQ-VERIFY-046: Thresholds are calibrated from the checked-in corpus and policy."""
    verifier = SemanticVerifierV2()

    assert verifier.thresholds.source_run_date == "20260413"
    assert verifier.thresholds.calibration_rows >= 560
    assert verifier.thresholds.support_max_error_probability > 0.0
    assert (
        verifier.thresholds.support_max_error_probability
        < verifier.thresholds.violation_min_error_probability
        < 1.0
    )
    assert verifier.thresholds.min_monitorability_confidence > 0.0


def test_claim_isolated_violation_gets_calibrated_confidence_and_constraints() -> None:
    """SCENARIO-VERIFY-047: Strong wrong-target evidence becomes a calibrated violation."""
    verifier = SemanticVerifierV2()

    result = verifier.verify(
        question=_QUESTION,
        response=_WRONG_TARGET_RESPONSE,
        task_slice="live_gsm8k_semantic_failure",
    )

    assert result.response_mode == "direct_json"
    assert result.recommended_response_mode == "grammar_gated_json"
    assert result.verdict == "violated"
    assert result.focus_claim_id is not None
    assert result.semantic_error_probability >= result.thresholds.violation_min_error_probability
    assert any(claim.status == "violated" for claim in result.claim_results)

    constraints = result.to_constraint_results()
    assert constraints
    assert all(constraint.constraint_type == "semantic_verifier_v2" for constraint in constraints)


def test_weak_semantic_evidence_abstains_and_emits_no_constraints() -> None:
    """SCENARIO-VERIFY-048: Bare-answer weak evidence becomes abstain, not a forced failure."""
    verifier = SemanticVerifierV2()

    result = verifier.verify(
        question=_QUESTION,
        response="6",
        task_slice="live_gsm8k_semantic_failure",
    )

    assert result.verdict == "abstain"
    assert result.monitorability_confidence < result.thresholds.min_monitorability_confidence
    assert result.semantic_error_probability < result.thresholds.violation_min_error_probability
    assert result.to_constraint_results() == []


def test_supported_result_serializes_deterministically() -> None:
    """SCENARIO-VERIFY-049: Repeated runs serialize identically for fixed inputs."""
    verifier = SemanticVerifierV2()

    first = verifier.verify(
        question=_QUESTION,
        response=_SUPPORTED_RESPONSE,
        task_slice="live_gsm8k_semantic_failure",
    )
    second = verifier.verify(
        question=_QUESTION,
        response=_SUPPORTED_RESPONSE,
        task_slice="live_gsm8k_semantic_failure",
    )

    assert first.verdict == "supported"
    assert first.run_date == "20260413"
    assert first.to_dict() == second.to_dict()
    assert first.to_json() == second.to_json()


def test_pipeline_exposes_semantic_verifier_v2_entry_point_and_result() -> None:
    """REQ-VERIFY-047: VerifyRepairPipeline wires semantic verifier v2 additively."""
    pipeline = VerifyRepairPipeline()

    typed_reasoning = pipeline.extract_typed_reasoning(_QUESTION, _WRONG_TARGET_RESPONSE)
    semantic_grounding = pipeline.verify_semantic_grounding(
        _QUESTION,
        _WRONG_TARGET_RESPONSE,
        typed_reasoning,
    )
    direct = pipeline.verify_semantic_verifier_v2(
        _QUESTION,
        _WRONG_TARGET_RESPONSE,
        typed_reasoning=typed_reasoning,
        semantic_grounding=semantic_grounding,
        task_slice="live_gsm8k_semantic_failure",
    )
    result = pipeline.verify(_QUESTION, _WRONG_TARGET_RESPONSE)

    assert direct is not None
    assert direct.verdict == "violated"
    assert result.semantic_verifier_v2 is not None
    assert result.semantic_verifier_v2.to_json() == direct.to_json()
    assert result.verified is False
    assert any(
        violation.constraint_type in {"semantic_grounding", "semantic_verifier_v2"}
        for violation in result.violations
    )


def test_pipeline_abstain_keeps_legacy_detail_but_avoids_false_positive() -> None:
    """REQ-VERIFY-047: Abstain suppresses automatic semantic false positives in verify()."""
    result = VerifyRepairPipeline().verify(_QUESTION, "6")

    assert result.semantic_grounding is not None
    assert result.semantic_verifier_v2 is not None
    assert result.semantic_verifier_v2.verdict == "abstain"
    assert result.verified is True
    assert all(
        violation.constraint_type != "semantic_verifier_v2" for violation in result.violations
    )
    assert all(violation.constraint_type != "semantic_grounding" for violation in result.violations)


def test_helper_paths_cover_wrapper_threshold_fallbacks_and_focus_constraint_fallback(
    tmp_path,
) -> None:
    """REQ-VERIFY-046: Helper fallbacks remain deterministic on missing or sparse inputs."""
    thresholds = semantic_verifier_v2._load_thresholds(
        calibration_path=tmp_path / "missing.jsonl",
        min_monitorability_confidence=0.4,
    )
    assert thresholds.calibration_rows == 0
    assert thresholds.min_monitorability_confidence == 0.4
    assert semantic_verifier_v2._ratio(1, 0) == 1.0
    assert semantic_verifier_v2._response_mode(None, "text") == "fallback_text"
    assert semantic_verifier_v2._response_mode(None, "") == "empty"
    assert semantic_verifier_v2._percentile([], 0.5) == 0.0
    assert (
        semantic_verifier_v2._target_coverage(
            QuestionProfile(
                question="q",
                prompt_clauses=[],
                target_clause=PromptClause("target", "q", [], [], "target"),
                target_keywords=[],
                target_cues=[],
            ),
            SemanticClaim("cl0", "anything", [], []),
        )
        == 1.0
    )
    assert (
        semantic_verifier_v2._supporting_claim_ids(
            claim=SemanticClaim("cl0", "covered", ["blueberry"], []),
            claims=[SemanticClaim("cl0", "covered", ["blueberry"], [])],
            clause_matches={"cl0": ["p1"]},
            required_clauses=[
                PromptClause("p1", "covered", ["blueberry"], [], "premise"),
            ],
        )
        == []
    )
    assert (
        semantic_verifier_v2._raw_error_probability(
            claim=SemanticClaim("cl1", "Assume the baskets were blueberry.", ["assume"], []),
            focus_claim=SemanticClaim("cl1", "Assume the baskets were blueberry.", ["assume"], []),
            target_coverage=0.0,
            premise_support=0.0,
            legacy_violation_types=["unsupported_reference"],
            response_mode="fallback_text",
            recommended_response_mode="grammar_gated_json",
            question_profile=QuestionProfile(
                question=_QUESTION,
                prompt_clauses=[],
                target_clause=PromptClause(
                    "target",
                    "How many blueberry muffins are in each basket?",
                    ["blueberry", "muffin"],
                    [],
                    "target",
                ),
                target_keywords=["blueberry", "muffin"],
                target_cues=[],
            ),
        )
        >= 0.25
    )
    assert semantic_verifier_v2.verify_semantic_verifier_v2(_QUESTION, "").verdict == "abstain"

    manual = semantic_verifier_v2.SemanticVerifierV2Result(
        question_profile=QuestionProfile(
            question=_QUESTION,
            prompt_clauses=[],
            target_clause=PromptClause(
                "target",
                "How many blueberry muffins are in each basket?",
                ["blueberry", "muffin"],
                [],
                "target",
            ),
            target_keywords=["blueberry", "muffin"],
            target_cues=[],
        ),
        claims=[SemanticClaim("cl1", "Answer: 12", ["muffin"], ["12"], True, "12")],
        claim_results=[
            semantic_verifier_v2.SemanticClaimResult(
                claim_id="cl1",
                text="Answer: 12",
                is_final=True,
                status="abstain",
                answer_target_coverage=0.5,
                premise_support=0.0,
                monitorability_confidence=0.5,
                semantic_error_probability=0.6,
            )
        ],
        focus_claim_id="cl1",
        verdict="violated",
        semantic_error_probability=0.6,
        monitorability_confidence=0.5,
        thresholds=semantic_verifier_v2.SemanticVerifierV2Thresholds(
            support_max_error_probability=0.125,
            violation_min_error_probability=0.45,
            min_monitorability_confidence=0.35,
            source_run_date="20260413",
            calibration_rows=0,
        ),
        response_mode="fallback_text",
        recommended_response_mode=None,
    )
    assert len(manual.to_constraint_results()) == 1


def test_degrade_and_merge_helper_cover_remaining_pipeline_paths() -> None:
    """SCENARIO-VERIFY-049: Degrade and merge helper branches stay pipeline-compatible."""

    class BrokenVerifier:
        def verify(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

    question_profile = QuestionProfile(
        question=_QUESTION,
        prompt_clauses=[],
        target_clause=PromptClause(
            "target",
            "How many blueberry muffins are in each basket?",
            ["blueberry", "muffin"],
            [],
            "target",
        ),
        target_keywords=["blueberry", "muffin"],
        target_cues=[],
    )
    semantic_grounding = SemanticGroundingResult(
        question_profile=question_profile,
        claims=[],
        violations=[
            SemanticGroundingViolation(
                violation_type="answer_target_mismatch",
                description="wrong target",
                claim_id="cl1",
            )
        ],
    )

    broken_pipeline = VerifyRepairPipeline(semantic_verifier_v2=BrokenVerifier())  # type: ignore[arg-type]
    assert broken_pipeline.verify_semantic_verifier_v2(_QUESTION, _WRONG_TARGET_RESPONSE) is None

    degraded = broken_pipeline.verify(_QUESTION, _WRONG_TARGET_RESPONSE)
    assert degraded.verified is False
    assert any(
        violation.constraint_type == "semantic_grounding" for violation in degraded.violations
    )

    base = broken_pipeline.verify("What is 1 + 1?", "1 + 1 = 2.", domain="arithmetic")
    semantic_v2 = semantic_verifier_v2.SemanticVerifierV2Result(
        question_profile=question_profile,
        claims=[SemanticClaim("cl1", "Each basket has 12 muffins.", ["basket"], ["12"])],
        claim_results=[
            semantic_verifier_v2.SemanticClaimResult(
                claim_id="cl1",
                text="Each basket has 12 muffins.",
                is_final=True,
                status="violated",
                answer_target_coverage=0.5,
                premise_support=0.25,
                monitorability_confidence=0.8,
                semantic_error_probability=0.7,
            )
        ],
        focus_claim_id="cl1",
        verdict="violated",
        semantic_error_probability=0.7,
        monitorability_confidence=0.8,
        thresholds=semantic_verifier_v2.SemanticVerifierV2Thresholds(
            support_max_error_probability=0.125,
            violation_min_error_probability=0.45,
            min_monitorability_confidence=0.35,
            source_run_date="20260413",
            calibration_rows=568,
        ),
        response_mode="direct_json",
        recommended_response_mode="grammar_gated_json",
    )
    merged = VerifyRepairPipeline._merge_semantic_analysis(base, semantic_grounding, semantic_v2)
    assert any(
        violation.constraint_type == "semantic_verifier_v2" for violation in merged.violations
    )
    assert merged.certificate["semantic_verifier_v2"]["verdict"] == "violated"


def test_typed_reasoning_fallback_path_is_exercised_when_direct_parse_fails() -> None:
    """REQ-VERIFY-046: Typed-reasoning extraction failures degrade to fallback text."""
    with patch(
        "carnot.pipeline.semantic_verifier_v2.build_typed_reasoning_ir",
        side_effect=ValueError("boom"),
    ):
        result = SemanticVerifierV2().verify(_QUESTION, "bare answer")

    assert result.response_mode == "fallback_text"
