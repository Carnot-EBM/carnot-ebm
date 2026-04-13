"""Tests for process_verifier.py — process-integrity defect detection.

Spec: REQ-VERIFY-061, REQ-VERIFY-062
SCENARIO-VERIFY-065 (clean trace → no defects)
SCENARIO-VERIFY-066 (unsupported claim → defect)
SCENARIO-VERIFY-067 (right_answer_wrong_process → defect)
SCENARIO-VERIFY-068 (repair regression → defect)
SCENARIO-VERIFY-069 (deterministic serialization)
"""

from __future__ import annotations

import json

import pytest

from carnot.pipeline.process_verifier import (
    ALL_DEFECT_KINDS,
    CONTRADICTORY_INTERMEDIATE,
    MISSING_PREMISE_JUMP,
    OUTCOME_CORRECT_PROCESS_INVALID,
    REPAIR_REGRESSION,
    REPAIR_STALL,
    UNSUPPORTED_STEP,
    ProcessDefect,
    ProcessVerificationResult,
    ProcessVerifier,
    RUN_DATE,
    verify_process_integrity,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clean_row(
    *,
    outcome_label: str = "correct",
    process_label: str = "clean",
    n_unsupported: int = 0,
    max_premise_support: float = 1.0,
    verifier_verdict: str = "supported",
    repair_context: dict | None = None,
) -> dict:
    """Build a minimal corpus row that exercises the evidence path."""
    return {
        "outcome_label": outcome_label,
        "process_label": process_label,
        "process_evidence": {
            "n_unsupported_claims": n_unsupported,
            "n_sound_claims": 3,
            "n_total_non_final_claims": 3 + n_unsupported,
            "max_premise_support": max_premise_support,
            "semantic_error_probability": 0.0,
            "verifier_verdict": verifier_verdict,
        },
        "repair_context": repair_context,
    }


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-065: Clean trace → no defects
# ---------------------------------------------------------------------------


class TestCleanTrace:
    """REQ-VERIFY-061 / SCENARIO-VERIFY-065."""

    def test_clean_correct_produces_no_defects(self):
        row = _clean_row(outcome_label="correct", process_label="clean")
        result = ProcessVerifier().verify_reasoning_trace(row)
        assert result.process_valid is True
        assert result.defects == []
        assert result.outcome_correct is True
        assert result.process_label == "clean"
        assert result.run_date == RUN_DATE

    def test_clean_incorrect_produces_no_defects(self):
        row = _clean_row(outcome_label="incorrect", process_label="clean")
        result = ProcessVerifier().verify_reasoning_trace(row)
        assert result.process_valid is True
        assert result.defects == []
        assert result.outcome_correct is False

    def test_partially_sound_abstain_no_regression(self):
        # wrong_answer_partially_sound_process with verifier_verdict=abstain
        # should not raise contradictory_intermediate or regression.
        row = _clean_row(
            outcome_label="incorrect",
            process_label="wrong_answer_partially_sound_process",
            n_unsupported=0,
            max_premise_support=1.0,
            verifier_verdict="abstain",
        )
        result = ProcessVerifier().verify_reasoning_trace(row)
        kinds = {d.kind for d in result.defects}
        assert CONTRADICTORY_INTERMEDIATE not in kinds
        assert REPAIR_REGRESSION not in kinds


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-066: Unsupported claim → defect
# ---------------------------------------------------------------------------


class TestUnsupportedClaim:
    """REQ-VERIFY-061 / SCENARIO-VERIFY-066."""

    def test_one_unsupported_claim_raises_defect(self):
        row = _clean_row(n_unsupported=1, max_premise_support=1.0)
        result = ProcessVerifier().verify_reasoning_trace(row)
        assert result.process_valid is False
        kinds = {d.kind for d in result.defects}
        assert UNSUPPORTED_STEP in kinds

    def test_unsupported_with_low_support_adds_missing_premise(self):
        row = _clean_row(n_unsupported=2, max_premise_support=0.5)
        result = ProcessVerifier().verify_reasoning_trace(row)
        kinds = {d.kind for d in result.defects}
        assert UNSUPPORTED_STEP in kinds
        assert MISSING_PREMISE_JUMP in kinds

    def test_zero_unsupported_no_unsupported_defect(self):
        row = _clean_row(n_unsupported=0, max_premise_support=0.9)
        result = ProcessVerifier().verify_reasoning_trace(row)
        kinds = {d.kind for d in result.defects}
        # Low max_support alone without unsupported claims should NOT trigger
        # missing_premise_jump because the logic requires both conditions.
        assert UNSUPPORTED_STEP not in kinds
        assert MISSING_PREMISE_JUMP not in kinds

    def test_contradictory_verdict_raises_defect(self):
        row = _clean_row(verifier_verdict="violated")
        result = ProcessVerifier().verify_reasoning_trace(row)
        kinds = {d.kind for d in result.defects}
        assert CONTRADICTORY_INTERMEDIATE in kinds
        assert result.process_valid is False


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-067: right_answer_wrong_process → defect
# ---------------------------------------------------------------------------


class TestOutcomeCorrectProcessInvalid:
    """REQ-VERIFY-061 / SCENARIO-VERIFY-067."""

    def test_right_answer_wrong_process_is_flagged(self):
        row = _clean_row(
            outcome_label="correct",
            process_label="right_answer_wrong_process",
        )
        result = ProcessVerifier().verify_reasoning_trace(row)
        kinds = {d.kind for d in result.defects}
        assert OUTCOME_CORRECT_PROCESS_INVALID in kinds
        assert result.process_valid is False

    def test_incorrect_outcome_wrong_process_not_flagged_as_oc_pi(self):
        # The "outcome_correct_process_invalid" defect requires outcome==correct.
        row = _clean_row(
            outcome_label="incorrect",
            process_label="right_answer_wrong_process",
        )
        result = ProcessVerifier().verify_reasoning_trace(row)
        kinds = {d.kind for d in result.defects}
        assert OUTCOME_CORRECT_PROCESS_INVALID not in kinds

    def test_corpus_row_from_exp248(self):
        # Mirrors the structure of an actual row in process_integrity_corpus_248.jsonl
        row = {
            "benchmark": "gsm8k_semantic",
            "case_id": "gsm8k-1009",
            "corpus_id": "pi248-235-gsm8k_semantic-gemma4_e4b_it-gsm8k-1009-it0",
            "domain": "reasoning",
            "experiment": 248,
            "final_answer": {"answer_type": "number", "normalized": 1500, "text": "1500"},
            "iteration": 0,
            "model": "Gemma4-E4B-it",
            "outcome_label": "correct",
            "process_evidence": {
                "max_premise_support": 1.0,
                "n_sound_claims": 6,
                "n_total_non_final_claims": 8,
                "n_unsupported_claims": 1,
                "semantic_error_probability": 0.2,
                "verifier_verdict": "abstain",
            },
            "process_label": "right_answer_wrong_process",
            "repair_context": None,
            "run_date": "20260413",
            "source_artifact": "results/experiment_235_results.json",
            "source_experiment": 235,
            "steps": [],
        }
        result = verify_process_integrity(row)
        kinds = {d.kind for d in result.defects}
        assert UNSUPPORTED_STEP in kinds
        assert OUTCOME_CORRECT_PROCESS_INVALID in kinds
        assert result.outcome_correct is True
        assert result.process_valid is False


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-068: Repair regression → defect
# ---------------------------------------------------------------------------


class TestRepairTraces:
    """REQ-VERIFY-061 / SCENARIO-VERIFY-068."""

    def test_regression_prior_correct_current_incorrect(self):
        row = _clean_row(
            outcome_label="incorrect",
            process_label="wrong_answer_partially_sound_process",
            repair_context={"prior_outcome": "correct"},
        )
        result = ProcessVerifier().verify_code_repair_trace(row)
        kinds = {d.kind for d in result.defects}
        assert REPAIR_REGRESSION in kinds
        assert result.process_valid is False

    def test_stall_prior_incorrect_current_incorrect(self):
        row = _clean_row(
            outcome_label="incorrect",
            process_label="wrong_answer_partially_sound_process",
            repair_context={"prior_outcome": "incorrect"},
        )
        result = ProcessVerifier().verify_code_repair_trace(row)
        kinds = {d.kind for d in result.defects}
        assert REPAIR_STALL in kinds

    def test_successful_repair_no_regression_or_stall(self):
        row = _clean_row(
            outcome_label="correct",
            process_label="clean",
            repair_context={"prior_outcome": "incorrect"},
        )
        result = ProcessVerifier().verify_code_repair_trace(row)
        kinds = {d.kind for d in result.defects}
        assert REPAIR_REGRESSION not in kinds
        assert REPAIR_STALL not in kinds

    def test_no_repair_context_no_repair_defects(self):
        row = _clean_row(outcome_label="incorrect", repair_context=None)
        result = ProcessVerifier().verify_code_repair_trace(row)
        kinds = {d.kind for d in result.defects}
        assert REPAIR_REGRESSION not in kinds
        assert REPAIR_STALL not in kinds

    def test_corpus_regression_row_from_exp248(self):
        # Matches gsm8k-1034 it1 in the corpus (prior_outcome='correct' → incorrect).
        row = {
            "outcome_label": "incorrect",
            "process_label": "wrong_answer_partially_sound_process",
            "process_evidence": {
                "max_premise_support": 0.5,
                "n_sound_claims": 2,
                "n_total_non_final_claims": 3,
                "n_unsupported_claims": 1,
                "semantic_error_probability": 0.672,
                "verifier_verdict": "violated",
            },
            "repair_context": {"prior_outcome": "correct"},
        }
        result = verify_process_integrity(row)
        kinds = {d.kind for d in result.defects}
        assert REPAIR_REGRESSION in kinds
        assert UNSUPPORTED_STEP in kinds
        assert CONTRADICTORY_INTERMEDIATE in kinds


# ---------------------------------------------------------------------------
# TypedReasoningIR integration
# ---------------------------------------------------------------------------


class TestTypedReasoningIR:
    """REQ-VERIFY-061 — IR-based defect detection without pre-computed evidence."""

    def _make_ir(self, *, grounded: bool = True):
        from carnot.pipeline.typed_reasoning import (
            AtomicClaim,
            ExtractionProvenance,
            FinalAnswer,
            ReasoningStep,
            TypedReasoningIR,
            UserConstraint,
        )

        step = ReasoningStep(step_id="s1", kind="arithmetic", text="2 + 2 = 4")
        step_id_ref = "s1" if grounded else None
        claim = AtomicClaim(
            claim_id="c1", kind="arithmetic", text="sum is 4", step_id=step_id_ref
        )
        answer = FinalAnswer(
            text="4",
            normalized=4,
            answer_type="number",
            source_step_id="s1",
        )
        constraint = UserConstraint(
            constraint_id="uc1", kind="numeric", text="add two numbers"
        )
        provenance = ExtractionProvenance(
            extraction_method="fallback_text",
            source_format="text",
            parser_version="20260412",
        )
        return TypedReasoningIR(
            question="What is 2+2?",
            user_constraints=[constraint],
            reasoning_steps=[step],
            atomic_claims=[claim],
            final_answer=answer,
            provenance=provenance,
        )

    def test_grounded_ir_no_defects(self):
        ir = self._make_ir(grounded=True)
        result = ProcessVerifier().verify_typed_reasoning(ir, outcome_correct=True)
        assert result.process_valid is True
        assert result.defects == []

    def test_ungrounded_claim_raises_defects(self):
        ir = self._make_ir(grounded=False)
        result = ProcessVerifier().verify_typed_reasoning(ir, outcome_correct=True)
        kinds = {d.kind for d in result.defects}
        assert UNSUPPORTED_STEP in kinds
        assert MISSING_PREMISE_JUMP in kinds
        assert result.process_valid is False

    def test_ir_with_process_evidence_combines_checks(self):
        ir = self._make_ir(grounded=True)
        # Inject external evidence that says verifier violated.
        evidence = {
            "n_unsupported_claims": 0,
            "max_premise_support": 1.0,
            "verifier_verdict": "violated",
        }
        result = ProcessVerifier().verify_typed_reasoning(
            ir, outcome_correct=True, process_evidence=evidence
        )
        kinds = {d.kind for d in result.defects}
        assert CONTRADICTORY_INTERMEDIATE in kinds


# ---------------------------------------------------------------------------
# SCENARIO-VERIFY-069: Deterministic serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    """REQ-VERIFY-061 / SCENARIO-VERIFY-069."""

    def _make_result(self) -> ProcessVerificationResult:
        row = _clean_row(
            n_unsupported=1,
            max_premise_support=0.5,
            verifier_verdict="violated",
            outcome_label="correct",
            process_label="right_answer_wrong_process",
        )
        return ProcessVerifier().verify_reasoning_trace(row)

    def test_to_dict_is_deterministic(self):
        result = self._make_result()
        d1 = result.to_dict()
        d2 = result.to_dict()
        assert d1 == d2

    def test_to_json_is_deterministic(self):
        result = self._make_result()
        j1 = result.to_json()
        j2 = result.to_json()
        assert j1 == j2

    def test_to_json_is_valid_json(self):
        result = self._make_result()
        parsed = json.loads(result.to_json())
        assert "defects" in parsed
        assert "process_valid" in parsed
        assert "run_date" in parsed
        assert parsed["run_date"] == RUN_DATE

    def test_defect_to_dict_has_sorted_keys(self):
        defect = ProcessDefect(
            kind=UNSUPPORTED_STEP,
            detail="test",
            step_id="s1",
            evidence={"z": 1, "a": 2},
        )
        d = defect.to_dict()
        # evidence keys must be sorted
        assert list(d["evidence"].keys()) == sorted(d["evidence"].keys())

    def test_run_date_constant(self):
        assert RUN_DATE == "20260413"

    def test_all_defect_kinds_are_strings(self):
        for kind in ALL_DEFECT_KINDS:
            assert isinstance(kind, str)


# ---------------------------------------------------------------------------
# Pipeline integration (REQ-VERIFY-062)
# ---------------------------------------------------------------------------


class TestPipelineIntegration:
    """REQ-VERIFY-062 — VerifyRepairPipeline.verify_process_integrity."""

    def test_pipeline_exposes_verify_process_integrity(self):
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        pipeline = VerifyRepairPipeline()
        assert hasattr(pipeline, "verify_process_integrity")

    def test_pipeline_returns_process_verification_result(self):
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        pipeline = VerifyRepairPipeline()
        row = _clean_row(outcome_label="correct", process_label="clean")
        result = pipeline.verify_process_integrity(row)
        assert isinstance(result, ProcessVerificationResult)
        assert result.process_valid is True

    def test_pipeline_detect_regression_via_entry_point(self):
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        pipeline = VerifyRepairPipeline()
        row = _clean_row(
            outcome_label="incorrect",
            repair_context={"prior_outcome": "correct"},
        )
        result = pipeline.verify_process_integrity(row)
        kinds = {d.kind for d in result.defects}
        assert REPAIR_REGRESSION in kinds

    def test_pipeline_verify_still_backward_compatible(self):
        """Adding verify_process_integrity must not affect existing verify()."""
        from carnot.pipeline.verify_repair import VerifyRepairPipeline

        pipeline = VerifyRepairPipeline()
        # verify() should still work without process_integrity argument.
        result = pipeline.verify(
            question="What is 2+2?",
            response="The answer is 4.",
        )
        # The standard VerificationResult has no process_integrity by default.
        assert hasattr(result, "verified")
        assert hasattr(result, "constraints")

    def test_convenience_helper_selects_repair_path_on_context(self):
        row_with_context = _clean_row(repair_context={"prior_outcome": "incorrect"})
        result_repair = verify_process_integrity(row_with_context)

        row_no_context = _clean_row(repair_context=None)
        result_plain = verify_process_integrity(row_no_context)

        # Both should return ProcessVerificationResult (no crash).
        assert isinstance(result_repair, ProcessVerificationResult)
        assert isinstance(result_plain, ProcessVerificationResult)
