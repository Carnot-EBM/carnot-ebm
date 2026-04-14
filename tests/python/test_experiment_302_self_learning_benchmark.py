"""Tests for Exp 302 — Integrated self-learning benchmark (Tier 1+2 live).

Covers:
- Artifact schema: batch1_accuracy, batch2_accuracy, improvement_delta, n_new_constraints
- generated_constraint_log: list of (pattern_type, constraint_id, confidence)
- constraint_count_before and constraint_count_after fields present
- Batch split: exactly 50 questions per batch
- n_new_constraints >= 0 (0 is valid when no high-precision patterns found)
- improvement_delta can be negative (honest reporting required)
- memory_patterns_found: count of patterns above min_precision=0.85
- inference_mode: "live_gpu" or "simulated" (explicit label)
- per_question records have correct, violation_detected, confidence_class, repaired fields
- batch2 uses enriched constraint set (constraint_count_after >= constraint_count_before)
- Simulated fallback: runs without GPU, labels inference_mode="simulated"

Spec: REQ-LEARN-010, REQ-LEARN-011, REQ-VERIFY-081, REQ-VERIFY-082,
      SCENARIO-LEARN-015, SCENARIO-LEARN-016, SCENARIO-LEARN-017, SCENARIO-LEARN-018,
      SCENARIO-VERIFY-105, SCENARIO-VERIFY-106, SCENARIO-VERIFY-107, SCENARIO-VERIFY-108
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from scripts.experiment_302_self_learning_benchmark import (
    BATCH_SIZE,
    CONFIDENCE_THRESHOLD,
    EXPERIMENT,
    MIN_PRECISION,
    BatchResult,
    ConstraintGenerationSummary,
    PerQuestionRecord,
    build_artifact,
    compute_improvement_delta,
    count_dynamic_constraints,
    run_batch,
    run_constraint_generation,
    simulate_gsm8k_questions,
)


# ---------------------------------------------------------------------------
# Constants
# REQ-LEARN-010
# ---------------------------------------------------------------------------


class TestConstants:
    """Exp 302 constants must match benchmark design."""

    def test_experiment_id(self) -> None:
        """EXPERIMENT constant is 302."""
        assert EXPERIMENT == 302

    def test_batch_size(self) -> None:
        """BATCH_SIZE is exactly 50 per design."""
        assert BATCH_SIZE == 50

    def test_confidence_threshold(self) -> None:
        """CONFIDENCE_THRESHOLD is 0.8 per Exp 301 design."""
        assert CONFIDENCE_THRESHOLD == 0.8

    def test_min_precision(self) -> None:
        """MIN_PRECISION is 0.85 per arXiv 2603.03538 soundness bound."""
        assert MIN_PRECISION == 0.85


# ---------------------------------------------------------------------------
# PerQuestionRecord schema
# REQ-VERIFY-081
# ---------------------------------------------------------------------------


class TestPerQuestionRecord:
    """PerQuestionRecord must carry all required per-question fields."""

    def _make_record(self, **kwargs: Any) -> PerQuestionRecord:
        defaults: dict[str, Any] = {
            "question_id": "q_0",
            "question": "What is 2+2?",
            "correct": True,
            "violation_detected": False,
            "confidence_class": "LOW",
            "repaired": False,
            "repair_triggered": False,
        }
        defaults.update(kwargs)
        return PerQuestionRecord(**defaults)

    def test_correct_field(self) -> None:
        """REQ-VERIFY-081: correct field is bool."""
        r = self._make_record(correct=True)
        assert r.correct is True

    def test_violation_detected_field(self) -> None:
        """REQ-VERIFY-082: violation_detected is bool."""
        r = self._make_record(violation_detected=True)
        assert r.violation_detected is True

    def test_confidence_class_high(self) -> None:
        """SCENARIO-VERIFY-105: confidence_class is HIGH/MEDIUM/LOW string."""
        r = self._make_record(confidence_class="HIGH")
        assert r.confidence_class == "HIGH"

    def test_confidence_class_medium(self) -> None:
        """SCENARIO-VERIFY-106: MEDIUM is valid confidence_class."""
        r = self._make_record(confidence_class="MEDIUM")
        assert r.confidence_class == "MEDIUM"

    def test_confidence_class_low(self) -> None:
        """SCENARIO-VERIFY-107: LOW is valid confidence_class."""
        r = self._make_record(confidence_class="LOW")
        assert r.confidence_class == "LOW"

    def test_repaired_field(self) -> None:
        """REQ-VERIFY-082: repaired is bool."""
        r = self._make_record(repaired=True)
        assert r.repaired is True

    def test_repair_triggered_implies_violation(self) -> None:
        """SCENARIO-VERIFY-108: repair_triggered=True only when violation_detected=True."""
        r = self._make_record(violation_detected=True, repair_triggered=True)
        assert r.repair_triggered is True
        assert r.violation_detected is True

    def test_repair_triggered_without_violation_raises(self) -> None:
        """Invariant enforcement: repair_triggered=True + violation_detected=False raises ValueError."""
        with pytest.raises(ValueError, match="repair_triggered"):
            self._make_record(violation_detected=False, repair_triggered=True)

    def test_to_dict_has_all_fields(self) -> None:
        """PerQuestionRecord.to_dict() emits all required schema fields."""
        r = self._make_record()
        d = r.to_dict()
        assert "question_id" in d
        assert "correct" in d
        assert "violation_detected" in d
        assert "confidence_class" in d
        assert "repaired" in d
        assert "repair_triggered" in d


# ---------------------------------------------------------------------------
# BatchResult schema
# REQ-LEARN-010
# ---------------------------------------------------------------------------


class TestBatchResult:
    """BatchResult must aggregate per-question records correctly."""

    def _make_records(self, n: int, correct_count: int) -> list[PerQuestionRecord]:
        records = []
        for i in range(n):
            records.append(
                PerQuestionRecord(
                    question_id=f"q_{i}",
                    question=f"Question {i}",
                    correct=(i < correct_count),
                    violation_detected=False,
                    confidence_class="LOW",
                    repaired=False,
                    repair_triggered=False,
                )
            )
        return records

    def test_accuracy_all_correct(self) -> None:
        """BatchResult.accuracy is 1.0 when all 50 questions correct."""
        records = self._make_records(50, 50)
        result = BatchResult(records=records, batch_index=1)
        assert result.accuracy == 1.0

    def test_accuracy_all_wrong(self) -> None:
        """BatchResult.accuracy is 0.0 when all 50 questions wrong."""
        records = self._make_records(50, 0)
        result = BatchResult(records=records, batch_index=1)
        assert result.accuracy == 0.0

    def test_accuracy_half(self) -> None:
        """BatchResult.accuracy is 0.5 for 25/50 correct."""
        records = self._make_records(50, 25)
        result = BatchResult(records=records, batch_index=1)
        assert result.accuracy == pytest.approx(0.5)

    def test_exactly_50_questions_under(self) -> None:
        """BATCH_SIZE=50 enforced: BatchResult with 49 records raises ValueError."""
        records = self._make_records(49, 25)
        with pytest.raises(ValueError, match="50"):
            BatchResult(records=records, batch_index=1)

    def test_exactly_50_questions_over(self) -> None:
        """BATCH_SIZE=50 enforced: BatchResult with 51 records raises ValueError."""
        records = self._make_records(51, 25)
        with pytest.raises(ValueError, match="50"):
            BatchResult(records=records, batch_index=1)

    def test_to_dict_has_accuracy(self) -> None:
        """BatchResult.to_dict() includes accuracy field."""
        records = self._make_records(50, 30)
        result = BatchResult(records=records, batch_index=1)
        d = result.to_dict()
        assert "accuracy" in d
        assert d["accuracy"] == pytest.approx(30 / 50)

    def test_to_dict_has_per_question(self) -> None:
        """BatchResult.to_dict() includes per_question list of dicts."""
        records = self._make_records(50, 25)
        result = BatchResult(records=records, batch_index=1)
        d = result.to_dict()
        assert "per_question" in d
        assert len(d["per_question"]) == 50


# ---------------------------------------------------------------------------
# ConstraintGenerationSummary schema
# REQ-LEARN-010, REQ-LEARN-011
# ---------------------------------------------------------------------------


class TestConstraintGenerationSummary:
    """ConstraintGenerationSummary captures all constraint-generation audit fields."""

    def _make_summary(self, **kwargs: Any) -> ConstraintGenerationSummary:
        defaults: dict[str, Any] = {
            "constraint_count_before": 3,
            "constraint_count_after": 5,
            "n_new_constraints": 2,
            "memory_patterns_found": 3,
            "generation_log": {"carry_check:carry_error": "added"},
            "generated_constraint_log": [
                {
                    "pattern_type": "carry_check",
                    "constraint_id": "learned:carry_error",
                    "confidence": 0.9,
                }
            ],
        }
        defaults.update(kwargs)
        return ConstraintGenerationSummary(**defaults)

    def test_constraint_count_before_present(self) -> None:
        """REQ-LEARN-010: constraint_count_before captures pre-generation size."""
        s = self._make_summary(constraint_count_before=3)
        assert s.constraint_count_before == 3

    def test_constraint_count_after_gte_before(self) -> None:
        """REQ-LEARN-010: constraint_count_after >= constraint_count_before (additive only)."""
        s = self._make_summary(constraint_count_before=3, constraint_count_after=5)
        assert s.constraint_count_after >= s.constraint_count_before

    def test_n_new_constraints_zero_is_valid(self) -> None:
        """SCENARIO-LEARN-016: n_new_constraints=0 is valid (no high-precision patterns)."""
        s = self._make_summary(
            n_new_constraints=0,
            constraint_count_before=3,
            constraint_count_after=3,
            generated_constraint_log=[],
        )
        assert s.n_new_constraints == 0

    def test_n_new_constraints_non_negative(self) -> None:
        """REQ-LEARN-010: n_new_constraints >= 0 always."""
        s = self._make_summary(n_new_constraints=0)
        assert s.n_new_constraints >= 0

    def test_n_new_constraints_negative_raises(self) -> None:
        """ConstraintGenerationSummary raises ValueError for n_new_constraints < 0."""
        with pytest.raises(ValueError, match="n_new_constraints"):
            self._make_summary(
                n_new_constraints=-1,
                constraint_count_before=3,
                constraint_count_after=3,
                generated_constraint_log=[],
            )

    def test_count_after_less_than_before_raises(self) -> None:
        """Additive-only invariant: constraint_count_after < constraint_count_before raises."""
        with pytest.raises(ValueError, match="additive"):
            ConstraintGenerationSummary(
                constraint_count_before=5,
                constraint_count_after=3,
                n_new_constraints=0,
                memory_patterns_found=0,
                generation_log={},
                generated_constraint_log=[],
            )

    def test_memory_patterns_found_non_negative(self) -> None:
        """SCENARIO-LEARN-015: memory_patterns_found >= 0."""
        s = self._make_summary(memory_patterns_found=0)
        assert s.memory_patterns_found >= 0

    def test_generated_constraint_log_structure(self) -> None:
        """SCENARIO-LEARN-017: generated_constraint_log entries have pattern_type/constraint_id/confidence."""
        s = self._make_summary()
        for entry in s.generated_constraint_log:
            assert "pattern_type" in entry
            assert "constraint_id" in entry
            assert "confidence" in entry

    def test_generation_log_has_outcomes(self) -> None:
        """SCENARIO-LEARN-018: generation_log maps pattern_key to outcome string."""
        s = self._make_summary(
            generation_log={
                "carry_check:carry_error": "added",
                "sign_consistency:sign_error": "rejected_soundness",
                "learned_check:unknown": "already_exists",
            }
        )
        valid_outcomes = {"added", "rejected_soundness", "already_exists"}
        for outcome in s.generation_log.values():
            assert outcome in valid_outcomes

    def test_to_dict_has_all_fields(self) -> None:
        """ConstraintGenerationSummary.to_dict() emits all required schema fields."""
        s = self._make_summary()
        d = s.to_dict()
        required = [
            "constraint_count_before",
            "constraint_count_after",
            "n_new_constraints",
            "memory_patterns_found",
            "generation_log",
            "generated_constraint_log",
        ]
        for field_name in required:
            assert field_name in d, f"Missing field: {field_name}"


# ---------------------------------------------------------------------------
# compute_improvement_delta
# REQ-LEARN-010
# ---------------------------------------------------------------------------


class TestComputeImprovementDelta:
    """compute_improvement_delta must be honest: negative values are valid."""

    def test_positive_delta(self) -> None:
        """Delta is positive when batch2 > batch1."""
        assert compute_improvement_delta(0.6, 0.7) == pytest.approx(0.1)

    def test_negative_delta(self) -> None:
        """Delta can be negative (batch2 regressed vs batch1). Honest reporting required."""
        assert compute_improvement_delta(0.7, 0.6) == pytest.approx(-0.1)

    def test_zero_delta(self) -> None:
        """Delta is 0.0 when batch1 == batch2."""
        assert compute_improvement_delta(0.6, 0.6) == pytest.approx(0.0)

    def test_perfect_improvement(self) -> None:
        """Delta can be +1.0 (all wrong in batch1, all correct in batch2)."""
        assert compute_improvement_delta(0.0, 1.0) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# count_dynamic_constraints
# REQ-LEARN-010
# ---------------------------------------------------------------------------


class TestCountDynamicConstraints:
    """count_dynamic_constraints reads extractor._dynamic_constraints safely."""

    def test_no_attribute(self) -> None:
        """Returns 0 when extractor has no _dynamic_constraints attribute."""
        extractor = object()
        assert count_dynamic_constraints(extractor) == 0

    def test_empty_list(self) -> None:
        """Returns 0 for empty _dynamic_constraints list."""
        extractor = MagicMock()
        extractor._dynamic_constraints = []
        assert count_dynamic_constraints(extractor) == 0

    def test_two_constraints(self) -> None:
        """Returns 2 for list with two learned constraints."""
        extractor = MagicMock()
        extractor._dynamic_constraints = [MagicMock(), MagicMock()]
        assert count_dynamic_constraints(extractor) == 2


# ---------------------------------------------------------------------------
# simulate_gsm8k_questions
# REQ-LEARN-010
# ---------------------------------------------------------------------------


class TestSimulateGsm8kQuestions:
    """simulate_gsm8k_questions produces deterministic synthetic GSM8K-style data."""

    def test_produces_100_questions(self) -> None:
        """simulate_gsm8k_questions produces exactly 100 questions total."""
        questions = simulate_gsm8k_questions(n=100, seed=42)
        assert len(questions) == 100

    def test_produces_50_questions(self) -> None:
        """Can produce exactly 50 questions for a single batch."""
        questions = simulate_gsm8k_questions(n=50, seed=0)
        assert len(questions) == 50

    def test_each_question_has_required_fields(self) -> None:
        """Each question dict has question, answer, correct_answer fields."""
        questions = simulate_gsm8k_questions(n=10, seed=0)
        for q in questions:
            assert "question" in q
            assert "answer" in q
            assert "correct_answer" in q

    def test_deterministic_with_same_seed(self) -> None:
        """Same seed produces identical questions (deterministic)."""
        q1 = simulate_gsm8k_questions(n=10, seed=42)
        q2 = simulate_gsm8k_questions(n=10, seed=42)
        assert q1 == q2

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different questions."""
        q1 = simulate_gsm8k_questions(n=10, seed=42)
        q2 = simulate_gsm8k_questions(n=10, seed=99)
        assert q1 != q2


# ---------------------------------------------------------------------------
# run_batch (simulated mode)
# REQ-VERIFY-081, REQ-VERIFY-082
# ---------------------------------------------------------------------------


class TestRunBatch:
    """run_batch returns a BatchResult with exactly 50 PerQuestionRecord entries."""

    def _make_questions(self, n: int = 50) -> list[dict[str, Any]]:
        return simulate_gsm8k_questions(n=n, seed=0)

    def test_returns_batch_result(self) -> None:
        """run_batch returns a BatchResult."""
        questions = self._make_questions()
        pipeline = MagicMock()
        pipeline.has_model = False
        result = run_batch(questions=questions, pipeline=pipeline, batch_index=1)
        assert isinstance(result, BatchResult)

    def test_exactly_50_records(self) -> None:
        """run_batch produces exactly 50 PerQuestionRecord entries for 50 questions."""
        questions = self._make_questions(50)
        pipeline = MagicMock()
        pipeline.has_model = False
        result = run_batch(questions=questions, pipeline=pipeline, batch_index=1)
        assert len(result.records) == 50

    def test_all_records_have_confidence_class(self) -> None:
        """SCENARIO-VERIFY-105: every PerQuestionRecord has a valid confidence_class."""
        questions = self._make_questions()
        pipeline = MagicMock()
        pipeline.has_model = False
        result = run_batch(questions=questions, pipeline=pipeline, batch_index=1)
        valid_classes = {"HIGH", "MEDIUM", "LOW", "NONE"}
        for rec in result.records:
            assert rec.confidence_class in valid_classes

    def test_repaired_requires_violation(self) -> None:
        """SCENARIO-VERIFY-108: repaired=True only when violation_detected=True."""
        questions = self._make_questions()
        pipeline = MagicMock()
        pipeline.has_model = False
        result = run_batch(questions=questions, pipeline=pipeline, batch_index=1)
        for rec in result.records:
            if rec.repaired:
                assert rec.violation_detected


# ---------------------------------------------------------------------------
# run_constraint_generation
# REQ-LEARN-010, REQ-LEARN-011
# ---------------------------------------------------------------------------


class TestRunConstraintGeneration:
    """run_constraint_generation wraps ConstraintGenerator and returns a summary."""

    def test_returns_constraint_generation_summary(self) -> None:
        """run_constraint_generation returns a ConstraintGenerationSummary."""
        from carnot.pipeline.case_memory import CaseMemory
        from carnot.pipeline.extract import AutoExtractor

        memory = CaseMemory()
        extractor = AutoExtractor()
        summary = run_constraint_generation(memory=memory, extractor=extractor)
        assert isinstance(summary, ConstraintGenerationSummary)

    def test_empty_memory_zero_constraints(self) -> None:
        """SCENARIO-LEARN-016: empty memory → n_new_constraints=0."""
        from carnot.pipeline.case_memory import CaseMemory
        from carnot.pipeline.extract import AutoExtractor

        memory = CaseMemory()
        extractor = AutoExtractor()
        summary = run_constraint_generation(memory=memory, extractor=extractor)
        assert summary.n_new_constraints == 0

    def test_constraint_count_before_gte_zero(self) -> None:
        """constraint_count_before is a non-negative integer."""
        from carnot.pipeline.case_memory import CaseMemory
        from carnot.pipeline.extract import AutoExtractor

        memory = CaseMemory()
        extractor = AutoExtractor()
        summary = run_constraint_generation(memory=memory, extractor=extractor)
        assert summary.constraint_count_before >= 0

    def test_additive_only(self) -> None:
        """REQ-LEARN-010: constraint_count_after >= constraint_count_before."""
        from carnot.pipeline.case_memory import CaseMemory
        from carnot.pipeline.extract import AutoExtractor

        memory = CaseMemory()
        extractor = AutoExtractor()
        summary = run_constraint_generation(memory=memory, extractor=extractor)
        assert summary.constraint_count_after >= summary.constraint_count_before

    def test_memory_patterns_found_non_negative(self) -> None:
        """SCENARIO-LEARN-015: memory_patterns_found >= 0 always."""
        from carnot.pipeline.case_memory import CaseMemory
        from carnot.pipeline.extract import AutoExtractor

        memory = CaseMemory()
        extractor = AutoExtractor()
        summary = run_constraint_generation(memory=memory, extractor=extractor)
        assert summary.memory_patterns_found >= 0


# ---------------------------------------------------------------------------
# build_artifact
# REQ-LEARN-010
# ---------------------------------------------------------------------------


def _make_dummy_batch(batch_index: int, correct: int = 30) -> BatchResult:
    records = [
        PerQuestionRecord(
            question_id=f"q_{i}",
            question=f"Q{i}",
            correct=(i < correct),
            violation_detected=False,
            confidence_class="LOW",
            repaired=False,
            repair_triggered=False,
        )
        for i in range(50)
    ]
    return BatchResult(records=records, batch_index=batch_index)


def _make_dummy_summary(n_new: int = 0) -> ConstraintGenerationSummary:
    return ConstraintGenerationSummary(
        constraint_count_before=3,
        constraint_count_after=3 + n_new,
        n_new_constraints=n_new,
        memory_patterns_found=0,
        generation_log={},
        generated_constraint_log=[],
    )


class TestBuildArtifact:
    """build_artifact produces the final Exp 302 output JSON schema."""

    def test_has_experiment_field(self) -> None:
        """Artifact has experiment=302."""
        batch1 = _make_dummy_batch(1, correct=30)
        batch2 = _make_dummy_batch(2, correct=32)
        summary = _make_dummy_summary()
        artifact = build_artifact(
            batch1=batch1,
            batch2=batch2,
            constraint_summary=summary,
            inference_mode="simulated",
        )
        assert artifact["experiment"] == 302

    def test_batch1_accuracy_field(self) -> None:
        """Artifact has batch1_accuracy field."""
        batch1 = _make_dummy_batch(1, correct=30)
        batch2 = _make_dummy_batch(2, correct=32)
        summary = _make_dummy_summary()
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="simulated"
        )
        assert "batch1_accuracy" in artifact
        assert artifact["batch1_accuracy"] == pytest.approx(30 / 50)

    def test_batch2_accuracy_field(self) -> None:
        """Artifact has batch2_accuracy field."""
        batch1 = _make_dummy_batch(1, correct=30)
        batch2 = _make_dummy_batch(2, correct=32)
        summary = _make_dummy_summary()
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="simulated"
        )
        assert "batch2_accuracy" in artifact
        assert artifact["batch2_accuracy"] == pytest.approx(32 / 50)

    def test_improvement_delta_positive(self) -> None:
        """improvement_delta > 0 when batch2 better than batch1."""
        batch1 = _make_dummy_batch(1, correct=30)
        batch2 = _make_dummy_batch(2, correct=35)
        summary = _make_dummy_summary()
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="simulated"
        )
        assert artifact["improvement_delta"] == pytest.approx(5 / 50)

    def test_improvement_delta_negative_allowed(self) -> None:
        """improvement_delta can be negative — honest reporting required."""
        batch1 = _make_dummy_batch(1, correct=35)
        batch2 = _make_dummy_batch(2, correct=30)
        summary = _make_dummy_summary()
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="simulated"
        )
        assert artifact["improvement_delta"] < 0.0

    def test_n_new_constraints_present(self) -> None:
        """Artifact has n_new_constraints field."""
        batch1 = _make_dummy_batch(1)
        batch2 = _make_dummy_batch(2)
        summary = _make_dummy_summary(n_new=2)
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="simulated"
        )
        assert artifact["n_new_constraints"] == 2

    def test_n_new_constraints_non_negative_in_artifact(self) -> None:
        """REQ-LEARN-010: n_new_constraints in artifact is always >= 0."""
        batch1 = _make_dummy_batch(1)
        batch2 = _make_dummy_batch(2)
        summary = _make_dummy_summary(n_new=0)
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="simulated"
        )
        assert artifact["n_new_constraints"] >= 0

    def test_constraint_count_before_and_after(self) -> None:
        """Artifact has constraint_count_before and constraint_count_after."""
        batch1 = _make_dummy_batch(1)
        batch2 = _make_dummy_batch(2)
        summary = _make_dummy_summary(n_new=1)
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="simulated"
        )
        assert "constraint_count_before" in artifact
        assert "constraint_count_after" in artifact
        assert artifact["constraint_count_after"] >= artifact["constraint_count_before"]

    def test_generated_constraint_log_present(self) -> None:
        """Artifact has generated_constraint_log list."""
        batch1 = _make_dummy_batch(1)
        batch2 = _make_dummy_batch(2)
        summary = _make_dummy_summary()
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="simulated"
        )
        assert "generated_constraint_log" in artifact
        assert isinstance(artifact["generated_constraint_log"], list)

    def test_memory_patterns_found_present(self) -> None:
        """Artifact has memory_patterns_found field."""
        batch1 = _make_dummy_batch(1)
        batch2 = _make_dummy_batch(2)
        summary = _make_dummy_summary()
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="simulated"
        )
        assert "memory_patterns_found" in artifact
        assert artifact["memory_patterns_found"] >= 0

    def test_inference_mode_simulated(self) -> None:
        """Artifact has inference_mode='simulated' when GPU unavailable."""
        batch1 = _make_dummy_batch(1)
        batch2 = _make_dummy_batch(2)
        summary = _make_dummy_summary()
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="simulated"
        )
        assert artifact["inference_mode"] == "simulated"

    def test_inference_mode_live_gpu(self) -> None:
        """Artifact has inference_mode='live_gpu' when GPU available."""
        batch1 = _make_dummy_batch(1)
        batch2 = _make_dummy_batch(2)
        summary = _make_dummy_summary()
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="live_gpu"
        )
        assert artifact["inference_mode"] == "live_gpu"

    def test_artifact_is_json_serializable(self) -> None:
        """Artifact dict must be JSON-serializable."""
        batch1 = _make_dummy_batch(1)
        batch2 = _make_dummy_batch(2)
        summary = _make_dummy_summary()
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="simulated"
        )
        serialized = json.dumps(artifact)
        assert len(serialized) > 0

    def test_batch_split_50_each(self) -> None:
        """Artifact records show exactly 50 questions in each batch."""
        batch1 = _make_dummy_batch(1, correct=25)
        batch2 = _make_dummy_batch(2, correct=28)
        summary = _make_dummy_summary()
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="simulated"
        )
        assert artifact["batch1"]["n_questions"] == 50
        assert artifact["batch2"]["n_questions"] == 50

    def test_run_date_present(self) -> None:
        """Artifact has run_date field (YYYYMMDD format)."""
        batch1 = _make_dummy_batch(1)
        batch2 = _make_dummy_batch(2)
        summary = _make_dummy_summary()
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="simulated"
        )
        assert "run_date" in artifact
        assert len(artifact["run_date"]) == 8
        assert artifact["run_date"].isdigit()

    def test_title_present(self) -> None:
        """Artifact has title field."""
        batch1 = _make_dummy_batch(1)
        batch2 = _make_dummy_batch(2)
        summary = _make_dummy_summary()
        artifact = build_artifact(
            batch1=batch1, batch2=batch2, constraint_summary=summary, inference_mode="simulated"
        )
        assert "title" in artifact
        assert len(artifact["title"]) > 0
