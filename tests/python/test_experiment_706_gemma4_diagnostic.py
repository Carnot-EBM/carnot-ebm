"""Tests for Exp 706: Gemma4-E4B-it VR Failure Mode Diagnostic.

WHY THIS TEST FILE EXISTS:
    Exp 694 revealed VR helps Qwen3.5-0.8B but hurts Gemma4-E4B-it.  Exp 706
    instruments the VR pipeline on 50 Gemma4 responses to identify which of three
    hypotheses explains the degradation:
      1. Extraction FP (extractor fires on correct responses)
      2. Repair regression (repair corrupts correct reasoning)
      3. Threshold miscalibration (extractor misses actual errors)

    This test suite validates:
    1. Instrument mode captures all required per-response fields (REQ-VERIFY-144).
    2. Failure mode classification logic is correct for all branching cases (REQ-VERIFY-145).
    3. The deliverable JSON is written, parseable, and contains the required schema fields.

Spec: REQ-VERIFY-144, REQ-VERIFY-145,
      SCENARIO-VERIFY-144, SCENARIO-VERIFY-145
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

_REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

_DELIVERABLE = _REPO_ROOT / "results" / "experiment_706_gemma4_vr_diagnostic.json"

# Import module under test.
import experiment_706_gemma4_diagnostic as exp706


# ---------------------------------------------------------------------------
# REQ-VERIFY-144: Instrument mode field completeness
# ---------------------------------------------------------------------------


class TestInstrumentModeFields:
    """Validate that instrument mode captures all required per-response fields.

    WHY: REQ-VERIFY-144 mandates that every per-response record contain
    exactly the five fields listed in the spec.  Missing fields prevent
    downstream failure mode classification from running correctly.
    Spec: REQ-VERIFY-144, SCENARIO-VERIFY-144.
    """

    _REQUIRED_FIELDS = {
        "extractor_fired",
        "constraint_type",
        "repair_applied",
        "answer_changed",
        "final_correct",
    }

    def _make_pipeline(self) -> Any:
        """Build a minimal VerifyRepairPipeline with no LLM for instrument tests."""
        from python.carnot.pipeline.verify_repair import VerifyRepairPipeline
        return VerifyRepairPipeline(
            model=None,
            domains=["arithmetic"],
            max_repairs=0,
            extractor=None,
            semantic_grounding_verifier=None,
            semantic_verifier_v2=None,
            timeout_seconds=30,
            memory=None,
            template_library=None,
            session_memory=None,
            constraint_memory=None,
            nup_probe=None,
            nup_probe_threshold=0.5,
        )

    def _make_extractor(self) -> Any:
        """Build AutoExtractor in arithmetic-only mode for instrument tests."""
        from python.carnot.pipeline.extract import AutoExtractor
        return AutoExtractor(enable_factual_extractor=False)

    def test_all_required_fields_present_for_correct_response(self) -> None:
        """Instrument record for a correct response must contain all five required fields.

        WHY: REQ-VERIFY-144-1 says records are emitted for EVERY response.  A record
        missing any of the five fields breaks the classify_failure_mode computation.
        Spec: REQ-VERIFY-144, SCENARIO-VERIFY-144.
        """
        pipeline = self._make_pipeline()
        extractor = self._make_extractor()
        response = "The answer is 8."
        rec = exp706._instrument_response(pipeline, extractor, "2 + 6 = ?", response, 8)
        for field in self._REQUIRED_FIELDS:
            assert field in rec, f"Required field '{field}' missing from instrument record"

    def test_all_required_fields_present_for_incorrect_response(self) -> None:
        """Instrument record for an incorrect response must contain all five required fields.

        WHY: REQ-VERIFY-144-1 applies to ALL responses, not just correct ones.  Incorrect
        responses must also emit complete records so threshold_miss_rate can be computed.
        Spec: REQ-VERIFY-144, SCENARIO-VERIFY-144.
        """
        pipeline = self._make_pipeline()
        extractor = self._make_extractor()
        response = "The answer is 5."  # wrong — ground truth is 8
        rec = exp706._instrument_response(pipeline, extractor, "2 + 6 = ?", response, 8)
        for field in self._REQUIRED_FIELDS:
            assert field in rec, f"Required field '{field}' missing from instrument record"

    def test_constraint_type_is_none_when_extractor_does_not_fire(self) -> None:
        """constraint_type must be 'none' when extractor_fired is False.

        WHY: REQ-VERIFY-144-2 — this field is used to determine WHICH extractor
        triggers on Gemma's outputs.  'none' is the correct sentinel for no-fire.
        Spec: REQ-VERIFY-144-2.
        """
        pipeline = self._make_pipeline()
        extractor = self._make_extractor()
        # Pure natural language with no arithmetic claim — extractor should not fire.
        response = "Based on careful reasoning, the total is eight."
        rec = exp706._instrument_response(pipeline, extractor, "What is 2+6?", response, 8)
        if not rec["extractor_fired"]:
            assert rec["constraint_type"] == "none", (
                f"constraint_type should be 'none' when extractor_fired=False, got {rec['constraint_type']}"
            )

    def test_repair_applied_is_false_in_verify_only_mode(self) -> None:
        """repair_applied must always be False when no LLM is loaded.

        WHY: REQ-VERIFY-144-3 — verify-only mode cannot repair; if this were True it
        would indicate a bug where the pipeline is claiming repair without a model.
        Spec: REQ-VERIFY-144-3.
        """
        pipeline = self._make_pipeline()
        extractor = self._make_extractor()
        response = "The answer is 3."  # wrong for "2+6=?"
        rec = exp706._instrument_response(pipeline, extractor, "2 + 6 = ?", response, 8)
        assert rec["repair_applied"] is False

    def test_answer_changed_is_false_when_repair_not_applied(self) -> None:
        """answer_changed must be False when repair_applied is False.

        WHY: REQ-VERIFY-144-4 — answer can only change if repair was applied.
        This invariant prevents spurious answer_changed=True counts in fp_rate computation.
        Spec: REQ-VERIFY-144-4.
        """
        pipeline = self._make_pipeline()
        extractor = self._make_extractor()
        response = "The answer is 3."
        rec = exp706._instrument_response(pipeline, extractor, "2 + 6 = ?", response, 8)
        assert not rec["repair_applied"]
        assert rec["answer_changed"] is False

    def test_final_correct_true_for_right_answer(self) -> None:
        """final_correct must be True when the response gives the correct answer.

        WHY: The primary diagnostic signal is final_correct.  If it misreports correct
        answers as wrong, fp_rate_on_correct will be inflated, producing a false diagnosis.
        Spec: REQ-VERIFY-144.
        """
        pipeline = self._make_pipeline()
        extractor = self._make_extractor()
        response = "The answer is 8."
        rec = exp706._instrument_response(pipeline, extractor, "2 + 6 = ?", response, 8)
        assert rec["final_correct"] is True

    def test_final_correct_false_for_wrong_answer(self) -> None:
        """final_correct must be False when the response gives the wrong answer.

        WHY: Wrong answers need to be captured accurately so threshold_miss_rate is
        not deflated by missed incorrect detections.
        Spec: REQ-VERIFY-144.
        """
        pipeline = self._make_pipeline()
        extractor = self._make_extractor()
        response = "The answer is 5."  # wrong — ground truth is 8
        rec = exp706._instrument_response(pipeline, extractor, "2 + 6 = ?", response, 8)
        assert rec["final_correct"] is False


# ---------------------------------------------------------------------------
# REQ-VERIFY-145: Failure mode classification logic
# ---------------------------------------------------------------------------


class TestFailureModeClassification:
    """Validate the classify_failure_mode logic for all branching cases.

    WHY: REQ-VERIFY-145 defines exact thresholds for each failure mode.
    These tests verify those thresholds are correctly implemented so the
    conductor can route to the right fix based on the honest_verdict.
    Spec: REQ-VERIFY-145, SCENARIO-VERIFY-145.
    """

    @staticmethod
    def _make_records(
        n: int,
        extractor_fired: bool = False,
        original_correct: bool = True,
        final_correct: bool = True,
    ) -> list[dict[str, Any]]:
        """Helper to create a batch of identical instrument records."""
        return [
            {
                "extractor_fired": extractor_fired,
                "constraint_type": "arithmetic" if extractor_fired else "none",
                "repair_applied": False,
                "answer_changed": False,
                "original_correct": original_correct,
                "final_correct": final_correct,
            }
        ] * n

    def test_extraction_fp_detected_when_rate_exceeds_threshold(self) -> None:
        """failure_mode = 'extraction_fp' when fp_rate_on_correct > 0.20.

        WHY: REQ-VERIFY-145-1 sets the threshold at 0.20.  If 40% of correct responses
        trigger the extractor, that is a clear false-positive problem.
        Spec: REQ-VERIFY-145-1, SCENARIO-VERIFY-145.
        """
        # 10 correct responses, 4 with extractor_fired=True → fp_rate=0.40 > 0.20
        records_correct = (
            self._make_records(4, extractor_fired=True, original_correct=True, final_correct=True)
            + self._make_records(6, extractor_fired=False, original_correct=True, final_correct=True)
        )
        records_incorrect = self._make_records(
            10, extractor_fired=True, original_correct=False, final_correct=False
        )
        result = exp706.classify_failure_mode(records_correct, records_incorrect)
        assert result["failure_mode"] == "extraction_fp", (
            f"Expected 'extraction_fp', got: {result['failure_mode']}"
        )
        assert result["honest_verdict"] == "failure_mode_identified"

    def test_no_failure_detected_below_all_thresholds(self) -> None:
        """failure_mode = 'no_clear_failure' when all rates are below their thresholds.

        WHY: REQ-VERIFY-145 must not over-diagnose.  Low rates on all three dimensions
        should produce 'no_clear_failure' to avoid routing to the wrong fix.
        Spec: REQ-VERIFY-145.
        """
        # fp_rate=0.10, regression_rate=0.0, miss_rate=0.10 — all below threshold
        records_correct = (
            self._make_records(1, extractor_fired=True)
            + self._make_records(9, extractor_fired=False)
        )
        records_incorrect = (
            self._make_records(1, extractor_fired=False, original_correct=False, final_correct=False)
            + self._make_records(9, extractor_fired=True, original_correct=False, final_correct=False)
        )
        result = exp706.classify_failure_mode(records_correct, records_incorrect)
        assert result["failure_mode"] == "no_clear_failure"
        assert result["honest_verdict"] == "failure_mode_ambiguous"

    def test_threshold_too_high_when_miss_rate_exceeds_0_50(self) -> None:
        """failure_mode = 'threshold_too_high' when threshold_miss_rate > 0.50.

        WHY: If more than half of incorrect responses escape detection, the extractor
        is miscalibrated — its energy threshold is too high, missing real errors.
        Spec: REQ-VERIFY-145-1.
        """
        records_correct = self._make_records(10, extractor_fired=False)
        # 6 out of 10 incorrect responses NOT detected → miss_rate=0.60 > 0.50
        records_incorrect = (
            self._make_records(6, extractor_fired=False, original_correct=False, final_correct=False)
            + self._make_records(4, extractor_fired=True, original_correct=False, final_correct=False)
        )
        result = exp706.classify_failure_mode(records_correct, records_incorrect)
        assert result["failure_mode"] == "threshold_too_high"
        assert result["honest_verdict"] == "failure_mode_identified"

    def test_combined_when_multiple_modes_exceed_threshold(self) -> None:
        """failure_mode = 'combined' when multiple conditions exceed their thresholds.

        WHY: A combined failure requires combined fixes — separately addressing only
        one mode would leave the other in place.  The conductor needs to know both.
        Spec: REQ-VERIFY-145.
        """
        # fp_rate=0.40 > 0.20 AND miss_rate=0.60 > 0.50
        records_correct = (
            self._make_records(4, extractor_fired=True)
            + self._make_records(6, extractor_fired=False)
        )
        records_incorrect = (
            self._make_records(6, extractor_fired=False, original_correct=False, final_correct=False)
            + self._make_records(4, extractor_fired=True, original_correct=False, final_correct=False)
        )
        result = exp706.classify_failure_mode(records_correct, records_incorrect)
        assert result["failure_mode"] == "combined"
        assert result["honest_verdict"] == "failure_mode_identified"

    def test_honest_verdict_identified_for_non_clear_modes(self) -> None:
        """honest_verdict = 'failure_mode_identified' for all non-ambiguous modes.

        WHY: REQ-VERIFY-145-2 — the conductor routes based on honest_verdict.
        Any non-ambiguous failure must produce 'failure_mode_identified'.
        Spec: REQ-VERIFY-145-2.
        """
        records_correct = (
            self._make_records(5, extractor_fired=True)
            + self._make_records(5, extractor_fired=False)
        )
        records_incorrect = self._make_records(
            10, extractor_fired=True, original_correct=False, final_correct=False
        )
        result = exp706.classify_failure_mode(records_correct, records_incorrect)
        # fp_rate=0.50 > 0.20 → extraction_fp
        assert result["honest_verdict"] == "failure_mode_identified"

    def test_honest_verdict_ambiguous_for_no_clear_failure(self) -> None:
        """honest_verdict = 'failure_mode_ambiguous' when failure_mode = 'no_clear_failure'.

        WHY: REQ-VERIFY-145-3 — ambiguous verdict routes to deeper investigation
        rather than a premature fix commitment.
        Spec: REQ-VERIFY-145-3.
        """
        records_correct = self._make_records(10, extractor_fired=False)
        records_incorrect = (
            self._make_records(4, extractor_fired=False, original_correct=False, final_correct=False)
            + self._make_records(6, extractor_fired=True, original_correct=False, final_correct=False)
        )
        result = exp706.classify_failure_mode(records_correct, records_incorrect)
        # fp_rate=0.0, miss_rate=0.40 — all below threshold
        assert result["failure_mode"] == "no_clear_failure"
        assert result["honest_verdict"] == "failure_mode_ambiguous"

    def test_fp_rate_computed_correctly(self) -> None:
        """fp_rate_on_correct must equal fp_count / n_correct.

        WHY: The exact rate value is logged in the artifact and used by the reconciler
        to cross-check the failure_mode label.  An off-by-one would produce a wrong label.
        Spec: REQ-VERIFY-145.
        """
        records_correct = (
            self._make_records(3, extractor_fired=True)
            + self._make_records(7, extractor_fired=False)
        )
        records_incorrect = self._make_records(10, extractor_fired=True, original_correct=False, final_correct=False)
        result = exp706.classify_failure_mode(records_correct, records_incorrect)
        assert abs(result["fp_rate_on_correct"] - 0.3) < 0.001, (
            f"fp_rate_on_correct should be 0.30, got {result['fp_rate_on_correct']}"
        )

    def test_threshold_miss_rate_computed_correctly(self) -> None:
        """threshold_miss_rate must equal miss_count / n_incorrect.

        WHY: Same reason as above — wrong rate leads to wrong failure_mode label.
        Spec: REQ-VERIFY-145.
        """
        records_correct = self._make_records(10, extractor_fired=False)
        records_incorrect = (
            self._make_records(3, extractor_fired=False, original_correct=False, final_correct=False)
            + self._make_records(7, extractor_fired=True, original_correct=False, final_correct=False)
        )
        result = exp706.classify_failure_mode(records_correct, records_incorrect)
        assert abs(result["threshold_miss_rate"] - 0.3) < 0.001, (
            f"threshold_miss_rate should be 0.30, got {result['threshold_miss_rate']}"
        )

    def test_empty_records_do_not_raise(self) -> None:
        """classify_failure_mode must not raise when given empty record lists.

        WHY: Edge case — if the model fails to generate any responses, the records
        lists will be empty.  A divide-by-zero crash would prevent the artifact from
        being written, halting the conductor.
        Spec: REQ-VERIFY-145.
        """
        result = exp706.classify_failure_mode([], [])
        assert result["failure_mode"] in {"no_clear_failure", "combined", "extraction_fp",
                                           "repair_regression", "threshold_too_high"}


# ---------------------------------------------------------------------------
# Deliverable schema tests
# ---------------------------------------------------------------------------


class TestDeliverableSchema:
    """Validate the experiment deliverable JSON has all required fields.

    WHY: The conductor checks the deliverable for schema completeness before
    scheduling the next experiment.  Missing required fields halt the pipeline.
    Spec: REQ-VERIFY-144, REQ-VERIFY-145.
    """

    _REQUIRED_FIELDS = [
        "experiment",
        "title",
        "run_date",
        "started_at",
        "finished_at",
        "duration_s",
        "status",
        "schema",
        "fp_rate_on_correct",
        "repair_regression_rate",
        "threshold_miss_rate",
        "failure_mode",
        "honest_verdict",
        "n_correct_tested",
        "n_incorrect_tested",
        "invariant_violations",
    ]

    def _load(self) -> dict:
        assert _DELIVERABLE.exists(), (
            f"Exp 706 deliverable missing: {_DELIVERABLE}. "
            "Run scripts/experiment_706_gemma4_diagnostic.py first."
        )
        return json.loads(_DELIVERABLE.read_text())

    def test_deliverable_exists(self) -> None:
        """Deliverable JSON must exist at the expected path.

        WHY: The conductor validates completion by checking this file.
        Missing deliverable halts the research pipeline.
        Spec: REQ-VERIFY-144.
        """
        assert _DELIVERABLE.exists(), f"Deliverable not found: {_DELIVERABLE}"

    def test_deliverable_is_valid_json(self) -> None:
        """Deliverable must be parseable JSON.

        WHY: Downstream tooling (conductor, reconciler) reads this as JSON.
        Invalid JSON silently skips reconciliation.
        Spec: REQ-VERIFY-144.
        """
        data = json.loads(_DELIVERABLE.read_text())
        assert isinstance(data, dict)

    def test_required_fields_present(self) -> None:
        """All required schema fields must be present in the deliverable.

        WHY: Missing fields cause silent downstream failures that are hard to diagnose.
        Spec: REQ-VERIFY-144, REQ-VERIFY-145.
        """
        data = self._load()
        for field in self._REQUIRED_FIELDS:
            assert field in data, f"Required field '{field}' missing from Exp 706 artifact"

    def test_experiment_id_is_706(self) -> None:
        """experiment field must equal 706.

        WHY: Prevents conductor from confusing this result with a prior run.
        Spec: REQ-VERIFY-144.
        """
        data = self._load()
        assert data["experiment"] == 706

    def test_status_is_success(self) -> None:
        """status must be 'success'.

        WHY: Non-success status causes the conductor to retry the experiment.
        Spec: REQ-VERIFY-144.
        """
        data = self._load()
        assert data["status"] == "success"

    def test_failure_mode_is_valid_label(self) -> None:
        """failure_mode must be one of the five valid labels.

        WHY: An unexpected label string prevents the conductor from routing
        to the correct follow-up experiment.
        Spec: REQ-VERIFY-145.
        """
        valid = {
            "extraction_fp",
            "repair_regression",
            "threshold_too_high",
            "combined",
            "no_clear_failure",
        }
        data = self._load()
        assert data["failure_mode"] in valid, (
            f"failure_mode '{data['failure_mode']}' is not a valid label. Expected one of: {valid}"
        )

    def test_honest_verdict_is_valid(self) -> None:
        """honest_verdict must be one of the two valid verdict strings.

        WHY: REQ-VERIFY-145-2/145-3 — the conductor routes based on this string.
        Spec: REQ-VERIFY-145-2, REQ-VERIFY-145-3.
        """
        valid = {"failure_mode_identified", "failure_mode_ambiguous"}
        data = self._load()
        assert data["honest_verdict"] in valid, (
            f"honest_verdict '{data['honest_verdict']}' invalid. Expected one of: {valid}"
        )

    def test_n_correct_tested_is_25(self) -> None:
        """n_correct_tested must equal 25.

        WHY: The experiment spec mandates exactly 25 correct-set responses.
        A different count indicates a setup failure.
        Spec: REQ-VERIFY-144.
        """
        data = self._load()
        assert data["n_correct_tested"] == 25, (
            f"Expected n_correct_tested=25, got {data['n_correct_tested']}"
        )

    def test_n_incorrect_tested_is_25(self) -> None:
        """n_incorrect_tested must equal 25.

        WHY: The experiment spec mandates exactly 25 incorrect-set responses.
        Spec: REQ-VERIFY-144.
        """
        data = self._load()
        assert data["n_incorrect_tested"] == 25, (
            f"Expected n_incorrect_tested=25, got {data['n_incorrect_tested']}"
        )

    def test_invariant_violations_empty(self) -> None:
        """invariant_violations must be an empty list.

        WHY: Non-empty invariant_violations signals a runtime schema error.
        Spec: REQ-VERIFY-144.
        """
        data = self._load()
        assert data.get("invariant_violations") == []

    def test_per_response_records_count(self) -> None:
        """per_response_records must contain exactly 50 entries.

        WHY: 25 correct + 25 incorrect = 50 records.  A different count means
        some responses were skipped silently.
        Spec: REQ-VERIFY-144-1.
        """
        data = self._load()
        records = data.get("per_response_records", [])
        assert len(records) == 50, f"Expected 50 per_response_records, got {len(records)}"

    def test_per_response_records_have_required_fields(self) -> None:
        """Every per_response_record must contain the five required instrument fields.

        WHY: REQ-VERIFY-144-1 mandates records for every response.  Missing fields
        prevent failure mode rate computation.
        Spec: REQ-VERIFY-144, SCENARIO-VERIFY-144.
        """
        required = {"extractor_fired", "constraint_type", "repair_applied", "answer_changed", "final_correct"}
        data = self._load()
        for i, rec in enumerate(data.get("per_response_records", [])):
            for field in required:
                assert field in rec, (
                    f"Record {i} missing required field '{field}'"
                )


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Unit tests for internal helper functions.

    WHY: _extract_numeric_answer and _answers_match are used in every instrument
    call.  Bugs here silently corrupt all correctness labels.
    Spec: REQ-VERIFY-144.
    """

    def test_extract_numeric_answer_from_plain_number(self) -> None:
        """_extract_numeric_answer extracts a plain number from response text.

        WHY: Some model responses end with just the number — we must handle this.
        Spec: REQ-VERIFY-144.
        """
        result = exp706._extract_numeric_answer("The answer is 42.")
        assert result == 42.0

    def test_extract_numeric_answer_returns_none_for_no_number(self) -> None:
        """_extract_numeric_answer returns None when no number is present.

        WHY: A None return from extract prevents false correct/wrong labels from
        being assigned when the model produces a non-numeric response.
        Spec: REQ-VERIFY-144.
        """
        result = exp706._extract_numeric_answer("I don't know the answer.")
        assert result is None

    def test_answers_match_within_tolerance(self) -> None:
        """_answers_match returns True when values differ by less than 0.5.

        WHY: GSM8K answers are integers; float representations like '35.0' must
        match the ground truth integer 35.
        Spec: REQ-VERIFY-144.
        """
        assert exp706._answers_match(35.0, 35) is True

    def test_answers_match_false_for_wrong_answer(self) -> None:
        """_answers_match returns False when values differ by more than 0.5.

        WHY: We must accurately detect wrong answers to compute threshold_miss_rate.
        Spec: REQ-VERIFY-144.
        """
        assert exp706._answers_match(5.0, 8) is False

    def test_answers_match_handles_none(self) -> None:
        """_answers_match returns False when either argument is None.

        WHY: None means the answer could not be extracted; treating it as correct
        would inflate the correct count and deflate fp_rate.
        Spec: REQ-VERIFY-144.
        """
        assert exp706._answers_match(None, 8) is False
        assert exp706._answers_match(8.0, None) is False
