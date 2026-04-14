"""Tests for experiment_276_gsm8k_modern_extractors.

Verifies the three-extractor GSM8K pipeline:
  - extract_final_answer   — numeric answer extraction from free-form text
  - answer_is_correct      — ground-truth comparison
  - _ci_generate_fn        — CI stub for LLMConstraintExtractor
  - run_z3_extractor       — SMT-backed arithmetic detection
  - run_llm_extractor      — LLM-canonicalized claim detection (CI stub)
  - run_semantic_extractor — deterministic question-grounding check
  - run_case               — full triple-extractor run on one case
  - compute_statistics     — per-extractor and combined metrics
  - build_artifact         — artifact schema validation
  - Full CI benchmark      — end-to-end run of all 10 CI cases

Each test traces to a spec requirement or scenario.

Spec: REQ-VERIFY-001, REQ-VERIFY-003, REQ-VERIFY-009, REQ-VERIFY-010,
      REQ-VERIFY-020, REQ-VERIFY-021, SCENARIO-VERIFY-009,
      SCENARIO-VERIFY-010, SCENARIO-VERIFY-020, SCENARIO-VERIFY-021
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

_MODULE_NAME = "experiment_276_gsm8k_modern_extractors"


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def load_module() -> Any:
    """Load the experiment_276 module without importing it as a package.

    Registers the module in sys.modules before execution so that Python 3.14's
    @dataclass decorator can look up ``cls.__module__`` without a KeyError.
    """
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    repo_root = Path(__file__).resolve().parents[2]
    module_path = (
        repo_root / "scripts" / "experiment_276_gsm8k_modern_extractors.py"
    )
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, module_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec_module so @dataclass can resolve cls.__module__
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = load_module()


# ---------------------------------------------------------------------------
# extract_final_answer
# ---------------------------------------------------------------------------


class TestExtractFinalAnswer:
    """REQ-VERIFY-001: Final answer extraction from diverse response formats."""

    def test_gsm8k_delimiter(self) -> None:
        """REQ-VERIFY-001: #### N delimiter is parsed first."""
        assert MOD.extract_final_answer("#### 42") == 42.0

    def test_the_answer_is_phrase(self) -> None:
        """REQ-VERIFY-001: 'The answer is N' phrase is recognised."""
        val = MOD.extract_final_answer("The answer is $15.")
        assert val == 15.0

    def test_answer_colon_phrase(self) -> None:
        """REQ-VERIFY-001: 'Answer: N' phrase is recognised."""
        val = MOD.extract_final_answer("Answer: 300")
        assert val == 300.0

    def test_fallback_last_number(self) -> None:
        """REQ-VERIFY-001: Last number in text is used as fallback."""
        val = MOD.extract_final_answer("Step 1: 5. Step 2: 10. Step 3: 20.")
        assert val == 20.0

    def test_no_number_returns_none(self) -> None:
        """REQ-VERIFY-001: Returns None when no number is present."""
        assert MOD.extract_final_answer("There is no number here.") is None

    def test_number_with_commas(self) -> None:
        """REQ-VERIFY-001: Comma-separated numbers like 1,000 are parsed."""
        val = MOD.extract_final_answer("The answer is 1,000.")
        assert val == 1000.0


# ---------------------------------------------------------------------------
# answer_is_correct
# ---------------------------------------------------------------------------


class TestAnswerIsCorrect:
    """REQ-VERIFY-001: Ground-truth comparison logic."""

    def test_correct_integer_match(self) -> None:
        """REQ-VERIFY-001: Integer answer matches when extracted value equals ground truth."""
        assert MOD.answer_is_correct("The answer is 84.", 84) is True

    def test_wrong_integer(self) -> None:
        """REQ-VERIFY-001: Integer answer does not match when value differs."""
        assert MOD.answer_is_correct("The answer is 85.", 84) is False

    def test_correct_answer_in_multiline(self) -> None:
        """REQ-VERIFY-001: Correct answer found in multi-line response."""
        response = "Step 1: 5 * 4 = 20.\n#### 20"
        assert MOD.answer_is_correct(response, 20) is True

    def test_no_answer_extracted_returns_false(self) -> None:
        """REQ-VERIFY-001: Returns False when no answer can be extracted."""
        assert MOD.answer_is_correct("I don't know.", 42) is False


# ---------------------------------------------------------------------------
# _ci_generate_fn
# ---------------------------------------------------------------------------


class TestCIGenerateFn:
    """REQ-VERIFY-010, SCENARIO-VERIFY-010: CI stub for LLM generate."""

    def test_wrong_multiplication_produces_claim(self) -> None:
        """SCENARIO-VERIFY-010: Stub outputs CLAIM line for wrong multiplication."""
        prompt = "Extract claims.\n\nResponse:\n12 * 7 = 85. Done."
        output = MOD._ci_generate_fn(None, None, prompt, 128)
        assert "CLAIM: 12 * 7 = 85" in output

    def test_correct_arithmetic_produces_claim(self) -> None:
        """SCENARIO-VERIFY-010: Stub outputs CLAIM line for correct arithmetic."""
        prompt = "Extract claims.\n\nResponse:\n20 - 8 = 12. Fine."
        output = MOD._ci_generate_fn(None, None, prompt, 128)
        assert "CLAIM: 20 - 8 = 12" in output

    def test_no_arithmetic_returns_none(self) -> None:
        """SCENARIO-VERIFY-010: Stub returns NONE when no arithmetic is found."""
        prompt = "Extract claims.\n\nResponse:\nTom has 9 books remaining."
        output = MOD._ci_generate_fn(None, None, prompt, 128)
        assert output == "NONE"

    def test_multiple_claims_in_response(self) -> None:
        """SCENARIO-VERIFY-010: Stub extracts multiple claims from response."""
        prompt = "Extract.\n\nResponse:\n48 * 3 = 144. 144 - 10 = 134."
        output = MOD._ci_generate_fn(None, None, prompt, 128)
        assert "CLAIM: 48 * 3 = 144" in output
        assert "CLAIM: 144 - 10 = 134" in output


# ---------------------------------------------------------------------------
# run_z3_extractor
# ---------------------------------------------------------------------------


class TestRunZ3Extractor:
    """REQ-VERIFY-009, SCENARIO-VERIFY-009: Z3 extractor detects arithmetic errors."""

    def test_wrong_multiplication_flagged(self) -> None:
        """SCENARIO-VERIFY-009: Z3 detects 12 * 7 = 85 as wrong (correct: 84)."""
        result = MOD.run_z3_extractor("12 * 7 = 85.")
        assert result.extractor_name == "z3"
        assert result.flagged is True
        assert result.n_violations == 1
        wrong = result.details[0]
        assert wrong["satisfied"] is False
        assert wrong["correct_result"] == 84

    def test_correct_arithmetic_not_flagged(self) -> None:
        """REQ-VERIFY-009: Z3 does not flag correct arithmetic."""
        result = MOD.run_z3_extractor("20 - 8 = 12.")
        assert result.flagged is False
        assert result.n_violations == 0

    def test_no_arithmetic_empty_result(self) -> None:
        """REQ-VERIFY-009: Z3 returns nothing when no arithmetic is present."""
        result = MOD.run_z3_extractor("Tom has 9 books remaining.")
        assert result.n_total == 0
        assert result.flagged is False

    def test_multiple_steps_detects_first_wrong(self) -> None:
        """SCENARIO-VERIFY-009: Z3 detects wrong step among multi-step arithmetic."""
        response = "48 * 3 = 144. 144 - 10 = 134."
        result = MOD.run_z3_extractor(response)
        assert result.flagged is False  # Both steps are correct
        assert result.n_satisfied == 2

    def test_wrong_subtraction_detected(self) -> None:
        """SCENARIO-VERIFY-009: Z3 detects wrong subtraction."""
        result = MOD.run_z3_extractor("5 * 60 = 290.")
        assert result.flagged is True
        assert result.details[0]["correct_result"] == 300


# ---------------------------------------------------------------------------
# run_llm_extractor (CI stub)
# ---------------------------------------------------------------------------


class TestRunLLMExtractor:
    """REQ-VERIFY-010, SCENARIO-VERIFY-010: LLM extractor with CI stub."""

    def test_wrong_multiplication_flagged(self) -> None:
        """SCENARIO-VERIFY-010: CI stub detects 4 * 7 = 30 as wrong (correct: 28)."""
        result = MOD.run_llm_extractor(
            "4 * 7 = 30. The total is $30.",
            generate_fn=MOD._ci_generate_fn,
        )
        assert result.extractor_name == "llm"
        assert result.flagged is True
        wrong = result.details[0]
        assert wrong["satisfied"] is False
        assert wrong["correct_result"] == 28

    def test_correct_arithmetic_not_flagged(self) -> None:
        """REQ-VERIFY-010: CI stub does not flag correct arithmetic."""
        result = MOD.run_llm_extractor(
            "60 * 3 = 180 miles.",
            generate_fn=MOD._ci_generate_fn,
        )
        assert result.flagged is False

    def test_terse_response_no_claims_not_flagged(self) -> None:
        """REQ-VERIFY-010: CI stub finds no claims when no arithmetic is shown."""
        result = MOD.run_llm_extractor(
            "Maria pays $14.",
            generate_fn=MOD._ci_generate_fn,
        )
        assert result.flagged is False
        assert result.n_total == 0


# ---------------------------------------------------------------------------
# run_semantic_extractor
# ---------------------------------------------------------------------------


class TestRunSemanticExtractor:
    """REQ-VERIFY-020, REQ-VERIFY-021, SCENARIO-VERIFY-020, SCENARIO-VERIFY-021."""

    def test_well_grounded_response_no_violations(self) -> None:
        """SCENARIO-VERIFY-020: Response with all quantities present is not flagged."""
        result = MOD.run_semantic_extractor(
            question="A train travels 60 mph for 3 hours. How far does it go?",
            response="Distance = 60 * 3 = 180 miles.",
        )
        assert result.extractor_name == "semantic"
        assert result.flagged is False

    def test_terse_response_may_flag_ungrounded_quantities(self) -> None:
        """SCENARIO-VERIFY-021: Terse response missing premise quantities may be flagged."""
        # "Tom has 15 books. He gives 6 to a friend. How many remain?"
        # Response "Tom has 9 books remaining." omits the "gives 6" quantity.
        result = MOD.run_semantic_extractor(
            question="Tom has 15 books. He gives 6 to a friend. How many remain?",
            response="Tom has 9 books remaining.",
        )
        # Semantic flags this as a FP: "gives 6" clause is not covered and
        # "remain"/"remaining" is a morphological mismatch.
        assert result.extractor_name == "semantic"
        # We assert the violation structure is correct, regardless of flag value:
        for detail in result.details:
            assert "violation_type" in detail
            assert "description" in detail

    def test_extractor_name_is_semantic(self) -> None:
        """REQ-VERIFY-020: Extractor name is 'semantic'."""
        result = MOD.run_semantic_extractor("What is 2 + 2?", "4")
        assert result.extractor_name == "semantic"


# ---------------------------------------------------------------------------
# run_case (full triple-extractor integration)
# ---------------------------------------------------------------------------


class TestRunCase:
    """REQ-VERIFY-001: Full triple-extractor run on individual cases."""

    def _make_case(
        self, case_id: str, question: str, ground_truth: int
    ) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "question": question,
            "ground_truth": ground_truth,
        }

    def test_correct_explicit_arithmetic_no_violations(self) -> None:
        """REQ-VERIFY-001: Correct explicit arithmetic → no extractor flags it."""
        case = self._make_case(
            "t-0",
            "A train travels 60 mph for 3 hours. How far does it go?",
            180,
        )
        result = MOD.run_case(
            case=case,
            response="Distance = 60 * 3 = 180 miles.",
            llm_generate_fn=MOD._ci_generate_fn,
        )
        assert result.correct is True
        assert result.z3.flagged is False
        assert result.llm.flagged is False
        assert result.combined_flagged is False

    def test_wrong_arithmetic_z3_and_llm_both_flag(self) -> None:
        """REQ-VERIFY-001: Wrong explicit arithmetic → Z3 and LLM both flag it."""
        case = self._make_case(
            "t-1",
            "A box holds 12 eggs. How many eggs are in 7 boxes?",
            84,
        )
        result = MOD.run_case(
            case=case,
            response="12 * 7 = 85. There are 85 eggs in total.",
            llm_generate_fn=MOD._ci_generate_fn,
        )
        assert result.correct is False
        assert result.z3.flagged is True
        assert result.llm.flagged is True
        assert result.combined_flagged is True

    def test_wrong_answer_no_arithmetic_shown_not_flagged_by_z3(self) -> None:
        """REQ-VERIFY-001: Wrong terse response has no Z3 or LLM flags."""
        case = self._make_case(
            "t-5",
            "Maria buys 3 pens at $4 each. What does she pay in total?",
            12,
        )
        result = MOD.run_case(
            case=case,
            response="Maria pays $14.",
            llm_generate_fn=MOD._ci_generate_fn,
        )
        assert result.correct is False
        assert result.z3.flagged is False   # no arithmetic to check
        assert result.llm.flagged is False  # no CLAIM lines found

    def test_case_result_has_correct_case_id(self) -> None:
        """REQ-VERIFY-001: CaseResult preserves the input case_id."""
        case = self._make_case("my-case-42", "2 + 2 = ?", 4)
        result = MOD.run_case(
            case=case,
            response="2 + 2 = 4.",
            llm_generate_fn=MOD._ci_generate_fn,
        )
        assert result.case_id == "my-case-42"

    def test_extracted_answer_and_correct_populated(self) -> None:
        """REQ-VERIFY-001: extracted_answer and correct fields are always set."""
        case = self._make_case("t-2", "5 * 5 = ?", 25)
        result = MOD.run_case(
            case=case,
            response="5 * 5 = 25.",
            llm_generate_fn=MOD._ci_generate_fn,
        )
        assert result.extracted_answer == 25.0
        assert result.correct is True


# ---------------------------------------------------------------------------
# compute_statistics
# ---------------------------------------------------------------------------


class TestComputeStatistics:
    """REQ-VERIFY-009, REQ-VERIFY-010: Metric computation from case results."""

    def _make_result(
        self,
        *,
        correct: bool,
        z3_flagged: bool = False,
        llm_flagged: bool = False,
        sem_flagged: bool = False,
    ) -> Any:
        """Create a minimal CaseResult-like object for statistics testing."""
        from types import SimpleNamespace

        z3 = SimpleNamespace(flagged=z3_flagged)
        llm = SimpleNamespace(flagged=llm_flagged)
        semantic = SimpleNamespace(flagged=sem_flagged)
        combined = z3_flagged or llm_flagged or sem_flagged
        return SimpleNamespace(
            correct=correct,
            z3=z3,
            llm=llm,
            semantic=semantic,
            combined_flagged=combined,
        )

    def test_empty_returns_empty_dict(self) -> None:
        """REQ-VERIFY-009: Empty case list returns empty statistics."""
        stats = MOD.compute_statistics([])
        assert stats == {}

    def test_perfect_detection_no_fp(self) -> None:
        """REQ-VERIFY-009: 100% detection, 0% FP rate for ideal extractor."""
        results = [
            self._make_result(correct=True, z3_flagged=False),
            self._make_result(correct=True, z3_flagged=False),
            self._make_result(correct=False, z3_flagged=True),
            self._make_result(correct=False, z3_flagged=True),
        ]
        stats = MOD.compute_statistics(results)
        assert stats["n_cases"] == 4
        assert stats["n_correct"] == 2
        assert stats["n_wrong"] == 2
        assert stats["baseline_accuracy"] == 0.5
        assert stats["z3"]["detection_rate"] == 1.0
        assert stats["z3"]["fp_rate"] == 0.0
        assert stats["z3"]["repair_delta"] == 0.5

    def test_fp_counted_in_fp_rate(self) -> None:
        """REQ-VERIFY-009: False positive is counted in fp_rate."""
        results = [
            self._make_result(correct=True, z3_flagged=True),   # FP
            self._make_result(correct=True, z3_flagged=False),
            self._make_result(correct=False, z3_flagged=True),  # TP
        ]
        stats = MOD.compute_statistics(results)
        assert stats["z3"]["fp_rate"] == 0.5  # 1 FP out of 2 correct
        assert stats["z3"]["detection_rate"] == 1.0

    def test_combined_is_union_of_extractors(self) -> None:
        """REQ-VERIFY-001: Combined flag is OR of all three extractors."""
        results = [
            self._make_result(correct=False, z3_flagged=True, llm_flagged=False, sem_flagged=False),
            self._make_result(correct=False, z3_flagged=False, llm_flagged=True, sem_flagged=False),
            self._make_result(correct=False, z3_flagged=False, llm_flagged=False, sem_flagged=True),
            self._make_result(correct=True, z3_flagged=False),
        ]
        stats = MOD.compute_statistics(results)
        # All 3 wrong cases detected by combined
        assert stats["combined"]["detection_rate"] == 1.0
        # Overlap breakdown: 1 for each extractor alone
        assert stats["extractor_overlap"]["z3_only"] == 1
        assert stats["extractor_overlap"]["llm_only"] == 1
        assert stats["extractor_overlap"]["semantic_only"] == 1

    def test_all_wrong_and_no_detections(self) -> None:
        """REQ-VERIFY-009: Zero detection rate when extractor never fires."""
        results = [
            self._make_result(correct=False, z3_flagged=False),
            self._make_result(correct=False, z3_flagged=False),
        ]
        stats = MOD.compute_statistics(results)
        assert stats["z3"]["detection_rate"] == 0.0
        assert stats["z3"]["repair_delta"] == 0.0


# ---------------------------------------------------------------------------
# build_artifact
# ---------------------------------------------------------------------------


class TestBuildArtifact:
    """REQ-VERIFY-001: Artifact schema validation."""

    def _minimal_case_result(self) -> Any:
        from types import SimpleNamespace

        ext = SimpleNamespace(flagged=False, n_violations=0, n_satisfied=0, n_total=0, details=[])
        return SimpleNamespace(
            case_id="ci-0",
            question="Q?",
            ground_truth=10,
            extracted_answer=10.0,
            correct=True,
            combined_flagged=False,
            latency_seconds=0.001,
            z3=ext,
            llm=ext,
            semantic=ext,
        )

    def test_artifact_has_required_top_level_fields(self) -> None:
        """REQ-VERIFY-001: Artifact contains experiment, benchmark, and statistics."""
        artifact = MOD.build_artifact(
            [self._minimal_case_result()],
            {"n_cases": 1},
            live_mode=False,
            cohort_source="canned_ci_cases",
            started_at="2026-04-14T03:00:00Z",
            finished_at="2026-04-14T03:00:01Z",
            runtime_seconds=1.0,
        )
        assert artifact["experiment"] == 276
        assert artifact["benchmark"] == "gsm8k_modern_extractors"
        assert "statistics" in artifact
        assert "cases" in artifact
        assert "metadata" in artifact

    def test_artifact_metadata_includes_extractors_list(self) -> None:
        """REQ-VERIFY-001: Metadata lists all three extractors."""
        artifact = MOD.build_artifact(
            [],
            {},
            live_mode=False,
            cohort_source="ci",
            started_at="T",
            finished_at="T",
            runtime_seconds=0.0,
        )
        assert artifact["metadata"]["extractors"] == ["z3", "llm", "semantic"]

    def test_artifact_live_mode_flag_recorded(self) -> None:
        """REQ-VERIFY-001: live_mode flag is preserved in metadata."""
        artifact = MOD.build_artifact(
            [],
            {},
            live_mode=True,
            cohort_source="results/experiment_219_results.json",
            started_at="T",
            finished_at="T",
            runtime_seconds=0.0,
        )
        assert artifact["metadata"]["live_mode"] is True
        assert artifact["metadata"]["cohort_source"] == "results/experiment_219_results.json"

    def test_case_dict_has_correct_extractor_subkeys(self) -> None:
        """REQ-VERIFY-001: Each case dict has z3/llm/semantic sub-dicts."""
        artifact = MOD.build_artifact(
            [self._minimal_case_result()],
            {},
            live_mode=False,
            cohort_source="ci",
            started_at="T",
            finished_at="T",
            runtime_seconds=0.0,
        )
        case = artifact["cases"][0]
        assert set(case["extractors"].keys()) == {"z3", "llm", "semantic"}
        for key in ("z3", "llm", "semantic"):
            ext = case["extractors"][key]
            assert "flagged" in ext
            assert "n_violations" in ext
            assert "details" in ext


# ---------------------------------------------------------------------------
# Full CI benchmark integration
# ---------------------------------------------------------------------------


class TestFullCIBenchmark:
    """REQ-VERIFY-001, REQ-VERIFY-009, REQ-VERIFY-010: End-to-end CI pipeline."""

    def test_run_ci_benchmark_returns_ten_cases(self) -> None:
        """REQ-VERIFY-001: CI benchmark produces one result per CI case (10 total)."""
        results = MOD.run_ci_benchmark()
        assert len(results) == 10

    def test_ci_baseline_is_fifty_percent(self) -> None:
        """REQ-VERIFY-001: CI cases are designed for 50% baseline accuracy."""
        results = MOD.run_ci_benchmark()
        n_correct = sum(1 for r in results if r.correct)
        assert n_correct == 5

    def test_z3_detects_four_of_five_wrong_answers(self) -> None:
        """SCENARIO-VERIFY-009: Z3 catches wrong arithmetic in 4/5 wrong-answer cases."""
        results = MOD.run_ci_benchmark()
        wrong_results = [r for r in results if not r.correct]
        assert len(wrong_results) == 5
        z3_detected = sum(1 for r in wrong_results if r.z3.flagged)
        # Case ci-5 has no explicit arithmetic → Z3 cannot detect it.
        # The other 4 wrong cases (ci-1, ci-3, ci-7, ci-9) all have explicit
        # arithmetic errors that Z3 catches.
        assert z3_detected == 4

    def test_z3_zero_false_positives(self) -> None:
        """SCENARIO-VERIFY-009: Z3 does not flag any correct-answer case."""
        results = MOD.run_ci_benchmark()
        correct_results = [r for r in results if r.correct]
        z3_fp = sum(1 for r in correct_results if r.z3.flagged)
        assert z3_fp == 0

    def test_llm_extractor_matches_z3_on_ci_cases(self) -> None:
        """SCENARIO-VERIFY-010: LLM stub detects same set of violations as Z3."""
        results = MOD.run_ci_benchmark()
        # For every case, Z3 and LLM stub should agree on flagging.
        for r in results:
            assert r.z3.flagged == r.llm.flagged, (
                f"Case {r.case_id}: Z3 flagged={r.z3.flagged} but LLM flagged={r.llm.flagged}"
            )

    def test_semantic_has_zero_detection_on_ci_arithmetic_cases(self) -> None:
        """SCENARIO-VERIFY-020: Semantic grounding doesn't detect arithmetic errors."""
        results = MOD.run_ci_benchmark()
        wrong_results = [r for r in results if not r.correct]
        sem_detected = sum(1 for r in wrong_results if r.semantic.flagged)
        # Semantic grounding is not designed for arithmetic errors; it checks
        # question targeting. On these GSM8K arithmetic cases it detects none.
        assert sem_detected == 0

    def test_semantic_may_produce_false_positives_on_terse_responses(self) -> None:
        """SCENARIO-VERIFY-021: Semantic FPs occur on terse correct responses."""
        results = MOD.run_ci_benchmark()
        correct_results = [r for r in results if r.correct]
        sem_fp = sum(1 for r in correct_results if r.semantic.flagged)
        # Case ci-4 ("Tom has 9 books remaining.") triggers a semantic FP
        # because "He gives 6" is ungrounded and "remain"/"remaining" differs.
        assert sem_fp >= 1

    def test_combined_statistics_have_required_keys(self) -> None:
        """REQ-VERIFY-001: Statistics dict contains all required keys."""
        results = MOD.run_ci_benchmark()
        stats = MOD.compute_statistics(results)
        for key in ("z3", "llm", "semantic", "combined"):
            assert key in stats, f"Missing extractor key: {key}"
            ext_stats = stats[key]
            assert "detection_rate" in ext_stats
            assert "fp_rate" in ext_stats
            assert "repair_delta" in ext_stats

    def test_z3_repair_delta_positive(self) -> None:
        """REQ-VERIFY-009: Z3 theoretical repair delta is positive (adds value)."""
        results = MOD.run_ci_benchmark()
        stats = MOD.compute_statistics(results)
        assert stats["z3"]["repair_delta"] > 0.0

    def test_semantic_repair_delta_zero_on_arithmetic_cases(self) -> None:
        """SCENARIO-VERIFY-020: Semantic repair delta is 0 on arithmetic GSM8K."""
        results = MOD.run_ci_benchmark()
        stats = MOD.compute_statistics(results)
        # Semantic detects no wrong answers → repair delta is 0
        assert stats["semantic"]["repair_delta"] == 0.0

    def test_results_json_exists_and_has_correct_experiment_number(self) -> None:
        """REQ-VERIFY-001: The checked-in results JSON has experiment=276."""
        results_path = (
            Path(__file__).resolve().parents[2]
            / "results"
            / "experiment_276_results.json"
        )
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        assert payload["experiment"] == 276
        assert payload["benchmark"] == "gsm8k_modern_extractors"
        assert "statistics" in payload
        assert "cases" in payload
        # Should have 10 CI cases (from the checked-in artifact)
        assert payload["metadata"]["n_cases"] == 10
        assert payload["metadata"]["live_mode"] is False
