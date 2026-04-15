"""Tests for python/carnot/pipeline/adversarial_gsm8k.py.

Covers 100% of the module:
- DISTRACTOR_SENTENCES: pool size, uniqueness, string types
- AdversarialGSMQuestion: construction, asdict, field presence
- build_adversarial_questions: basic round-trip, seed reproducibility,
  adversarial_question contains original, distractor appended, empty input,
  custom question_id, auto-generated question_id format
- AdversarialBenchmarkResult: construction, defaults, asdict
- compute_adversarial_results: basic computation, zero-length input,
  mismatched lengths raise ValueError, negative accuracy_drop (no clamping),
  negative repair_improvement (no clamping), inference_mode passthrough
- SYNTHETIC_CI_RESULTS: field values match docstring, inference_mode="simulated"
- build_adversarial_artifact:
  - schema field
  - inference_mode="simulated" -> honest_verdict="blocked_simulated"
  - live_gpu + repair_improvement > 0 -> honest_verdict="improvement_positive"
  - live_gpu + repair_improvement <= 0 + accuracy_drop > 0 -> "degradation_positive"
  - live_gpu + repair_improvement <= 0 + accuracy_drop <= 0 -> "neutral"
  - robustness_invariant_holds True/False boundary
  - headline_result keys present
  - all scalar fields echoed into artifact

Spec: REQ-BENCH-006, REQ-BENCH-007,
      SCENARIO-BENCH-014, SCENARIO-BENCH-015, SCENARIO-BENCH-016
"""

from __future__ import annotations

import dataclasses

import pytest

from carnot.pipeline.adversarial_gsm8k import (
    DISTRACTOR_SENTENCES,
    SYNTHETIC_CI_RESULTS,
    AdversarialBenchmarkResult,
    AdversarialGSMQuestion,
    build_adversarial_artifact,
    build_adversarial_questions,
    compute_adversarial_results,
)


# ---------------------------------------------------------------------------
# DISTRACTOR_SENTENCES
# ---------------------------------------------------------------------------


class TestDistractorSentences:
    """REQ-BENCH-006: DISTRACTOR_SENTENCES pool."""

    def test_exactly_20_sentences(self):
        """SCENARIO-BENCH-014: pool must have exactly 20 entries."""
        assert len(DISTRACTOR_SENTENCES) == 20

    def test_all_strings(self):
        """Every entry is a non-empty string."""
        for s in DISTRACTOR_SENTENCES:
            assert isinstance(s, str)
            assert len(s) > 0

    def test_all_unique(self):
        """No duplicate distractor sentences."""
        assert len(set(DISTRACTOR_SENTENCES)) == 20


# ---------------------------------------------------------------------------
# AdversarialGSMQuestion
# ---------------------------------------------------------------------------


class TestAdversarialGSMQuestion:
    """REQ-BENCH-006, SCENARIO-BENCH-014: AdversarialGSMQuestion dataclass."""

    def _make(self, **overrides) -> AdversarialGSMQuestion:
        defaults = dict(
            question_id="q_0001",
            original_question="There are 5 apples. If 2 are eaten, how many remain?",
            adversarial_question=(
                "There are 5 apples. If 2 are eaten, how many remain? "
                "The weather was sunny that day."
            ),
            ground_truth_answer="3",
            irrelevant_sentence="The weather was sunny that day.",
        )
        defaults.update(overrides)
        return AdversarialGSMQuestion(**defaults)

    def test_construction_with_all_fields(self):
        """SCENARIO-BENCH-014: dataclass constructs with all five required fields."""
        q = self._make()
        assert q.question_id == "q_0001"
        assert "5 apples" in q.original_question
        assert "sunny" in q.adversarial_question
        assert q.ground_truth_answer == "3"
        assert q.irrelevant_sentence == "The weather was sunny that day."

    def test_asdict_has_all_keys(self):
        """SCENARIO-BENCH-014: asdict() produces dict with exactly the five spec keys."""
        q = self._make()
        d = dataclasses.asdict(q)
        assert set(d.keys()) == {
            "question_id",
            "original_question",
            "adversarial_question",
            "ground_truth_answer",
            "irrelevant_sentence",
        }

    def test_adversarial_contains_original(self):
        """adversarial_question contains original_question as substring."""
        q = self._make()
        assert q.original_question in q.adversarial_question

    def test_irrelevant_sentence_in_adversarial(self):
        """The irrelevant_sentence appears at the end of adversarial_question."""
        q = self._make()
        assert q.adversarial_question.endswith(q.irrelevant_sentence)

    def test_equality(self):
        """Two identical AdversarialGSMQuestion instances are equal (dataclass default)."""
        q1 = self._make()
        q2 = self._make()
        assert q1 == q2

    def test_question_id_can_be_any_string(self):
        """question_id accepts arbitrary string values."""
        q = self._make(question_id="gsm8k_train_042")
        assert q.question_id == "gsm8k_train_042"


# ---------------------------------------------------------------------------
# build_adversarial_questions
# ---------------------------------------------------------------------------


class TestBuildAdversarialQuestions:
    """REQ-BENCH-006, SCENARIO-BENCH-014: build_adversarial_questions."""

    def _sample_inputs(self, n: int = 5) -> list[dict[str, str]]:
        return [
            {"question": f"What is {i} + {i}?", "answer": str(i * 2)}
            for i in range(n)
        ]

    def test_returns_correct_count(self):
        """SCENARIO-BENCH-014: output list has same length as input."""
        questions = self._sample_inputs(5)
        result = build_adversarial_questions(questions)
        assert len(result) == 5

    def test_adversarial_contains_original(self):
        """Each adversarial_question contains the original question text."""
        questions = self._sample_inputs(3)
        result = build_adversarial_questions(questions)
        for aq in result:
            assert aq.original_question in aq.adversarial_question

    def test_distractor_appended_with_space(self):
        """Distractor is appended with a single space separator."""
        questions = self._sample_inputs(3)
        result = build_adversarial_questions(questions)
        for aq in result:
            expected = f"{aq.original_question} {aq.irrelevant_sentence}"
            assert aq.adversarial_question == expected

    def test_distractor_from_pool(self):
        """Every appended distractor comes from DISTRACTOR_SENTENCES."""
        questions = self._sample_inputs(20)
        result = build_adversarial_questions(questions)
        for aq in result:
            assert aq.irrelevant_sentence in DISTRACTOR_SENTENCES

    def test_seed_reproducibility(self):
        """SCENARIO-BENCH-014: same seed always produces the same distractor assignment."""
        questions = self._sample_inputs(10)
        r1 = build_adversarial_questions(questions, seed=42)
        r2 = build_adversarial_questions(questions, seed=42)
        assert [q.irrelevant_sentence for q in r1] == [q.irrelevant_sentence for q in r2]

    def test_different_seed_different_result(self):
        """Different seeds produce different distractor assignments (probabilistically)."""
        questions = self._sample_inputs(20)
        r1 = build_adversarial_questions(questions, seed=1)
        r2 = build_adversarial_questions(questions, seed=99)
        # With 20 questions and 20 distractors, seed difference almost certainly changes assignment
        distractors1 = [q.irrelevant_sentence for q in r1]
        distractors2 = [q.irrelevant_sentence for q in r2]
        assert distractors1 != distractors2

    def test_auto_generated_question_id(self):
        """When no question_id key present, IDs are auto-generated as q_NNNN."""
        questions = self._sample_inputs(3)
        result = build_adversarial_questions(questions)
        assert result[0].question_id == "q_0000"
        assert result[1].question_id == "q_0001"
        assert result[2].question_id == "q_0002"

    def test_custom_question_id_used(self):
        """If question_id key is present in input dict, it is used as-is."""
        questions = [
            {"question_id": "gsm_042", "question": "What is 6 * 7?", "answer": "42"},
        ]
        result = build_adversarial_questions(questions)
        assert result[0].question_id == "gsm_042"

    def test_ground_truth_answer_preserved(self):
        """ground_truth_answer is taken from the input dict's 'answer' field."""
        questions = [{"question": "What is 3 + 3?", "answer": "6"}]
        result = build_adversarial_questions(questions)
        assert result[0].ground_truth_answer == "6"

    def test_empty_input_returns_empty_list(self):
        """Empty input list returns an empty output list."""
        result = build_adversarial_questions([])
        assert result == []

    def test_returns_list_of_adversarial_questions(self):
        """Each element of the output is an AdversarialGSMQuestion instance."""
        questions = self._sample_inputs(3)
        result = build_adversarial_questions(questions)
        for item in result:
            assert isinstance(item, AdversarialGSMQuestion)

    def test_single_question(self):
        """Single-element input produces a single-element output."""
        questions = [{"question": "How many legs does a spider have?", "answer": "8"}]
        result = build_adversarial_questions(questions)
        assert len(result) == 1
        assert result[0].ground_truth_answer == "8"

    def test_default_seed_is_42(self):
        """Default seed is 42 — omitting seed argument gives same result as seed=42."""
        questions = self._sample_inputs(5)
        r_default = build_adversarial_questions(questions)
        r_explicit = build_adversarial_questions(questions, seed=42)
        assert [q.irrelevant_sentence for q in r_default] == [
            q.irrelevant_sentence for q in r_explicit
        ]


# ---------------------------------------------------------------------------
# AdversarialBenchmarkResult
# ---------------------------------------------------------------------------


class TestAdversarialBenchmarkResult:
    """REQ-BENCH-006, SCENARIO-BENCH-015: AdversarialBenchmarkResult dataclass."""

    def _make(self, **overrides) -> AdversarialBenchmarkResult:
        defaults = dict(
            standard_accuracy=0.80,
            adversarial_accuracy=0.65,
            accuracy_drop=0.15,
            repaired_adversarial_accuracy=0.68,
            repair_improvement=0.03,
            inference_mode="simulated",
        )
        defaults.update(overrides)
        return AdversarialBenchmarkResult(**defaults)

    def test_construction_all_fields(self):
        """SCENARIO-BENCH-015: dataclass accepts all six fields."""
        r = self._make()
        assert r.standard_accuracy == 0.80
        assert r.adversarial_accuracy == 0.65
        assert r.accuracy_drop == 0.15
        assert r.repaired_adversarial_accuracy == 0.68
        assert r.repair_improvement == 0.03
        assert r.inference_mode == "simulated"

    def test_asdict_has_all_keys(self):
        """SCENARIO-BENCH-015: asdict() produces dict with all six keys."""
        r = self._make()
        d = dataclasses.asdict(r)
        assert set(d.keys()) == {
            "standard_accuracy",
            "adversarial_accuracy",
            "accuracy_drop",
            "repaired_adversarial_accuracy",
            "repair_improvement",
            "inference_mode",
        }

    def test_negative_accuracy_drop_stored(self):
        """Negative accuracy_drop is stored without clamping (honest research finding)."""
        r = self._make(accuracy_drop=-0.05)
        assert r.accuracy_drop == -0.05

    def test_negative_repair_improvement_stored(self):
        """Negative repair_improvement is stored without clamping."""
        r = self._make(repair_improvement=-0.02)
        assert r.repair_improvement == -0.02

    def test_live_gpu_mode(self):
        """inference_mode='live_gpu' is accepted."""
        r = self._make(inference_mode="live_gpu")
        assert r.inference_mode == "live_gpu"


# ---------------------------------------------------------------------------
# compute_adversarial_results
# ---------------------------------------------------------------------------


class TestComputeAdversarialResults:
    """REQ-BENCH-006, SCENARIO-BENCH-015: compute_adversarial_results."""

    def test_basic_computation(self):
        """SCENARIO-BENCH-015: 8/10, 6/10, 7/10 correct."""
        std = [True] * 8 + [False] * 2
        adv = [True] * 6 + [False] * 4
        rep = [True] * 7 + [False] * 3
        result = compute_adversarial_results(std, adv, rep)
        assert result.standard_accuracy == pytest.approx(0.80)
        assert result.adversarial_accuracy == pytest.approx(0.60)
        assert result.accuracy_drop == pytest.approx(0.20)
        assert result.repaired_adversarial_accuracy == pytest.approx(0.70)
        assert result.repair_improvement == pytest.approx(0.10)

    def test_returns_adversarial_benchmark_result(self):
        """Return type is AdversarialBenchmarkResult."""
        result = compute_adversarial_results(
            [True, False], [True, False], [True, False]
        )
        assert isinstance(result, AdversarialBenchmarkResult)

    def test_inference_mode_passthrough(self):
        """inference_mode parameter is passed through to the result."""
        result = compute_adversarial_results(
            [True], [True], [True], inference_mode="live_gpu"
        )
        assert result.inference_mode == "live_gpu"

    def test_default_inference_mode_simulated(self):
        """Default inference_mode is 'simulated'."""
        result = compute_adversarial_results([True], [True], [True])
        assert result.inference_mode == "simulated"

    def test_empty_lists_return_zeros(self):
        """Empty lists return a result with all zeros."""
        result = compute_adversarial_results([], [], [])
        assert result.standard_accuracy == 0.0
        assert result.adversarial_accuracy == 0.0
        assert result.accuracy_drop == 0.0
        assert result.repaired_adversarial_accuracy == 0.0
        assert result.repair_improvement == 0.0

    def test_mismatched_standard_adversarial_raises(self):
        """ValueError raised when standard and adversarial lists differ in length."""
        with pytest.raises(ValueError, match="equal lengths"):
            compute_adversarial_results([True, False], [True], [True, False])

    def test_mismatched_repaired_raises(self):
        """ValueError raised when repaired list differs in length from others."""
        with pytest.raises(ValueError, match="equal lengths"):
            compute_adversarial_results([True, False], [True, False], [True])

    def test_all_correct(self):
        """All True inputs give accuracy=1.0 and no drop."""
        result = compute_adversarial_results(
            [True] * 5, [True] * 5, [True] * 5
        )
        assert result.standard_accuracy == 1.0
        assert result.adversarial_accuracy == 1.0
        assert result.accuracy_drop == pytest.approx(0.0)
        assert result.repair_improvement == pytest.approx(0.0)

    def test_all_incorrect(self):
        """All False inputs give accuracy=0.0."""
        result = compute_adversarial_results(
            [False] * 4, [False] * 4, [False] * 4
        )
        assert result.standard_accuracy == 0.0
        assert result.adversarial_accuracy == 0.0

    def test_negative_accuracy_drop_no_clamping(self):
        """Negative accuracy_drop is preserved (adversarial somehow better than standard)."""
        std = [True] * 6 + [False] * 4   # 0.60
        adv = [True] * 8 + [False] * 2   # 0.80
        rep = [True] * 8 + [False] * 2   # 0.80
        result = compute_adversarial_results(std, adv, rep)
        # accuracy_drop = 0.60 - 0.80 = -0.20
        assert result.accuracy_drop == pytest.approx(-0.20)

    def test_negative_repair_improvement_no_clamping(self):
        """Negative repair_improvement is preserved (repair made things worse)."""
        std = [True] * 5 + [False] * 5   # 0.50
        adv = [True] * 6 + [False] * 4   # 0.60
        rep = [True] * 4 + [False] * 6   # 0.40
        result = compute_adversarial_results(std, adv, rep)
        # repair_improvement = 0.40 - 0.60 = -0.20
        assert result.repair_improvement == pytest.approx(-0.20)

    def test_single_question_correct(self):
        """Single question, all correct."""
        result = compute_adversarial_results([True], [True], [True])
        assert result.standard_accuracy == 1.0
        assert result.adversarial_accuracy == 1.0
        assert result.repaired_adversarial_accuracy == 1.0

    def test_single_question_incorrect(self):
        """Single question, all incorrect."""
        result = compute_adversarial_results([False], [False], [False])
        assert result.standard_accuracy == 0.0


# ---------------------------------------------------------------------------
# SYNTHETIC_CI_RESULTS
# ---------------------------------------------------------------------------


class TestSyntheticCIResults:
    """REQ-BENCH-006, SCENARIO-BENCH-015: SYNTHETIC_CI_RESULTS sentinel values."""

    def test_standard_accuracy(self):
        """standard_accuracy is 0.80 as documented."""
        assert SYNTHETIC_CI_RESULTS.standard_accuracy == 0.80

    def test_adversarial_accuracy(self):
        """adversarial_accuracy is 0.65 as documented."""
        assert SYNTHETIC_CI_RESULTS.adversarial_accuracy == 0.65

    def test_repaired_accuracy(self):
        """repaired_adversarial_accuracy is 0.68 as documented."""
        assert SYNTHETIC_CI_RESULTS.repaired_adversarial_accuracy == 0.68

    def test_inference_mode_simulated(self):
        """inference_mode is 'simulated' — never live provenance."""
        assert SYNTHETIC_CI_RESULTS.inference_mode == "simulated"

    def test_is_adversarial_benchmark_result(self):
        """SYNTHETIC_CI_RESULTS is an AdversarialBenchmarkResult instance."""
        assert isinstance(SYNTHETIC_CI_RESULTS, AdversarialBenchmarkResult)

    def test_accuracy_drop_consistent(self):
        """accuracy_drop matches standard - adversarial."""
        expected = SYNTHETIC_CI_RESULTS.standard_accuracy - SYNTHETIC_CI_RESULTS.adversarial_accuracy
        assert SYNTHETIC_CI_RESULTS.accuracy_drop == pytest.approx(expected)

    def test_repair_improvement_consistent(self):
        """repair_improvement matches repaired - adversarial."""
        expected = (
            SYNTHETIC_CI_RESULTS.repaired_adversarial_accuracy
            - SYNTHETIC_CI_RESULTS.adversarial_accuracy
        )
        assert SYNTHETIC_CI_RESULTS.repair_improvement == pytest.approx(expected)


# ---------------------------------------------------------------------------
# build_adversarial_artifact
# ---------------------------------------------------------------------------


def _make_result(
    *,
    standard_accuracy: float = 0.80,
    adversarial_accuracy: float = 0.70,
    repaired_adversarial_accuracy: float = 0.75,
    accuracy_drop: float | None = None,
    repair_improvement: float | None = None,
    inference_mode: str = "live_gpu",
) -> AdversarialBenchmarkResult:
    if accuracy_drop is None:
        accuracy_drop = standard_accuracy - adversarial_accuracy
    if repair_improvement is None:
        repair_improvement = repaired_adversarial_accuracy - adversarial_accuracy
    return AdversarialBenchmarkResult(
        standard_accuracy=standard_accuracy,
        adversarial_accuracy=adversarial_accuracy,
        accuracy_drop=accuracy_drop,
        repaired_adversarial_accuracy=repaired_adversarial_accuracy,
        repair_improvement=repair_improvement,
        inference_mode=inference_mode,
    )


class TestBuildAdversarialArtifact:
    """REQ-BENCH-006, REQ-BENCH-007, SCENARIO-BENCH-016: build_adversarial_artifact."""

    def test_schema_field(self):
        """SCENARIO-BENCH-016: schema must be 'carnot.adversarial_gsm8k.v1'."""
        artifact = build_adversarial_artifact(_make_result())
        assert artifact["schema"] == "carnot.adversarial_gsm8k.v1"

    def test_simulated_mode_blocked_verdict(self):
        """SCENARIO-BENCH-016: inference_mode='simulated' -> honest_verdict='blocked_simulated'."""
        result = _make_result(inference_mode="simulated")
        artifact = build_adversarial_artifact(result)
        assert artifact["honest_verdict"] == "blocked_simulated"

    def test_live_gpu_positive_repair_improvement_verdict(self):
        """SCENARIO-BENCH-016: live_gpu + repair_improvement > 0 -> 'improvement_positive'."""
        result = _make_result(
            inference_mode="live_gpu",
            adversarial_accuracy=0.70,
            repaired_adversarial_accuracy=0.75,
        )
        artifact = build_adversarial_artifact(result)
        assert artifact["honest_verdict"] == "improvement_positive"

    def test_live_gpu_zero_repair_improvement_accuracy_drop_degradation(self):
        """SCENARIO-BENCH-016: live_gpu + repair_improvement=0 + accuracy_drop>0 -> 'degradation_positive'."""
        result = _make_result(
            inference_mode="live_gpu",
            standard_accuracy=0.80,
            adversarial_accuracy=0.70,
            repaired_adversarial_accuracy=0.70,
            repair_improvement=0.0,
            accuracy_drop=0.10,
        )
        artifact = build_adversarial_artifact(result)
        assert artifact["honest_verdict"] == "degradation_positive"

    def test_live_gpu_negative_repair_improvement_with_drop_degradation(self):
        """live_gpu + repair_improvement < 0 + accuracy_drop > 0 -> 'degradation_positive'."""
        result = _make_result(
            inference_mode="live_gpu",
            standard_accuracy=0.80,
            adversarial_accuracy=0.70,
            repaired_adversarial_accuracy=0.65,
            repair_improvement=-0.05,
            accuracy_drop=0.10,
        )
        artifact = build_adversarial_artifact(result)
        assert artifact["honest_verdict"] == "degradation_positive"

    def test_live_gpu_no_drop_neutral_verdict(self):
        """live_gpu + repair_improvement <= 0 + accuracy_drop <= 0 -> 'neutral'."""
        result = _make_result(
            inference_mode="live_gpu",
            standard_accuracy=0.80,
            adversarial_accuracy=0.82,   # adversarial BETTER than standard
            repaired_adversarial_accuracy=0.82,
            accuracy_drop=-0.02,
            repair_improvement=0.0,
        )
        artifact = build_adversarial_artifact(result)
        assert artifact["honest_verdict"] == "neutral"

    def test_robustness_invariant_true_when_drop_within_tolerance(self):
        """REQ-BENCH-007: robustness_invariant_holds=True when drop <= 0.05."""
        result = _make_result(
            standard_accuracy=0.80,
            adversarial_accuracy=0.76,   # drop = 0.04, within 0.05 tolerance
        )
        artifact = build_adversarial_artifact(result)
        assert artifact["robustness_invariant_holds"] is True

    def test_robustness_invariant_true_at_exact_tolerance(self):
        """robustness_invariant_holds=True when drop equals exactly 0.05."""
        result = _make_result(
            standard_accuracy=0.80,
            adversarial_accuracy=0.75,   # drop = exactly 0.05
        )
        artifact = build_adversarial_artifact(result)
        assert artifact["robustness_invariant_holds"] is True

    def test_robustness_invariant_false_when_drop_exceeds_tolerance(self):
        """REQ-BENCH-007: robustness_invariant_holds=False when drop > 0.05."""
        result = _make_result(
            standard_accuracy=0.80,
            adversarial_accuracy=0.74,   # drop = 0.06, exceeds 0.05 tolerance
        )
        artifact = build_adversarial_artifact(result)
        assert artifact["robustness_invariant_holds"] is False

    def test_robustness_invariant_true_when_adversarial_better(self):
        """robustness_invariant_holds=True when adversarial_accuracy > standard_accuracy."""
        result = _make_result(
            standard_accuracy=0.75,
            adversarial_accuracy=0.80,   # adversarial better (drop = -0.05)
        )
        artifact = build_adversarial_artifact(result)
        assert artifact["robustness_invariant_holds"] is True

    def test_headline_result_keys(self):
        """SCENARIO-BENCH-016: headline_result contains all expected keys."""
        artifact = build_adversarial_artifact(_make_result())
        hr = artifact["headline_result"]
        required_keys = {
            "standard_accuracy",
            "adversarial_accuracy",
            "accuracy_drop",
            "repair_improvement",
            "robustness_invariant_holds",
            "inference_mode",
        }
        assert required_keys.issubset(set(hr.keys()))

    def test_scalar_fields_echoed(self):
        """All scalar accuracy fields appear at the top level of the artifact."""
        result = _make_result(
            standard_accuracy=0.82,
            adversarial_accuracy=0.79,
            repaired_adversarial_accuracy=0.81,
        )
        artifact = build_adversarial_artifact(result)
        assert artifact["standard_accuracy"] == pytest.approx(0.82)
        assert artifact["adversarial_accuracy"] == pytest.approx(0.79)
        assert artifact["repaired_adversarial_accuracy"] == pytest.approx(0.81)

    def test_inference_mode_in_artifact(self):
        """inference_mode field appears at the top level of the artifact."""
        artifact = build_adversarial_artifact(_make_result(inference_mode="live_gpu"))
        assert artifact["inference_mode"] == "live_gpu"

    def test_accuracy_drop_in_artifact(self):
        """accuracy_drop appears at the top level."""
        result = _make_result(accuracy_drop=0.08)
        artifact = build_adversarial_artifact(result)
        assert artifact["accuracy_drop"] == pytest.approx(0.08)

    def test_repair_improvement_in_artifact(self):
        """repair_improvement appears at the top level."""
        result = _make_result(repair_improvement=0.04)
        artifact = build_adversarial_artifact(result)
        assert artifact["repair_improvement"] == pytest.approx(0.04)

    def test_simulated_synthetic_ci_results(self):
        """SYNTHETIC_CI_RESULTS produces blocked_simulated verdict via build_adversarial_artifact."""
        artifact = build_adversarial_artifact(SYNTHETIC_CI_RESULTS)
        assert artifact["honest_verdict"] == "blocked_simulated"
        assert artifact["schema"] == "carnot.adversarial_gsm8k.v1"
