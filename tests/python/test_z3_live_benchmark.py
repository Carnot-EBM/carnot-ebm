"""Tests for the Exp 206/207 live benchmark helpers.

Each test references REQ-VERIFY-* or SCENARIO-VERIFY-* per spec-anchored
development requirements.

Spec: REQ-VERIFY-009, REQ-VERIFY-010, SCENARIO-VERIFY-009, SCENARIO-VERIFY-010
"""

from __future__ import annotations

from carnot.pipeline.extract import ConstraintResult
from carnot.pipeline.verify_repair import VerificationResult
from carnot.pipeline.z3_live_benchmark import (
    build_comparison_payload,
    build_results_payload,
    compare_extractors,
    compare_named_extractors,
    run_repair_loop,
    run_verify_only,
    sample_questions,
    summarize_cases,
)


def _verification_result(
    *,
    verified: bool,
    n_constraints: int = 1,
    n_violations: int | None = None,
) -> VerificationResult:
    if n_violations is None:
        n_violations = 0 if verified else 1

    constraints = [
        ConstraintResult(
            constraint_type="arithmetic",
            description="47 + 28 = 76",
            metadata={
                "satisfied": verified,
                "correct_result": 75,
            },
        )
        for _ in range(n_constraints)
    ]
    violations = constraints[:n_violations] if not verified else []
    return VerificationResult(
        verified=verified,
        constraints=constraints,
        energy=0.0,
        violations=violations,
        certificate={},
    )


class FakePipeline:
    """Minimal verify() stub for repair-loop tests."""

    def __init__(self, by_response: dict[str, VerificationResult]) -> None:
        self._by_response = by_response
        self.calls: list[tuple[str, str, str | None]] = []

    def verify(self, question: str, response: str, domain: str | None = None) -> VerificationResult:
        self.calls.append((question, response, domain))
        return self._by_response[response]


def test_sample_questions_uses_seeded_shuffle_first_n() -> None:
    """REQ-VERIFY-009: Exp 206 question sampling is deterministic."""
    questions = [{"idx": idx} for idx in range(10)]

    sample = sample_questions(questions, sample_size=4, sample_seed=5)

    assert [item["idx"] for item in sample] == [2, 3, 1, 0]


def test_run_repair_loop_repairs_once_after_detected_violation() -> None:
    """SCENARIO-VERIFY-009: Repair loop retries after a solver-backed violation."""
    pipeline = FakePipeline(
        {
            "Answer: 76": _verification_result(verified=False),
            "Answer: 75": _verification_result(verified=True),
        }
    )
    prompts: list[str] = []

    def generate_response(prompt: str) -> str:
        prompts.append(prompt)
        return "Answer: 75"

    def extract_answer(response: str) -> int | None:
        if response == "Answer: 75":
            return 75
        if response == "Answer: 76":
            return 76
        return None

    result = run_repair_loop(
        question="What is 47 + 28?",
        ground_truth=75,
        initial_response="Answer: 76",
        pipeline=pipeline,
        generate_response=generate_response,
        extract_answer=extract_answer,
        max_repairs=3,
    )

    assert result["initial_correct"] is False
    assert result["correct"] is True
    assert result["repaired"] is True
    assert result["iterations"] == 1
    assert result["n_repairs"] == 1
    assert result["final_response"] == "Answer: 75"
    assert "Your previous answer was" in prompts[0]
    assert "correct answer: 75" in prompts[0]
    assert pipeline.calls == [
        ("What is 47 + 28?", "Answer: 76", "arithmetic"),
        ("What is 47 + 28?", "Answer: 75", "arithmetic"),
    ]


def test_run_repair_loop_exits_without_generation_when_initial_response_verifies() -> None:
    """REQ-VERIFY-009: No repair generation is attempted when the first pass is SAT."""
    pipeline = FakePipeline({"Answer: 75": _verification_result(verified=True)})
    prompts: list[str] = []

    result = run_repair_loop(
        question="What is 47 + 28?",
        ground_truth=75,
        initial_response="Answer: 75",
        pipeline=pipeline,
        generate_response=prompts.append,
        extract_answer=lambda response: 75 if response == "Answer: 75" else None,
        max_repairs=3,
    )

    assert result["correct"] is True
    assert result["repaired"] is False
    assert result["iterations"] == 0
    assert result["n_repairs"] == 0
    assert prompts == []


def test_run_verify_only_serializes_violations_and_flagged_state() -> None:
    """REQ-VERIFY-009: Verify-only mode exposes violation details for repair feedback."""
    pipeline = FakePipeline({"Answer: 76": _verification_result(verified=False)})

    result = run_verify_only(
        question="What is 47 + 28?",
        response="Answer: 76",
        pipeline=pipeline,
    )

    assert result["verified"] is False
    assert result["flagged"] is True
    assert result["n_constraints"] == 1
    assert result["n_violations"] == 1
    assert result["violations"][0]["constraint_type"] == "arithmetic"
    assert result["violations"][0]["metadata"]["correct_result"] == 75


def test_summarize_cases_reports_detection_fpr_and_delta() -> None:
    """REQ-VERIFY-009: Benchmark summary computes the key live metrics."""
    cases = [
        {
            "baseline": {"correct": True},
            "z3": {
                "verify_only": {"flagged": False, "n_violations": 0},
                "verify_repair": {"correct": True, "repaired": False},
            },
        },
        {
            "baseline": {"correct": True},
            "z3": {
                "verify_only": {"flagged": False, "n_violations": 0},
                "verify_repair": {"correct": True, "repaired": False},
            },
        },
        {
            "baseline": {"correct": False},
            "z3": {
                "verify_only": {"flagged": True, "n_violations": 1},
                "verify_repair": {"correct": True, "repaired": True},
            },
        },
        {
            "baseline": {"correct": False},
            "z3": {
                "verify_only": {"flagged": False, "n_violations": 0},
                "verify_repair": {"correct": False, "repaired": False},
            },
        },
    ]

    summary = summarize_cases(cases, extractor_key="z3", n_bootstrap=256, seed=9)

    assert summary["n_questions"] == 4
    assert summary["baseline"]["accuracy"] == 0.5
    assert summary["verify_only"]["accuracy"] == 0.5
    assert summary["verify_only"]["n_wrong_answers"] == 2
    assert summary["verify_only"]["n_wrong_detected"] == 1
    assert summary["verify_only"]["wrong_detection_rate"] == 0.5
    assert summary["verify_only"]["false_positive_rate"] == 0.0
    assert summary["verify_repair"]["accuracy"] == 0.75
    assert summary["verify_repair"]["n_repaired"] == 1
    assert summary["verify_repair"]["improvement_delta"] == 0.25
    assert summary["verify_repair"]["ci_delta_lower"] <= 0.25
    assert summary["verify_repair"]["ci_delta_upper"] >= 0.25


def test_summarize_cases_handles_empty_detection_denominators() -> None:
    """REQ-VERIFY-009: Summary rates fall back to zero when a denominator is empty."""
    cases = [
        {
            "baseline": {"correct": True},
            "z3": {
                "verify_only": {"flagged": False, "n_violations": 0},
                "verify_repair": {"correct": True, "repaired": False, "n_repairs": 0},
            },
        }
    ]

    summary = summarize_cases(cases, extractor_key="z3", n_bootstrap=32, seed=11)

    assert summary["verify_only"]["n_wrong_answers"] == 0
    assert summary["verify_only"]["wrong_detection_rate"] == 0.0
    assert summary["verify_only"]["false_positive_rate"] == 0.0


def test_compare_extractors_requires_non_regression_and_one_strict_win() -> None:
    """SCENARIO-VERIFY-009: Z3 can be declared strictly better than regex."""
    z3_summary = {
        "verify_only": {
            "n_wrong_detected": 3,
            "false_positive_rate": 0.0,
        },
        "verify_repair": {
            "improvement_delta": 0.08,
        },
    }
    regex_summary = {
        "verify_only": {
            "n_wrong_detected": 2,
            "false_positive_rate": 0.05,
        },
        "verify_repair": {
            "improvement_delta": 0.06,
        },
    }

    comparison = compare_extractors(z3_summary, regex_summary)

    assert comparison["detected_at_least_as_many_wrong_answers"] is True
    assert comparison["lower_or_equal_false_positive_rate"] is True
    assert comparison["higher_or_equal_repair_delta"] is True
    assert comparison["strictly_better"] is True


def test_compare_named_extractors_reports_metric_winners_for_exp207() -> None:
    """SCENARIO-VERIFY-010: Exp 207 can compare LLM extraction directly against Z3."""
    llm_summary = {
        "verify_only": {
            "n_wrong_detected": 4,
            "n_violations_on_wrong_answers": 6,
            "false_positive_rate": 0.01,
        },
        "verify_repair": {
            "improvement_delta": 0.12,
        },
    }
    z3_summary = {
        "verify_only": {
            "n_wrong_detected": 2,
            "n_violations_on_wrong_answers": 3,
            "false_positive_rate": 0.03,
        },
        "verify_repair": {
            "improvement_delta": 0.05,
        },
    }

    comparison = compare_named_extractors(
        llm_summary,
        z3_summary,
        primary_label="llm",
        secondary_label="z3",
    )

    assert comparison["llm_found_at_least_as_many_wrong_answers"] is True
    assert comparison["llm_lower_or_equal_false_positive_rate"] is True
    assert comparison["llm_higher_or_equal_repair_delta"] is True
    assert comparison["llm_strictly_better"] is True
    assert comparison["llm_minus_z3"]["wrong_answers_detected"] == 2
    assert comparison["llm_minus_z3"]["violations_on_wrong_answers"] == 3
    assert comparison["llm_minus_z3"]["false_positive_rate"] == -0.02
    assert comparison["llm_minus_z3"]["repair_delta"] == 0.07
    assert comparison["winner_by_metric"] == {
        "wrong_answers_detected": "llm",
        "violations_on_wrong_answers": "llm",
        "false_positive_rate": "llm",
        "repair_delta": "llm",
    }


def test_compare_named_extractors_reports_ties_without_strict_win() -> None:
    """REQ-VERIFY-010: Generic comparison does not overstate tied extractor results."""
    summary = {
        "verify_only": {
            "n_wrong_detected": 1,
            "n_violations_on_wrong_answers": 1,
            "false_positive_rate": 0.0,
        },
        "verify_repair": {
            "improvement_delta": 0.0,
        },
    }

    comparison = compare_named_extractors(
        summary,
        summary,
        primary_label="llm",
        secondary_label="z3",
    )

    assert comparison["llm_found_at_least_as_many_wrong_answers"] is True
    assert comparison["llm_lower_or_equal_false_positive_rate"] is True
    assert comparison["llm_higher_or_equal_repair_delta"] is True
    assert comparison["llm_strictly_better"] is False
    assert comparison["winner_by_metric"] == {
        "wrong_answers_detected": "tie",
        "violations_on_wrong_answers": "tie",
        "false_positive_rate": "tie",
        "repair_delta": "tie",
    }


def test_compare_named_extractors_reports_when_secondary_wins_metrics() -> None:
    """REQ-VERIFY-010: Generic comparison can attribute wins to the secondary extractor."""
    llm_summary = {
        "verify_only": {
            "n_wrong_detected": 1,
            "n_violations_on_wrong_answers": 1,
            "false_positive_rate": 0.04,
        },
        "verify_repair": {
            "improvement_delta": -0.01,
        },
    }
    z3_summary = {
        "verify_only": {
            "n_wrong_detected": 2,
            "n_violations_on_wrong_answers": 3,
            "false_positive_rate": 0.01,
        },
        "verify_repair": {
            "improvement_delta": 0.02,
        },
    }

    comparison = compare_named_extractors(
        llm_summary,
        z3_summary,
        primary_label="llm",
        secondary_label="z3",
    )

    assert comparison["llm_found_at_least_as_many_wrong_answers"] is False
    assert comparison["llm_lower_or_equal_false_positive_rate"] is False
    assert comparison["llm_higher_or_equal_repair_delta"] is False
    assert comparison["llm_strictly_better"] is False
    assert comparison["winner_by_metric"] == {
        "wrong_answers_detected": "z3",
        "violations_on_wrong_answers": "z3",
        "false_positive_rate": "z3",
        "repair_delta": "z3",
    }


def test_build_results_payload_records_live_gpu_metadata_and_comparison() -> None:
    """REQ-VERIFY-009: Final artifact retains the benchmark metadata and verdict."""
    payload = build_results_payload(
        timestamp="2026-04-12T04:00:00Z",
        sample_seed=5,
        sample_size=100,
        sample_dataset_indices=[1, 2, 3],
        max_new_tokens=768,
        max_repairs=3,
        runtime_seconds=12.5,
        model_name="Gemma4-E4B-it",
        hf_id="google/gemma-4-E4B-it",
        inference_mode="live_gpu",
        z3_summary={"verify_repair": {"improvement_delta": 0.05}},
        regex_summary={"verify_repair": {"improvement_delta": 0.01}},
        comparison={"strictly_better": True},
        cases=[{"dataset_idx": 1}],
    )

    assert payload["experiment"] == 206
    assert payload["metadata"]["inference_mode"] == "live_gpu"
    assert payload["metadata"]["sample_seed"] == 5
    assert payload["statistics"]["z3"]["verify_repair"]["improvement_delta"] == 0.05
    assert payload["comparison"]["strictly_better"] is True
    assert payload["results"][0]["dataset_idx"] == 1


def test_build_comparison_payload_supports_exp207_metadata() -> None:
    """REQ-VERIFY-010: Generic payload builder preserves shared-baseline metadata."""
    payload = build_comparison_payload(
        experiment=207,
        title="LLM extractor on 100 live GSM8K (Gemma4-E4B-it) — head-to-head vs Z3",
        timestamp="2026-04-12T05:00:00Z",
        sample_seed=5,
        sample_size=100,
        sample_dataset_indices=[1, 2, 3],
        max_new_tokens=768,
        max_repairs=3,
        runtime_seconds=45.6,
        model_name="Gemma4-E4B-it",
        hf_id="google/gemma-4-E4B-it",
        inference_mode="live_gpu",
        statistics={
            "llm": {"verify_repair": {"improvement_delta": 0.04}},
            "z3": {"verify_repair": {"improvement_delta": 0.0}},
        },
        comparison={"llm_strictly_better": True},
        cases=[{"dataset_idx": 1}],
        extra_metadata={
            "baseline_source_experiment": 206,
            "shared_baseline_responses": True,
            "extractor_max_new_tokens": 256,
        },
    )

    assert payload["experiment"] == 207
    assert payload["metadata"]["inference_mode"] == "live_gpu"
    assert payload["metadata"]["baseline_source_experiment"] == 206
    assert payload["metadata"]["shared_baseline_responses"] is True
    assert payload["metadata"]["extractor_max_new_tokens"] == 256
    assert payload["statistics"]["llm"]["verify_repair"]["improvement_delta"] == 0.04
    assert payload["comparison"]["llm_strictly_better"] is True
    assert payload["results"][0]["dataset_idx"] == 1


def test_summarize_cases_rejects_empty_inputs() -> None:
    """SCENARIO-VERIFY-009: Summary refuses an empty benchmark cohort."""
    try:
        summarize_cases([], extractor_key="z3")
    except ValueError as exc:
        assert "requires at least one case" in str(exc)
    else:
        raise AssertionError("Expected summarize_cases() to reject empty input.")
