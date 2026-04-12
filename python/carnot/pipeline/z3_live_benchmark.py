"""Helpers for Exp 206: live GSM8K benchmarking with Z3-backed verification.

Provides the benchmark bookkeeping for paired baseline / verify-only /
verify-repair evaluation on shared baseline responses, plus side-by-side
comparison against the legacy regex arithmetic extractor.

Spec: REQ-VERIFY-009, SCENARIO-VERIFY-009
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import numpy as np

from carnot.pipeline.verify_repair import VerifyRepairPipeline

if TYPE_CHECKING:
    from collections.abc import Callable

    from carnot.pipeline.extract import ConstraintResult
    from carnot.pipeline.verify_repair import VerificationResult


def sample_questions(
    questions: list[dict[str, Any]], sample_size: int, sample_seed: int
) -> list[dict[str, Any]]:
    """Return the first ``sample_size`` questions from a seeded shuffle."""
    shuffled = list(questions)
    random.Random(sample_seed).shuffle(shuffled)
    return shuffled[:sample_size]


def run_verify_only(
    *,
    question: str,
    response: str,
    pipeline: Any,
    domain: str = "arithmetic",
) -> dict[str, Any]:
    """Run one verification pass and return a JSON-safe summary."""
    verification = pipeline.verify(question, response, domain=domain)
    flagged = (not verification.verified) or bool(verification.violations)
    return {
        "verified": verification.verified,
        "flagged": flagged,
        "energy": verification.energy,
        "n_constraints": len(verification.constraints),
        "n_violations": len(verification.violations),
        "violations": [_serialize_constraint(constraint) for constraint in verification.violations],
        "certificate": dict(verification.certificate),
    }


def run_repair_loop(
    *,
    question: str,
    ground_truth: int,
    initial_response: str,
    pipeline: Any,
    generate_response: Callable[[str], str],
    extract_answer: Callable[[str], int | None],
    max_repairs: int = 3,
    domain: str = "arithmetic",
) -> dict[str, Any]:
    """Run a verify-repair loop starting from a shared baseline response."""
    current_response = initial_response
    current_answer = extract_answer(current_response)
    initial_correct = current_answer == ground_truth

    verification = pipeline.verify(question, current_response, domain=domain)
    history = [_serialize_verification(verification)]

    if verification.verified:
        return {
            "initial_response": initial_response,
            "final_response": current_response,
            "initial_answer": current_answer,
            "final_answer": current_answer,
            "initial_correct": initial_correct,
            "correct": initial_correct,
            "verified": True,
            "repaired": False,
            "iterations": 0,
            "n_repairs": 0,
            "history": history,
        }

    n_repairs = 0
    for iteration in range(1, max_repairs + 1):
        feedback = VerifyRepairPipeline._format_violations(verification.violations)
        prompt = (
            f"Question: {question}\n\n"
            f"Your previous answer was:\n{current_response}\n\n"
            f"The following issues were found:\n{feedback}\n\n"
            "Please recalculate step by step and give a corrected answer.\n"
            "Format:\nAnswer: <number>"
        )
        current_response = generate_response(prompt)
        current_answer = extract_answer(current_response)
        verification = pipeline.verify(question, current_response, domain=domain)
        history.append(_serialize_verification(verification))
        n_repairs = iteration
        if verification.verified:
            break

    final_correct = current_answer == ground_truth
    return {
        "initial_response": initial_response,
        "final_response": current_response,
        "initial_answer": extract_answer(initial_response),
        "final_answer": current_answer,
        "initial_correct": initial_correct,
        "correct": final_correct,
        "verified": verification.verified,
        "repaired": (not initial_correct) and final_correct,
        "iterations": n_repairs,
        "n_repairs": n_repairs,
        "history": history,
    }


def summarize_cases(
    cases: list[dict[str, Any]],
    *,
    extractor_key: str,
    n_bootstrap: int = 10_000,
    seed: int = 206,
) -> dict[str, Any]:
    """Compute the Exp 206 benchmark metrics for one extractor."""
    if not cases:
        raise ValueError("summarize_cases() requires at least one case.")

    baseline_flags = [bool(case["baseline"]["correct"]) for case in cases]
    verify_only_flags = [
        bool(case["baseline"]["correct"])
        and not bool(case[extractor_key]["verify_only"]["flagged"])
        for case in cases
    ]
    repair_flags = [bool(case[extractor_key]["verify_repair"]["correct"]) for case in cases]

    n_questions = len(cases)
    n_wrong_answers = sum(1 for correct in baseline_flags if not correct)
    n_correct_answers = n_questions - n_wrong_answers
    n_wrong_detected = sum(
        1
        for case in cases
        if (not case["baseline"]["correct"]) and case[extractor_key]["verify_only"]["flagged"]
    )
    n_false_positives = sum(
        1
        for case in cases
        if case["baseline"]["correct"] and case[extractor_key]["verify_only"]["flagged"]
    )
    n_violations_on_wrong_answers = sum(
        int(case[extractor_key]["verify_only"]["n_violations"])
        for case in cases
        if not case["baseline"]["correct"]
    )

    baseline_acc, baseline_lo, baseline_hi = _bootstrap_ci(
        baseline_flags, n_bootstrap=n_bootstrap, seed=seed
    )
    verify_acc, verify_lo, verify_hi = _bootstrap_ci(
        verify_only_flags, n_bootstrap=n_bootstrap, seed=seed + 1
    )
    repair_acc, repair_lo, repair_hi = _bootstrap_ci(
        repair_flags, n_bootstrap=n_bootstrap, seed=seed + 2
    )
    delta, delta_lo, delta_hi = _bootstrap_delta_ci(
        baseline_flags, repair_flags, n_bootstrap=n_bootstrap, seed=seed + 3
    )

    accepted_count = sum(1 for case in cases if not case[extractor_key]["verify_only"]["flagged"])
    accepted_correct = sum(
        1
        for case in cases
        if case["baseline"]["correct"] and not case[extractor_key]["verify_only"]["flagged"]
    )
    n_repaired = sum(1 for case in cases if bool(case[extractor_key]["verify_repair"]["repaired"]))
    avg_repairs_per_question = (
        sum(int(case[extractor_key]["verify_repair"].get("n_repairs", 0)) for case in cases)
        / n_questions
    )

    return {
        "n_questions": n_questions,
        "baseline": {
            "accuracy": round(baseline_acc, 6),
            "ci_lower": round(baseline_lo, 6),
            "ci_upper": round(baseline_hi, 6),
            "n_correct": int(sum(baseline_flags)),
        },
        "verify_only": {
            "accuracy": round(verify_acc, 6),
            "ci_lower": round(verify_lo, 6),
            "ci_upper": round(verify_hi, 6),
            "n_correct": int(sum(verify_only_flags)),
            "n_wrong_answers": n_wrong_answers,
            "n_wrong_detected": n_wrong_detected,
            "wrong_detection_rate": round(_safe_rate(n_wrong_detected, n_wrong_answers), 6),
            "n_correct_answers": n_correct_answers,
            "n_false_positives": n_false_positives,
            "false_positive_rate": round(_safe_rate(n_false_positives, n_correct_answers), 6),
            "n_violations_on_wrong_answers": n_violations_on_wrong_answers,
            "accepted_count": accepted_count,
            "accepted_precision": round(_safe_rate(accepted_correct, accepted_count), 6),
        },
        "verify_repair": {
            "accuracy": round(repair_acc, 6),
            "ci_lower": round(repair_lo, 6),
            "ci_upper": round(repair_hi, 6),
            "n_correct": int(sum(repair_flags)),
            "n_repaired": n_repaired,
            "avg_repairs_per_question": round(avg_repairs_per_question, 6),
            "improvement_delta": round(delta, 6),
            "ci_delta_lower": round(delta_lo, 6),
            "ci_delta_upper": round(delta_hi, 6),
        },
    }


def compare_extractors(z3_summary: dict[str, Any], regex_summary: dict[str, Any]) -> dict[str, Any]:
    """Return the side-by-side Z3 vs regex verdict."""
    z3_detected = int(z3_summary["verify_only"]["n_wrong_detected"])
    regex_detected = int(regex_summary["verify_only"]["n_wrong_detected"])
    z3_fp = float(z3_summary["verify_only"]["false_positive_rate"])
    regex_fp = float(regex_summary["verify_only"]["false_positive_rate"])
    z3_delta = float(z3_summary["verify_repair"]["improvement_delta"])
    regex_delta = float(regex_summary["verify_repair"]["improvement_delta"])

    detected_at_least_as_many = z3_detected >= regex_detected
    lower_or_equal_fp = z3_fp <= regex_fp
    higher_or_equal_delta = z3_delta >= regex_delta
    has_strict_win = (
        (z3_detected > regex_detected) or (z3_fp < regex_fp) or (z3_delta > regex_delta)
    )

    return {
        "detected_at_least_as_many_wrong_answers": detected_at_least_as_many,
        "lower_or_equal_false_positive_rate": lower_or_equal_fp,
        "higher_or_equal_repair_delta": higher_or_equal_delta,
        "strictly_better": (
            detected_at_least_as_many
            and lower_or_equal_fp
            and higher_or_equal_delta
            and has_strict_win
        ),
        "z3_minus_regex": {
            "wrong_answers_detected": z3_detected - regex_detected,
            "false_positive_rate": round(z3_fp - regex_fp, 6),
            "repair_delta": round(z3_delta - regex_delta, 6),
        },
    }


def build_results_payload(
    *,
    timestamp: str,
    sample_seed: int,
    sample_size: int,
    sample_dataset_indices: list[int],
    max_new_tokens: int,
    max_repairs: int,
    runtime_seconds: float,
    model_name: str,
    hf_id: str,
    inference_mode: str,
    z3_summary: dict[str, Any],
    regex_summary: dict[str, Any],
    comparison: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the final JSON payload for ``results/experiment_206_results.json``."""
    return {
        "experiment": 206,
        "title": "Z3 extractor on 100 live GSM8K (Gemma4-E4B-it) — the real test",
        "metadata": {
            "timestamp": timestamp,
            "model_name": model_name,
            "hf_id": hf_id,
            "inference_mode": inference_mode,
            "sample_size": sample_size,
            "sample_seed": sample_seed,
            "sample_strategy": "seeded_shuffle_first_n",
            "sample_dataset_indices": sample_dataset_indices,
            "max_new_tokens": max_new_tokens,
            "max_repairs": max_repairs,
            "bootstrap_samples": 10_000,
            "confidence_level": 0.95,
            "runtime_seconds": round(runtime_seconds, 3),
        },
        "statistics": {
            "z3": z3_summary,
            "regex": regex_summary,
        },
        "comparison": comparison,
        "results": cases,
    }


def _serialize_verification(verification: VerificationResult) -> dict[str, Any]:
    return {
        "verified": verification.verified,
        "energy": verification.energy,
        "n_constraints": len(verification.constraints),
        "n_violations": len(verification.violations),
        "violations": [_serialize_constraint(constraint) for constraint in verification.violations],
        "certificate": dict(verification.certificate),
    }


def _serialize_constraint(constraint: ConstraintResult) -> dict[str, Any]:
    return {
        "constraint_type": constraint.constraint_type,
        "description": constraint.description,
        "metadata": dict(constraint.metadata),
    }


def _bootstrap_ci(
    correct_flags: list[bool],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    arr = np.array(correct_flags, dtype=float)
    n = len(arr)
    point_estimate = float(np.mean(arr))
    indices = np.random.default_rng(seed).integers(0, n, size=(n_bootstrap, n))
    bootstrap_means = arr[indices].mean(axis=1)
    return (
        point_estimate,
        float(np.percentile(bootstrap_means, 2.5)),
        float(np.percentile(bootstrap_means, 97.5)),
    )


def _bootstrap_delta_ci(
    baseline_flags: list[bool],
    repair_flags: list[bool],
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float, float]:
    base = np.array(baseline_flags, dtype=float)
    repair = np.array(repair_flags, dtype=float)
    delta_per_question = repair - base
    n = len(delta_per_question)
    point_delta = float(np.mean(delta_per_question))
    indices = np.random.default_rng(seed).integers(0, n, size=(n_bootstrap, n))
    bootstrap_deltas = delta_per_question[indices].mean(axis=1)
    return (
        point_delta,
        float(np.percentile(bootstrap_deltas, 2.5)),
        float(np.percentile(bootstrap_deltas, 97.5)),
    )


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
