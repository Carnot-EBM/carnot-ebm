"""Comparative extraction benchmark: ArithmeticExtractor vs LLMConstraintExtractor vs LLMz3Formalizer.

**Researcher summary:**
    ArithmeticExtractor found 0/20 violations on Gemma4-E4B-it responses (Exp 353/355)
    because instruction-tuned models use markdown prose rather than bare "X + Y = Z" lines.
    This module benchmarks all three extractors on the same set of live responses so we
    can measure whether LLMConstraintExtractor or LLMz3Formalizer actually improves
    violation detection on IT-format output.

**What this module provides:**
    - `ExtractionBenchmarkResult`: per-extractor metrics dataclass (TP, FP, FN, TN,
      detection_rate, false_positive_rate, extractor_name, inference_mode).
    - `run_extraction_benchmark()`: runs one extractor against a set of labelled responses
      and returns an `ExtractionBenchmarkResult`.
    - `build_extraction_comparison_artifact()`: combines per-extractor results into a
      comparison dict with a winner selection and an honest_verdict string.

**Honest verdict contract (REQ-EXTRACT-021):**
    `honest_verdict == "live_gpu_llm_extractor_wins"` ONLY when:
      1. inference_mode of LLMConstraintExtractor result is "live_gpu", AND
      2. LLMConstraintExtractor.detection_rate > ArithmeticExtractor.detection_rate.
    All other conditions yield a non-winning string so simulated runs can never claim
    a headline result.

Spec: REQ-EXTRACT-021, SCENARIO-EXTRACT-042, SCENARIO-EXTRACT-043
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Protocol: ExtractorWrapper
# ---------------------------------------------------------------------------


@runtime_checkable
class ViolationDetector(Protocol):
    """Protocol for an object that can detect violations in a (question, response) pair.

    **Detailed explanation for engineers:**
        The three extractors in this benchmark have different APIs:
        - ArithmeticExtractor.extract(text, domain) -> list[ConstraintResult]
        - LLMConstraintExtractor.extract(text, domain) -> list[ConstraintResult]
        - LLMz3Formalizer.formalize(question, response) -> Z3FormalizationResult

        Rather than hard-coding each API, we wrap each extractor in a thin callable
        that takes (question: str, response: str) and returns True if a violation was
        detected.  The `run_extraction_benchmark` function accepts any such callable
        via the `inference_fn` parameter, making the benchmark extractor-agnostic.

    Spec: REQ-EXTRACT-021
    """

    def __call__(self, question: str, response: str) -> bool:
        """Return True if the extractor detected a violation in this (question, response) pair."""
        ...


# ---------------------------------------------------------------------------
# ExtractionBenchmarkResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExtractionBenchmarkResult:
    """Per-extractor metrics from a comparative extraction benchmark run.

    **Detailed explanation for engineers:**
        Every extractor is benchmarked on the same set of labelled responses.
        For wrong answers (ground_truth_wrong=True), a TP is scored when the
        extractor detects a violation.  For correct answers, a FP is scored when
        the extractor raises a false alarm.

        detection_rate = TP / (TP + FN) — how often the extractor catches known errors.
        false_positive_rate = FP / (FP + TN) — how often it fires on correct answers.

        Both rates are 0.0 when the denominator is zero (no wrong/correct answers).

        inference_mode is passed in by the caller and should be "live_gpu" when
        CARNOT_FORCE_LIVE=1 was set for the outer inference loop, or "simulated" otherwise.
        This field is critical: the honest_verdict in the comparison artifact is only
        "live_gpu_llm_extractor_wins" when inference_mode == "live_gpu".

    Attributes:
        extractor_name:       Short identifier (e.g. "arithmetic", "llm", "z3").
        n_questions:          Total number of (question, response) pairs evaluated.
        n_violations_found:   Number of responses where the extractor raised a violation.
        n_true_positives:     Violations found on known-wrong responses (TP).
        n_false_positives:    Violations found on known-correct responses (FP).
        detection_rate:       TP / (TP + FN).
        false_positive_rate:  FP / (FP + TN).
        inference_mode:       "live_gpu" or "simulated".

    Spec: REQ-EXTRACT-021, SCENARIO-EXTRACT-042
    """

    extractor_name: str
    n_questions: int
    n_violations_found: int
    n_true_positives: int
    n_false_positives: int
    detection_rate: float
    false_positive_rate: float
    inference_mode: str


# ---------------------------------------------------------------------------
# run_extraction_benchmark
# ---------------------------------------------------------------------------


def run_extraction_benchmark(
    extractor_name: str,
    inference_fn: Callable[[str, str], bool],
    questions: list[dict],
    ground_truth_wrong: list[bool],
    inference_mode: str,
) -> ExtractionBenchmarkResult:
    """Run one extractor against a labelled set of (question, response) pairs.

    **Detailed explanation for engineers:**
        Each entry in `questions` must have at least "question" and "response" keys.
        The caller provides `ground_truth_wrong[i] = True` when the i-th response is
        known to be incorrect (e.g. answer does not match GSM8K ground truth).

        The function calls `inference_fn(question, response)` for each pair.  If it
        returns True, a violation was detected.  We then cross-reference with ground_truth_wrong:
        - violation AND wrong  → True Positive (TP)
        - violation AND correct → False Positive (FP)
        - no violation AND wrong → False Negative (FN, implicit: contributes to denominator)
        - no violation AND correct → True Negative (TN, implicit)

        detection_rate and false_positive_rate are computed from TP/FP/FN/TN counts.
        Both are 0.0 if their denominator is zero.

    Args:
        extractor_name:      Short identifier for the extractor (e.g. "arithmetic").
        inference_fn:        Callable (question, response) -> bool: True if violation detected.
        questions:           List of dicts, each with "question" (str) and "response" (str).
        ground_truth_wrong:  Parallel bool list; True means the response is known-wrong.
        inference_mode:      "live_gpu" or "simulated".

    Returns:
        ExtractionBenchmarkResult populated with TP, FP, detection_rate, etc.

    Spec: REQ-EXTRACT-021, SCENARIO-EXTRACT-042
    """
    if len(questions) != len(ground_truth_wrong):
        raise ValueError(
            f"questions length ({len(questions)}) must equal "
            f"ground_truth_wrong length ({len(ground_truth_wrong)})"
        )

    n_questions = len(questions)
    n_violations_found = 0
    n_true_positives = 0
    n_false_positives = 0
    n_wrong = sum(1 for g in ground_truth_wrong if g)
    n_correct = n_questions - n_wrong

    for item, is_wrong in zip(questions, ground_truth_wrong):
        question = item["question"]
        response = item["response"]
        violated = inference_fn(question, response)

        if violated:
            n_violations_found += 1
            if is_wrong:
                n_true_positives += 1
            else:
                n_false_positives += 1

    n_false_negatives = n_wrong - n_true_positives
    n_true_negatives = n_correct - n_false_positives

    detection_rate = (
        n_true_positives / (n_true_positives + n_false_negatives)
        if (n_true_positives + n_false_negatives) > 0
        else 0.0
    )
    false_positive_rate = (
        n_false_positives / (n_false_positives + n_true_negatives)
        if (n_false_positives + n_true_negatives) > 0
        else 0.0
    )

    return ExtractionBenchmarkResult(
        extractor_name=extractor_name,
        n_questions=n_questions,
        n_violations_found=n_violations_found,
        n_true_positives=n_true_positives,
        n_false_positives=n_false_positives,
        detection_rate=round(detection_rate, 6),
        false_positive_rate=round(false_positive_rate, 6),
        inference_mode=inference_mode,
    )


# ---------------------------------------------------------------------------
# build_extraction_comparison_artifact
# ---------------------------------------------------------------------------


def build_extraction_comparison_artifact(
    results: list[ExtractionBenchmarkResult],
) -> dict:
    """Combine per-extractor benchmark results into a comparison summary dict.

    **Detailed explanation for engineers:**
        The artifact records which extractor achieved the highest detection_rate
        (ties broken by lowest false_positive_rate, then by extractor_name for
        determinism).  The `improvement_over_arithmetic_extractor` field is the
        absolute detection_rate improvement of the winner over "arithmetic" baseline.

        honest_verdict contract (REQ-EXTRACT-021):
        - "live_gpu_llm_extractor_wins": only when at least one result has
          inference_mode="live_gpu" AND the "llm" extractor has strictly higher
          detection_rate than the "arithmetic" extractor.
        - "live_gpu_no_improvement": live_gpu mode confirmed but llm extractor does
          not improve over arithmetic.
        - "simulated_no_verdict": inference_mode != "live_gpu" for all results.
        - "insufficient_data": fewer than 2 results provided, no comparison possible.

    Args:
        results: List of ExtractionBenchmarkResult, one per extractor.

    Returns:
        dict with keys: per_extractor_results, winner, improvement_over_arithmetic_extractor,
        honest_verdict, n_extractors.

    Spec: REQ-EXTRACT-021, SCENARIO-EXTRACT-043
    """
    if not results:
        return {
            "per_extractor_results": [],
            "winner": None,
            "improvement_over_arithmetic_extractor": 0.0,
            "honest_verdict": "insufficient_data",
            "n_extractors": 0,
        }

    per_extractor = [
        {
            "extractor_name": r.extractor_name,
            "n_questions": r.n_questions,
            "n_violations_found": r.n_violations_found,
            "n_true_positives": r.n_true_positives,
            "n_false_positives": r.n_false_positives,
            "detection_rate": r.detection_rate,
            "false_positive_rate": r.false_positive_rate,
            "inference_mode": r.inference_mode,
        }
        for r in results
    ]

    # Select winner: highest detection_rate; tiebreak: lowest fp_rate; final: name sort
    winner_result = max(
        results,
        key=lambda r: (r.detection_rate, -r.false_positive_rate, r.extractor_name),
    )
    winner = winner_result.extractor_name

    # Compute improvement over arithmetic baseline
    arithmetic_rate = next(
        (r.detection_rate for r in results if r.extractor_name == "arithmetic"),
        None,
    )
    improvement = 0.0
    if arithmetic_rate is not None:
        improvement = round(winner_result.detection_rate - arithmetic_rate, 6)

    # Determine honest_verdict
    if len(results) < 2:
        honest_verdict = "insufficient_data"
    else:
        any_live = any(r.inference_mode == "live_gpu" for r in results)
        llm_rate = next(
            (r.detection_rate for r in results if r.extractor_name == "llm"),
            None,
        )
        arith_rate = arithmetic_rate

        if not any_live:
            honest_verdict = "simulated_no_verdict"
        elif llm_rate is not None and arith_rate is not None and llm_rate > arith_rate:
            honest_verdict = "live_gpu_llm_extractor_wins"
        else:
            honest_verdict = "live_gpu_no_improvement"

    return {
        "per_extractor_results": per_extractor,
        "winner": winner,
        "improvement_over_arithmetic_extractor": improvement,
        "honest_verdict": honest_verdict,
        "n_extractors": len(results),
    }
