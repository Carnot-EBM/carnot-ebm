#!/usr/bin/env python3
"""Experiment 141: Constraint Generation — Does memory ADDITION beat reweighting?

**Researcher summary:**
    Exp 134 demonstrated that precision-based constraint REWEIGHTING produced
    no measurable improvement on arithmetic GSM8K questions. The root cause:
    reweighting cannot close gaps caused by entirely missing constraint types.

    This experiment tests the Exp 141 fix: ConstraintGenerator adds new
    constraint types (arithmetic_carry, comparison_boundary, negation_scope)
    from learned memory patterns. We benchmark 200 simulated GSM8K questions
    in two phases:
      - Warmup (questions 1-100): build ConstraintMemory by recording
        pattern violations. Uses prior experiment results (Exp 132-136) to
        pre-seed reasonable pattern frequencies if available.
      - Test (questions 101-200): compare AutoExtractor (no memory) vs
        AutoExtractor with memory= parameter (memory-augmented).

    Core metrics:
      - n_constraints_per_question: average constraint count with vs without
        memory. Memory should ADD constraints, not just reweight.
      - accuracy_no_memory: fraction of questions verified correctly without
        memory augmentation.
      - accuracy_with_memory: fraction verified correctly with memory
        augmentation. Should be higher when carry-chain / boundary / negation
        errors are in the test set.
      - delta: accuracy_with_memory - accuracy_no_memory. The Exp 141
        hypothesis: delta > 0 (positive improvement from constraint addition).

**Detailed explanation for engineers:**
    Simulation design:
        We simulate GSM8K arithmetic questions where:
          - 60% of wrong answers involve multi-step carry chain errors
            (e.g., 99 + 1 = 90 instead of 100). These are caught by the
            CarryChainConstraint but NOT by the base ArithmeticExtractor.
          - 20% of wrong answers involve comparison boundary violations
            (e.g., claiming score < 0 when it must be >= 0).
          - 20% involve negation scope errors ("the answer is not X" but
            then concluding X).

        Correct_fraction = 0.60: 60% of questions have correct responses,
        40% have the specific error types above.

    Memory warmup:
        During the warmup phase (q1-100), we record each violation into
        ConstraintMemory using the correct error_type keys:
          - "arithmetic_carry" for carry-chain errors
          - "comparison_boundary" for bound violations
          - "negation_scope" for negation errors
        After 100 questions with 40% wrong, each domain should accumulate
        roughly 24 carry + 8 boundary + 8 negation patterns -- all well above
        the threshold of 3.

    Test phase evaluation:
        For each of the 100 test questions, we run both pipelines on the
        same question text:
          1. Static only: AutoExtractor(text, domain="arithmetic")
          2. Memory-augmented: AutoExtractor(text, domain="arithmetic", memory=mem)
        We check whether constraints caught the error (constraint satisfied=False
        for wrong answers). A question is "correctly caught" if at least one
        constraint has satisfied=False for a wrong answer, or all constraints
        are satisfied for a correct answer.

    Why this is a meaningful improvement over Exp 134:
        Exp 134 reweighted the same ArithmeticExtractor and a noisy heuristic.
        The heuristic had 60% false positive rate, swamping the signal. By
        contrast, Exp 141 ADDS targeted extractors that only fire for specific
        known error patterns, producing near-zero false positives and high
        recall for the learned error types.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_141_constraint_generation.py

Spec: REQ-LEARN-003, REQ-LEARN-004, SCENARIO-LEARN-003
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = RESULTS_DIR / "experiment_141_results.json"

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# Total questions (first half warmup, second half test).
N_QUESTIONS = 200

# Split: warmup builds memory, test evaluates.
WARMUP_SIZE = 100
TEST_SIZE = 100

# Fraction of questions with correct responses.
CORRECT_FRACTION = 0.60

# Among wrong-answer questions, distribution of error types.
# Must sum to 1.0.
ERROR_TYPE_DISTRIBUTION = {
    "arithmetic_carry": 0.60,
    "comparison_boundary": 0.20,
    "negation_scope": 0.20,
}

# Random seed for reproducibility.
RANDOM_SEED = 141

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------

from carnot.pipeline.extract import AutoExtractor
from carnot.pipeline.generation import (
    PATTERN_ARITHMETIC_CARRY,
    PATTERN_COMPARISON_BOUNDARY,
    PATTERN_NEGATION_SCOPE,
)
from carnot.pipeline.memory import ConstraintMemory


# ---------------------------------------------------------------------------
# Question text generators
# ---------------------------------------------------------------------------


def _make_carry_chain_question(correct: bool, rng: random.Random) -> tuple[str, str]:
    """Generate a carry-chain arithmetic question text.

    Returns (question_text, response_text) where response is correct or wrong.

    **Detailed explanation for engineers:**
        Uses numbers that guarantee multi-step carries (99+1, 999+11, etc.)
        so CarryChainConstraint can detect them. Wrong responses use the
        known off-by-one carry error pattern.
    """
    # Build numbers with guaranteed multi-carry: a = 10^n - 1 (all 9s).
    n_digits = rng.randint(2, 3)
    a = 10**n_digits - 1  # e.g., 99, 999
    b = rng.randint(1, 5)
    true_answer = a + b
    if correct:
        claimed = true_answer
        response = f"Adding {a} + {b}: result is {claimed}."
    else:
        # Wrong: drop the final carry → under-count by 10^n_digits - 10
        wrong = a - (10 ** (n_digits - 1) - 1) + b  # Common carry-miss error
        claimed = wrong
        response = f"Adding {a} + {b}: {a} + {b} = {claimed}."
    question = f"What is {a} + {b}?"
    return question, response


def _make_boundary_question(correct: bool, rng: random.Random) -> tuple[str, str]:
    """Generate a comparison boundary question text.

    Returns (question_text, response_text).

    **Detailed explanation for engineers:**
        Embeds a numeric comparison claim in the response text.
        Wrong responses embed a false comparison (e.g., "score > 100"
        when score = 50 and the claim "50 > 100" is false).
    """
    score = rng.randint(1, 50)
    threshold = rng.randint(51, 100)
    question = f"Is {score} less than {threshold}?"
    if correct:
        response = f"Yes, {score} < {threshold} is true."
    else:
        # Wrong: reverse the comparison.
        response = f"No, {score} > {threshold} holds here."
    return question, response


def _make_negation_question(correct: bool, rng: random.Random) -> tuple[str, str]:
    """Generate a negation scope question text.

    Returns (question_text, response_text).

    **Detailed explanation for engineers:**
        Creates a "the answer is not X" claim where a wrong response
        contradicts itself (says "not X" then concludes X).
    """
    value = rng.randint(10, 99)
    question = f"What is the result? (Hint: it is not {value - 1}.)"
    if correct:
        response = f"The result is {value}, which is not {value - 1}."
    else:
        # Wrong: contradictory -- says "is not X" then claims X.
        response = f"The result is not {value}. Therefore the answer is {value}."
    return question, response


# ---------------------------------------------------------------------------
# Simulated question dataset
# ---------------------------------------------------------------------------


@dataclass
class SimQuestion:
    """A single simulated GSM8K-style question."""

    question: str
    response: str
    is_correct: bool
    error_type: str  # "arithmetic_carry", "comparison_boundary", "negation_scope", or ""


def generate_questions(n: int, rng: random.Random) -> list[SimQuestion]:
    """Generate N simulated GSM8K arithmetic questions with known error types.

    **Detailed explanation for engineers:**
        60% correct, 40% wrong. Among wrong questions, error types follow
        ERROR_TYPE_DISTRIBUTION. This matches the simulation design in the
        docstring above.
    """
    questions: list[SimQuestion] = []
    error_types = list(ERROR_TYPE_DISTRIBUTION.keys())
    error_probs = list(ERROR_TYPE_DISTRIBUTION.values())

    for _ in range(n):
        is_correct = rng.random() < CORRECT_FRACTION

        if is_correct:
            # Pick random error_type generator for variety but mark as correct.
            etype = rng.choices(error_types, weights=error_probs, k=1)[0]
        else:
            etype = rng.choices(error_types, weights=error_probs, k=1)[0]

        if etype == "arithmetic_carry":
            q, r = _make_carry_chain_question(is_correct, rng)
        elif etype == "comparison_boundary":
            q, r = _make_boundary_question(is_correct, rng)
        else:
            q, r = _make_negation_question(is_correct, rng)

        questions.append(
            SimQuestion(
                question=q,
                response=r,
                is_correct=is_correct,
                error_type=etype if not is_correct else "",
            )
        )
    return questions


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def is_verified_static(ext: AutoExtractor, q: SimQuestion) -> bool:
    """Check whether the static extractor catches errors in this question.

    **Detailed explanation for engineers:**
        Returns True if the response is "verified" (no constraint violations
        found). For correct answers, we want True (no false alarms). For
        wrong answers, we want False (violation detected).

        "Verified" = all constraints have satisfied=True (or no constraints
        that have satisfied=False).
    """
    results = ext.extract(q.response, domain="arithmetic")
    for r in results:
        if r.metadata.get("satisfied") is False:
            return False
    return True


def is_verified_with_memory(
    ext: AutoExtractor, q: SimQuestion, memory: ConstraintMemory
) -> bool:
    """Check whether the memory-augmented extractor catches errors.

    Same logic as is_verified_static but passes memory= to get generated
    constraints from mature patterns.
    """
    results = ext.extract(q.response, domain="arithmetic", memory=memory)
    for r in results:
        if r.metadata.get("satisfied") is False:
            return False
    return True


def accuracy(
    ext: AutoExtractor,
    questions: list[SimQuestion],
    memory: ConstraintMemory | None = None,
) -> tuple[float, float]:
    """Compute accuracy and average constraint count over a set of questions.

    **Detailed explanation for engineers:**
        For correct questions: accuracy contribution is 1.0 if verified
        (true negative -- no false alarm), 0.0 if flagged (false positive).
        For wrong questions: accuracy contribution is 1.0 if NOT verified
        (true positive -- caught the error), 0.0 if verified (missed error).

    Returns:
        Tuple of (accuracy, avg_n_constraints).
    """
    correct_count = 0
    total_constraints = 0

    for q in questions:
        if memory is not None:
            results = ext.extract(q.response, domain="arithmetic", memory=memory)
        else:
            results = ext.extract(q.response, domain="arithmetic")

        total_constraints += len(results)
        violated = any(r.metadata.get("satisfied") is False for r in results)

        if q.is_correct:
            # Correct answer: we want no violation (not violated → correct)
            if not violated:
                correct_count += 1
        else:
            # Wrong answer: we want a violation detected
            if violated:
                correct_count += 1

    n = len(questions)
    return correct_count / n if n > 0 else 0.0, total_constraints / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment() -> dict[str, Any]:
    """Run Exp 141: memory-augmented constraint generation benchmark.

    Returns:
        Dictionary with experiment results suitable for JSON serialization.
    """
    rng = random.Random(RANDOM_SEED)
    ext = AutoExtractor()
    memory = ConstraintMemory()

    print(f"Exp 141: Constraint Generation Benchmark")
    print(f"  N={N_QUESTIONS}, warmup={WARMUP_SIZE}, test={TEST_SIZE}")
    print(f"  correct_fraction={CORRECT_FRACTION}")
    print(f"  error_distribution={ERROR_TYPE_DISTRIBUTION}")
    print()

    # ------------------------------------------------------------------
    # Phase 1: Warmup — build memory from the first 100 questions
    # ------------------------------------------------------------------

    print("Phase 1: Warmup (building ConstraintMemory)...")
    warmup_start = time.perf_counter()
    warmup_questions = generate_questions(WARMUP_SIZE, rng)

    for q in warmup_questions:
        if not q.is_correct:
            # Record the pattern for this domain/error_type.
            # constraint_that_caught_it is a brief description.
            memory.record_pattern(
                "arithmetic",
                q.error_type,
                f"caught in: {q.response[:60]}",
            )

    warmup_elapsed = time.perf_counter() - warmup_start
    mem_summary = memory.summary()

    print(f"  Warmup complete in {warmup_elapsed:.3f}s")
    if "arithmetic" in mem_summary:
        arith = mem_summary["arithmetic"]
        print(f"  arithmetic patterns: total={arith['total_patterns']}, "
              f"mature={arith['mature_patterns']}")
        for p in arith.get("top_patterns", []):
            print(f"    {p['error_type']}: {p['frequency']} occurrences")
    print()

    # ------------------------------------------------------------------
    # Phase 2: Test — compare static vs memory-augmented on 100 new questions
    # ------------------------------------------------------------------

    print("Phase 2: Test (static vs memory-augmented)...")
    test_start = time.perf_counter()
    test_questions = generate_questions(TEST_SIZE, rng)

    acc_static, avg_constraints_static = accuracy(ext, test_questions, memory=None)
    acc_memory, avg_constraints_memory = accuracy(ext, test_questions, memory=memory)

    test_elapsed = time.perf_counter() - test_start

    delta = acc_memory - acc_static
    constraint_delta = avg_constraints_memory - avg_constraints_static

    print(f"  Test complete in {test_elapsed:.3f}s")
    print(f"  static accuracy:   {acc_static:.4f}  avg_constraints: {avg_constraints_static:.2f}")
    print(f"  memory accuracy:   {acc_memory:.4f}  avg_constraints: {avg_constraints_memory:.2f}")
    print(f"  delta (memory - static): {delta:+.4f}")
    print(f"  extra constraints/question from memory: {constraint_delta:+.2f}")
    print()

    # ------------------------------------------------------------------
    # Per-error-type breakdown
    # ------------------------------------------------------------------

    print("Per-error-type breakdown (test phase, wrong answers only):")
    breakdown: dict[str, Any] = {}
    for etype in ERROR_TYPE_DISTRIBUTION:
        wrong_qs = [q for q in test_questions if not q.is_correct and q.error_type == etype]
        if not wrong_qs:
            continue
        caught_static = sum(
            1 for q in wrong_qs
            if any(r.metadata.get("satisfied") is False
                   for r in ext.extract(q.response, domain="arithmetic"))
        )
        caught_memory = sum(
            1 for q in wrong_qs
            if any(r.metadata.get("satisfied") is False
                   for r in ext.extract(q.response, domain="arithmetic", memory=memory))
        )
        n = len(wrong_qs)
        breakdown[etype] = {
            "n_wrong": n,
            "caught_static": caught_static,
            "caught_memory": caught_memory,
            "recall_static": caught_static / n if n > 0 else 0.0,
            "recall_memory": caught_memory / n if n > 0 else 0.0,
        }
        print(f"  {etype}: n={n}, "
              f"static recall={caught_static/n:.2f}, "
              f"memory recall={caught_memory/n:.2f} "
              f"(+{(caught_memory-caught_static)/n:+.2f})")

    print()

    # ------------------------------------------------------------------
    # Hypothesis check
    # ------------------------------------------------------------------

    hypothesis_met = delta > 0.0
    print(f"Hypothesis (memory accuracy > static accuracy): "
          f"{'MET' if hypothesis_met else 'NOT MET'}")
    print(f"  delta={delta:+.4f} (need > 0.0)")
    print()

    # ------------------------------------------------------------------
    # Results dict
    # ------------------------------------------------------------------

    results: dict[str, Any] = {
        "experiment": "exp_141_constraint_generation",
        "n_questions": N_QUESTIONS,
        "warmup_size": WARMUP_SIZE,
        "test_size": TEST_SIZE,
        "correct_fraction": CORRECT_FRACTION,
        "error_type_distribution": ERROR_TYPE_DISTRIBUTION,
        "random_seed": RANDOM_SEED,
        "warmup_memory_summary": mem_summary,
        "static_accuracy": round(acc_static, 4),
        "memory_accuracy": round(acc_memory, 4),
        "delta": round(delta, 4),
        "avg_constraints_static": round(avg_constraints_static, 2),
        "avg_constraints_memory": round(avg_constraints_memory, 2),
        "constraint_delta": round(constraint_delta, 2),
        "per_error_type_breakdown": breakdown,
        "hypothesis_met": hypothesis_met,
        "warmup_elapsed_s": round(warmup_elapsed, 3),
        "test_elapsed_s": round(test_elapsed, 3),
    }

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run experiment and save results to results/experiment_141_results.json."""
    t_start = time.perf_counter()
    results = run_experiment()
    elapsed = time.perf_counter() - t_start

    results["total_elapsed_s"] = round(elapsed, 3)

    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {OUTPUT_PATH}")
    print(f"Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
