#!/usr/bin/env python3
"""Experiment 173: Constraint Generation v2 — NegationConstraint + CarryChainConstraint improvements.

**Researcher summary:**
    Exp 141 showed that memory-augmented constraint ADDITION boosts accuracy
    +0.11 (0.85 → 0.96) via carry-chain and comparison-boundary extraction.
    However, negation_scope recall was 0% because NegationConstraint only
    DETECTED patterns without checking for violations (no satisfied/violated
    discrimination). Similarly, CarryChainConstraint missed subtraction
    borrow chains and negative-result errors.

    This experiment benchmarks the v2 improvements on 300 targeted questions:
      - 100 negation violations: "X is not Y" in context, LLM asserts Y anyway
      - 100 carry/borrow violations: multi-digit arithmetic with carry/borrow errors
      - 100 mixed regression set (same distribution as Exp 141 test set)

    Target: +5pp over Exp 141's 0.96 = 0.98+ on the targeted set.

    Core metrics:
      - accuracy_static: fraction correct without memory augmentation
      - accuracy_memory_v2: fraction correct with v2 memory augmentation
      - delta_vs_exp141: improvement over Exp 141's memory accuracy (0.96)
      - negation_recall_v2: recall on negation_scope errors (was 0.0 in Exp 141)
      - subtraction_recall_v2: recall on subtraction borrow errors (new in v2)

**Detailed explanation for engineers:**
    Three test cohorts:

    Cohort A — Negation violations (100 questions):
        Generates texts where the context states "X is not Y" and the LLM
        response either respects or violates it. Violations: "not 42" then
        concludes "42". Respecting: "not 42" then concludes "17". Also tests
        "not all A are B" and "no A are B" patterns.

    Cohort B — Carry/borrow violations (100 questions):
        Equal mix of:
          - Addition carry-chain errors (from Exp 141)
          - Subtraction borrow-chain errors (1000-1=1099)
          - Negative-result errors (3-10=7 instead of -7)
        Each wrong answer embeds the error in a reasoning trace.

    Cohort C — Regression (100 questions):
        Same generator as Exp 141 with the same seed offset to produce a
        fresh set of arithmetic_carry / comparison_boundary / negation_scope
        questions. Verifies no regression from v2 changes.

    Memory warmup:
        A shared warmup pool of 100 questions (same as Exp 141's warmup) is
        used to pre-populate ConstraintMemory with mature patterns before
        the test phase begins.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_173_constraint_gen_v2.py

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

OUTPUT_PATH = RESULTS_DIR / "experiment_173_results.json"

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# Warmup size matches Exp 141 to compare fairly.
WARMUP_SIZE = 100

# Three cohorts of 100 questions each.
COHORT_NEGATION_SIZE = 100
COHORT_CARRY_SIZE = 100
COHORT_REGRESSION_SIZE = 100

# Fraction of correct answers per cohort.
CORRECT_FRACTION = 0.60

# Exp 141 baseline accuracy (memory-augmented) for comparison.
EXP_141_BASELINE = 0.96

# Random seeds: warmup uses 141 (matches Exp 141), cohorts use 173+offset.
WARMUP_SEED = 141
COHORT_NEGATION_SEED = 173
COHORT_CARRY_SEED = 174
COHORT_REGRESSION_SEED = 175

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
# Shared simulation helpers (copied from Exp 141 for regression cohort)
# ---------------------------------------------------------------------------


def _make_carry_chain_question(correct: bool, rng: random.Random) -> tuple[str, str]:
    """Generate an addition carry-chain question (same as Exp 141).

    Returns (question_text, response_text).
    """
    n_digits = rng.randint(2, 3)
    a = 10**n_digits - 1
    b = rng.randint(1, 5)
    true_answer = a + b
    if correct:
        claimed = true_answer
        response = f"Adding {a} + {b}: result is {claimed}."
    else:
        wrong = a - (10 ** (n_digits - 1) - 1) + b
        claimed = wrong
        response = f"Adding {a} + {b}: {a} + {b} = {claimed}."
    question = f"What is {a} + {b}?"
    return question, response


def _make_boundary_question(correct: bool, rng: random.Random) -> tuple[str, str]:
    """Generate a comparison boundary question (same as Exp 141)."""
    score = rng.randint(1, 50)
    threshold = rng.randint(51, 100)
    question = f"Is {score} less than {threshold}?"
    if correct:
        response = f"Yes, {score} < {threshold} is true."
    else:
        response = f"No, {score} > {threshold} holds here."
    return question, response


def _make_negation_question_v1(correct: bool, rng: random.Random) -> tuple[str, str]:
    """Generate an Exp 141-style negation question (for regression cohort).

    The v1 generator creates self-contradictory 'not X then concludes X'
    patterns that the v2 NegationConstraint can now actually detect.
    """
    value = rng.randint(10, 99)
    question = f"What is the result? (Hint: it is not {value - 1}.)"
    if correct:
        response = f"The result is {value}, which is not {value - 1}."
    else:
        response = f"The result is not {value}. Therefore the answer is {value}."
    return question, response


# ---------------------------------------------------------------------------
# Cohort A: Negation violation questions
# ---------------------------------------------------------------------------


def _make_negation_v2_question(correct: bool, rng: random.Random) -> tuple[str, str]:
    """Generate a targeted negation violation question (Cohort A).

    **Detailed explanation for engineers:**
        Three sub-types are generated equally:
        1. "X is not Y" → wrong response asserts "X is Y" (violation)
        2. "not all A are B" → wrong response asserts "all A are B"
        3. "no A are B" → wrong response asserts "A are B"

        Correct responses use a different value/assertion that doesn't
        contradict the negation.
    """
    sub_type = rng.choice(["is_not", "not_all", "no_are"])

    if sub_type == "is_not":
        # "The answer is not X" — violated by asserting X, respected by asserting Y.
        value = rng.randint(10, 99)
        other = value + rng.randint(1, 10)
        question = f"What is the answer? Note: it is NOT {value}."
        if correct:
            response = f"The answer is not {value}. The result is {other}."
        else:
            # Violation: say "not X" then conclude "is X".
            response = f"The answer is not {value}. The answer is {value}."
        return question, response

    elif sub_type == "not_all":
        # "not all A are B" — violated by asserting "all A are B".
        nouns = [("swans", "white"), ("dogs", "friendly"), ("birds", "small"),
                 ("cats", "black"), ("cars", "fast")]
        noun, prop = rng.choice(nouns)
        question = f"True or false: not all {noun} are {prop}?"
        if correct:
            response = f"True. Not all {noun} are {prop}. Some may be different."
        else:
            response = f"Not all {noun} are {prop}. Actually, all {noun} are {prop}."
        return question, response

    else:  # no_are
        # "no A are B" — violated by asserting "A are B".
        nouns = [("fish", "mammals"), ("rocks", "alive"), ("plants", "animals"),
                 ("metals", "liquids"), ("deserts", "forests")]
        subj, pred = rng.choice(nouns)
        question = f"Is it true that no {subj} are {pred}?"
        if correct:
            response = f"Yes, no {subj} are {pred}. This is correct."
        else:
            response = f"No {subj} are {pred}. However, {subj} are {pred} in some contexts."
        return question, response


# ---------------------------------------------------------------------------
# Cohort B: Carry/borrow violation questions
# ---------------------------------------------------------------------------


def _make_carry_v2_question(correct: bool, rng: random.Random) -> tuple[str, str]:
    """Generate a targeted carry/borrow violation question (Cohort B).

    **Detailed explanation for engineers:**
        Three sub-types are generated equally:
        1. Addition carry chain (same as Exp 141)
        2. Subtraction borrow chain (1000-1, 100-1)
        3. Negative result error (a - b where b > a)
    """
    sub_type = rng.choice(["add_carry", "sub_borrow", "negative"])

    if sub_type == "add_carry":
        return _make_carry_chain_question(correct, rng)

    elif sub_type == "sub_borrow":
        # Build numbers guaranteed to need cascade borrows: a = 10^n, b = small.
        n_digits = rng.randint(2, 3)
        a = 10**n_digits  # e.g., 100, 1000
        b = rng.randint(1, 5)
        true_answer = a - b
        question = f"What is {a} - {b}?"
        if correct:
            response = f"Subtracting {b} from {a}: {a} - {b} = {true_answer}."
        else:
            # Wrong: forgot to borrow, common error is to get a+b or wrong leading digit.
            wrong = a + b  # Borrow error: added instead of subtracted
            response = f"Computing {a} - {b}: {a} - {b} = {wrong}."
        return question, response

    else:  # negative
        # b > a: result must be negative.
        b = rng.randint(5, 20)
        a = rng.randint(1, b - 1)
        true_answer = a - b  # Negative
        question = f"What is {a} - {b}?"
        if correct:
            response = f"Since {b} > {a}, {a} - {b} = {true_answer}."
        else:
            # Wrong: claims positive result.
            wrong = b - a  # Forgot sign
            response = f"Computing {a} - {b}: {a} - {b} = {wrong}."
        return question, response


# ---------------------------------------------------------------------------
# SimQuestion dataclass
# ---------------------------------------------------------------------------


@dataclass
class SimQuestion:
    """A single simulated question for Exp 173."""

    question: str
    response: str
    is_correct: bool
    error_type: str
    cohort: str  # "negation", "carry", "regression"


# ---------------------------------------------------------------------------
# Question generators
# ---------------------------------------------------------------------------


def generate_negation_cohort(n: int, rng: random.Random) -> list[SimQuestion]:
    """Generate Cohort A: targeted negation violation questions."""
    questions: list[SimQuestion] = []
    for _ in range(n):
        is_correct = rng.random() < CORRECT_FRACTION
        q, r = _make_negation_v2_question(is_correct, rng)
        questions.append(SimQuestion(
            question=q,
            response=r,
            is_correct=is_correct,
            error_type="negation_scope" if not is_correct else "",
            cohort="negation",
        ))
    return questions


def generate_carry_cohort(n: int, rng: random.Random) -> list[SimQuestion]:
    """Generate Cohort B: targeted carry/borrow violation questions."""
    questions: list[SimQuestion] = []
    for _ in range(n):
        is_correct = rng.random() < CORRECT_FRACTION
        q, r = _make_carry_v2_question(is_correct, rng)
        questions.append(SimQuestion(
            question=q,
            response=r,
            is_correct=is_correct,
            error_type="arithmetic_carry" if not is_correct else "",
            cohort="carry",
        ))
    return questions


def generate_regression_cohort(n: int, rng: random.Random) -> list[SimQuestion]:
    """Generate Cohort C: Exp 141-style regression questions."""
    questions: list[SimQuestion] = []
    error_types = ["arithmetic_carry", "comparison_boundary", "negation_scope"]
    error_probs = [0.60, 0.20, 0.20]

    for _ in range(n):
        is_correct = rng.random() < CORRECT_FRACTION
        etype = rng.choices(error_types, weights=error_probs, k=1)[0]

        if etype == "arithmetic_carry":
            q, r = _make_carry_chain_question(is_correct, rng)
        elif etype == "comparison_boundary":
            q, r = _make_boundary_question(is_correct, rng)
        else:
            q, r = _make_negation_question_v1(is_correct, rng)

        questions.append(SimQuestion(
            question=q,
            response=r,
            is_correct=is_correct,
            error_type=etype if not is_correct else "",
            cohort="regression",
        ))
    return questions


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def compute_accuracy(
    ext: AutoExtractor,
    questions: list[SimQuestion],
    memory: ConstraintMemory | None = None,
) -> tuple[float, float]:
    """Compute accuracy and average constraint count.

    Correct answers: verified (no false alarms) → +1.
    Wrong answers: violation detected (not verified) → +1.

    Returns (accuracy, avg_n_constraints).
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
            if not violated:
                correct_count += 1
        else:
            if violated:
                correct_count += 1

    n = len(questions)
    return (
        correct_count / n if n > 0 else 0.0,
        total_constraints / n if n > 0 else 0.0,
    )


def per_error_breakdown(
    ext: AutoExtractor,
    questions: list[SimQuestion],
    memory: ConstraintMemory,
) -> dict[str, Any]:
    """Compute per-error-type recall for static vs memory-augmented."""
    breakdown: dict[str, Any] = {}
    error_types = {q.error_type for q in questions if q.error_type}

    for etype in sorted(error_types):
        wrong_qs = [q for q in questions if not q.is_correct and q.error_type == etype]
        if not wrong_qs:
            continue

        caught_static = sum(
            1 for q in wrong_qs
            if any(
                r.metadata.get("satisfied") is False
                for r in ext.extract(q.response, domain="arithmetic")
            )
        )
        caught_memory = sum(
            1 for q in wrong_qs
            if any(
                r.metadata.get("satisfied") is False
                for r in ext.extract(q.response, domain="arithmetic", memory=memory)
            )
        )
        n = len(wrong_qs)
        breakdown[etype] = {
            "n_wrong": n,
            "caught_static": caught_static,
            "caught_memory": caught_memory,
            "recall_static": round(caught_static / n, 4) if n > 0 else 0.0,
            "recall_memory": round(caught_memory / n, 4) if n > 0 else 0.0,
        }

    return breakdown


# ---------------------------------------------------------------------------
# Warmup: build ConstraintMemory from 100 questions (mirrors Exp 141)
# ---------------------------------------------------------------------------


def build_warmup_memory(rng: random.Random) -> ConstraintMemory:
    """Build ConstraintMemory from 100 warmup questions (Exp 141 protocol)."""
    memory = ConstraintMemory()
    error_types = ["arithmetic_carry", "comparison_boundary", "negation_scope"]
    error_probs = [0.60, 0.20, 0.20]

    for _ in range(WARMUP_SIZE):
        is_correct = rng.random() < CORRECT_FRACTION
        if not is_correct:
            etype = rng.choices(error_types, weights=error_probs, k=1)[0]
            memory.record_pattern(
                "arithmetic",
                etype,
                f"warmup example",
            )
    return memory


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_experiment() -> dict[str, Any]:
    """Run Exp 173: v2 constraint generation benchmark on 300 targeted questions."""
    print("Exp 173: Constraint Generation v2 Benchmark")
    print(f"  Cohort A (negation):    {COHORT_NEGATION_SIZE} questions")
    print(f"  Cohort B (carry/borrow): {COHORT_CARRY_SIZE} questions")
    print(f"  Cohort C (regression):   {COHORT_REGRESSION_SIZE} questions")
    print(f"  Warmup size: {WARMUP_SIZE}")
    print(f"  Correct fraction: {CORRECT_FRACTION}")
    print()

    ext = AutoExtractor()
    t_total = time.perf_counter()

    # ------------------------------------------------------------------
    # Warmup: build memory (same as Exp 141 protocol)
    # ------------------------------------------------------------------
    print("Building ConstraintMemory (warmup)...")
    t0 = time.perf_counter()
    warmup_rng = random.Random(WARMUP_SEED)
    memory = build_warmup_memory(warmup_rng)
    warmup_elapsed = time.perf_counter() - t0
    mem_summary = memory.summary()

    if "arithmetic" in mem_summary:
        arith = mem_summary["arithmetic"]
        print(f"  arithmetic patterns: total={arith['total_patterns']}, "
              f"mature={arith['mature_patterns']}")
        for p in arith.get("top_patterns", []):
            print(f"    {p['error_type']}: {p['frequency']} occurrences")
    print(f"  Warmup elapsed: {warmup_elapsed:.3f}s")
    print()

    # ------------------------------------------------------------------
    # Generate all three cohorts
    # ------------------------------------------------------------------
    cohort_a = generate_negation_cohort(COHORT_NEGATION_SIZE, random.Random(COHORT_NEGATION_SEED))
    cohort_b = generate_carry_cohort(COHORT_CARRY_SIZE, random.Random(COHORT_CARRY_SEED))
    cohort_c = generate_regression_cohort(COHORT_REGRESSION_SIZE, random.Random(COHORT_REGRESSION_SEED))
    all_questions = cohort_a + cohort_b + cohort_c

    # ------------------------------------------------------------------
    # Evaluate each cohort + combined
    # ------------------------------------------------------------------
    results_by_cohort: dict[str, Any] = {}
    all_breakdown: dict[str, Any] = {}

    for cohort_name, cohort_qs in [("negation", cohort_a), ("carry", cohort_b),
                                    ("regression", cohort_c)]:
        print(f"Evaluating cohort '{cohort_name}' ({len(cohort_qs)} questions)...")
        t0 = time.perf_counter()

        acc_static, avg_c_static = compute_accuracy(ext, cohort_qs, memory=None)
        acc_memory, avg_c_memory = compute_accuracy(ext, cohort_qs, memory=memory)
        cohort_elapsed = time.perf_counter() - t0

        breakdown = per_error_breakdown(ext, cohort_qs, memory)
        all_breakdown[cohort_name] = breakdown

        delta = acc_memory - acc_static
        print(f"  static:  {acc_static:.4f}  avg_constraints: {avg_c_static:.2f}")
        print(f"  memory:  {acc_memory:.4f}  avg_constraints: {avg_c_memory:.2f}")
        print(f"  delta:   {delta:+.4f}")
        for etype, bd in breakdown.items():
            print(f"  [{etype}] n={bd['n_wrong']}, "
                  f"static_recall={bd['recall_static']:.2f}, "
                  f"memory_recall={bd['recall_memory']:.2f}")
        print(f"  elapsed: {cohort_elapsed:.3f}s")
        print()

        results_by_cohort[cohort_name] = {
            "n_questions": len(cohort_qs),
            "accuracy_static": round(acc_static, 4),
            "accuracy_memory": round(acc_memory, 4),
            "delta": round(delta, 4),
            "avg_constraints_static": round(avg_c_static, 2),
            "avg_constraints_memory": round(avg_c_memory, 2),
            "per_error_breakdown": breakdown,
            "elapsed_s": round(cohort_elapsed, 3),
        }

    # ------------------------------------------------------------------
    # Combined (all 300 questions)
    # ------------------------------------------------------------------
    print(f"Evaluating combined (all 300 questions)...")
    t0 = time.perf_counter()
    acc_static_all, avg_c_static_all = compute_accuracy(ext, all_questions, memory=None)
    acc_memory_all, avg_c_memory_all = compute_accuracy(ext, all_questions, memory=memory)
    combined_elapsed = time.perf_counter() - t0

    delta_all = acc_memory_all - acc_static_all
    delta_vs_141 = acc_memory_all - EXP_141_BASELINE

    print(f"  static:        {acc_static_all:.4f}")
    print(f"  memory v2:     {acc_memory_all:.4f}")
    print(f"  delta (v2-static): {delta_all:+.4f}")
    print(f"  delta_vs_exp141:   {delta_vs_141:+.4f}  (baseline={EXP_141_BASELINE})")
    print(f"  elapsed: {combined_elapsed:.3f}s")
    print()

    # ------------------------------------------------------------------
    # Hypothesis checks
    # ------------------------------------------------------------------
    hyp_negation_recall_improved = (
        results_by_cohort["negation"].get("per_error_breakdown", {})
        .get("negation_scope", {})
        .get("recall_memory", 0.0) > 0.0
    )
    hyp_delta_vs_141_positive = delta_vs_141 > 0.0
    hyp_combined_above_098 = acc_memory_all >= 0.98

    print("Hypothesis checks:")
    print(f"  negation recall > 0 (was 0% in Exp 141): "
          f"{'MET' if hyp_negation_recall_improved else 'NOT MET'}")
    print(f"  delta_vs_exp141 > 0:                     "
          f"{'MET' if hyp_delta_vs_141_positive else 'NOT MET'}")
    print(f"  combined accuracy >= 0.98:               "
          f"{'MET' if hyp_combined_above_098 else 'NOT MET'}")
    print()

    total_elapsed = time.perf_counter() - t_total

    # ------------------------------------------------------------------
    # Results dict
    # ------------------------------------------------------------------
    results: dict[str, Any] = {
        "experiment": "exp_173_constraint_gen_v2",
        "date": "20260411",
        "parent_experiment": "exp_141",
        "n_questions_total": len(all_questions),
        "warmup_size": WARMUP_SIZE,
        "correct_fraction": CORRECT_FRACTION,
        "exp_141_baseline": EXP_141_BASELINE,
        "warmup_memory_summary": mem_summary,
        "by_cohort": results_by_cohort,
        "combined": {
            "n_questions": len(all_questions),
            "accuracy_static": round(acc_static_all, 4),
            "accuracy_memory_v2": round(acc_memory_all, 4),
            "delta": round(delta_all, 4),
            "delta_vs_exp141": round(delta_vs_141, 4),
            "avg_constraints_static": round(avg_c_static_all, 2),
            "avg_constraints_memory": round(avg_c_memory_all, 2),
        },
        "hypotheses": {
            "negation_recall_improved": hyp_negation_recall_improved,
            "delta_vs_exp141_positive": hyp_delta_vs_141_positive,
            "combined_accuracy_above_098": hyp_combined_above_098,
        },
        "warmup_elapsed_s": round(warmup_elapsed, 3),
        "total_elapsed_s": round(total_elapsed, 3),
    }

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run experiment and save results to results/experiment_173_results.json."""
    results = run_experiment()
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
