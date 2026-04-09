#!/usr/bin/env python3
"""Experiment 88: Failure mining — discover missing constraint extractors.

**Researcher summary:**
    Analyzes verify-repair pipeline false negatives to find what TYPES of
    claims the pipeline consistently fails to catch. Generates 200 questions
    across 5 domains with known-wrong responses, runs them through the
    pipeline, identifies false negatives, categorizes the gaps, and suggests
    new extraction patterns with estimated coverage improvement.

**Detailed explanation for engineers:**
    This experiment builds on the coverage analysis from Exp 73 (which
    measured WHAT FRACTION of claims get extracted) by going deeper: for
    the claims that ARE missed, WHY are they missed? What category of
    claim is it? And what new regex/AST patterns could we add to catch them?

    The experiment proceeds in 5 phases:

    Phase 1 — Generate test data: 200 questions across 5 domains (arithmetic,
    code, logic, factual, scheduling), each with a known-wrong response that
    contains a subtle error. The wrong responses are designed to include
    various types of claims that the pipeline should but doesn't catch.

    Phase 2 — Run pipeline verification: Each wrong response goes through
    the VerifyRepairPipeline in verify-only mode. We record which ones the
    pipeline correctly flags (true positives) vs. misses (false negatives).

    Phase 3 — Failure analysis: The FailureAnalyzer categorizes each false
    negative by what types of uncovered claims were present (arithmetic_chain,
    implicit_logic, world_knowledge, code_semantics, comparison, negation).

    Phase 4 — Pattern suggestion: For the top failure categories, suggest
    new extraction patterns and test them against the false negatives to
    estimate how much coverage they'd add.

    Phase 5 — Report: Produce a JSON results file with the full analysis
    and print a summary table.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/experiment_88_failure_mining.py

Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-005
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from carnot.pipeline.mining import (
    CLAIM_CATEGORIES,
    FailureAnalyzer,
    FailureReport,
)
from carnot.pipeline.verify_repair import VerifyRepairPipeline


# ---------------------------------------------------------------------------
# 1. Question + known-wrong-response generation
# ---------------------------------------------------------------------------


def generate_arithmetic_wrong(n: int = 40, seed: int = 88) -> list[dict[str, Any]]:
    """Generate arithmetic questions with deliberately wrong responses.

    **Detailed explanation for engineers:**
        Each question has a correct answer and a plausible-but-wrong response.
        The wrong responses are crafted to contain different failure modes:
        - Off-by-one errors (75 instead of 76)
        - Intermediate step errors ("first add 3+5=9, then multiply by 2=18")
        - Implicit arithmetic without explicit "A + B = C" format
        - Comparison claims ("X is greater than Y" when it's not)
        These target the GAPS in the current extractor pipeline.
    """
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []

    for i in range(n):
        a = rng.randint(10, 200)
        b = rng.randint(10, 200)
        correct = a + b
        # Deliberately wrong by 1-5
        wrong = correct + rng.choice([-3, -2, -1, 1, 2, 3])

        if i % 4 == 0:
            # Explicit arithmetic — current pipeline SHOULD catch this.
            response = f"The answer is {a} + {b} = {wrong}."
            checker = _make_number_checker(correct)
        elif i % 4 == 1:
            # Multi-step chain — pipeline likely misses intermediate error.
            mid = a + rng.randint(1, 10)
            response = (
                f"First, we compute {a} + {rng.randint(1,10)} which gives us {mid}. "
                f"Then we add {b}, resulting in {wrong}."
            )
            checker = _make_number_checker(correct)
        elif i % 4 == 2:
            # Comparison claim — pipeline doesn't handle comparisons.
            other = correct + rng.randint(10, 50)
            response = (
                f"The sum is {wrong}. Since {wrong} is greater than {other}, "
                f"the result is significant."
            )
            # Checker: the sum should be `correct`, and wrong < other is likely
            checker = _make_number_checker(correct)
        else:
            # Implicit logic: "since X, therefore Y"
            response = (
                f"Since {a} and {b} are both positive, their sum must be "
                f"positive. Therefore the answer is {wrong}."
            )
            checker = _make_number_checker(correct)

        items.append({
            "domain": "arithmetic",
            "question": f"What is {a} + {b}?",
            "ground_truth": str(correct),
            "response": response,
            "check_answer": checker,
        })

    return items


def generate_code_wrong(n: int = 40, seed: int = 88) -> list[dict[str, Any]]:
    """Generate code questions with subtly wrong responses.

    **Detailed explanation for engineers:**
        The wrong code responses contain various error types that the current
        CodeExtractor can't catch because they're SEMANTIC, not SYNTACTIC:
        - Off-by-one in loop bounds (correct syntax, wrong behavior)
        - Wrong algorithm complexity claims
        - Missing edge case handling claims
        - Wrong return values for specific inputs
    """
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []

    code_problems = [
        {
            "question": "Write a Python function to compute factorial.",
            "wrong_response": (
                "```python\ndef factorial(n: int) -> int:\n"
                "    result = 1\n"
                "    for i in range(1, n):  # Bug: should be n+1\n"
                "        result *= i\n"
                "    return result\n```\n"
                "This function correctly computes n! in O(n) time. "
                "It never returns negative values and handles all positive integers."
            ),
            "ground_truth": "def factorial",
            "keywords": ["factorial", "range", "n+1"],
        },
        {
            "question": "Write a Python function to check if a number is prime.",
            "wrong_response": (
                "```python\ndef is_prime(n: int) -> bool:\n"
                "    if n < 2:\n        return False\n"
                "    for i in range(2, n):  # Correct but O(n), claims O(sqrt(n))\n"
                "        if n % i == 0:\n            return False\n"
                "    return True\n```\n"
                "This function runs in O(sqrt(n)) time and correctly identifies "
                "all prime numbers."
            ),
            "ground_truth": "def is_prime",
            "keywords": ["prime", "is_prime", "sqrt"],
        },
        {
            "question": "Write a Python function to reverse a string.",
            "wrong_response": (
                "```python\ndef reverse_string(s: str) -> str:\n"
                "    return s[1::-1]  # Bug: should be s[::-1]\n```\n"
                "This function reverses any string in O(1) time using Python "
                "slice notation. It handles empty strings without errors."
            ),
            "ground_truth": "def reverse",
            "keywords": ["reverse", "[::-1]"],
        },
        {
            "question": "Write a Python function to find the maximum in a list.",
            "wrong_response": (
                "```python\ndef find_max(lst: list) -> int:\n"
                "    max_val = lst[0]\n"
                "    for i in range(1, len(lst)):\n"
                "        if lst[i] >= max_val:  # >= instead of > is fine\n"
                "            max_val = lst[i]\n"
                "    return max_val\n```\n"
                "This function is the fastest possible approach at O(n). "
                "It correctly handles lists with duplicate maximum values."
            ),
            "ground_truth": "def find_max",
            "keywords": ["max", "find_max"],
        },
        {
            "question": "Write a Python function to compute Fibonacci numbers.",
            "wrong_response": (
                "```python\ndef fibonacci(n: int) -> int:\n"
                "    if n <= 1:\n        return n\n"
                "    return fibonacci(n-1) + fibonacci(n-2)\n```\n"
                "This recursive implementation runs in O(n) time and uses "
                "O(1) space. It never causes a stack overflow for reasonable inputs."
            ),
            "ground_truth": "def fibonacci",
            "keywords": ["fibonacci", "fib"],
        },
    ]

    # Cycle through problems and add variations.
    for i in range(n):
        problem = code_problems[i % len(code_problems)]
        # For code, the "wrong" part is semantic — the code has bugs or
        # wrong complexity claims that the pipeline can't detect.
        items.append({
            "domain": "code",
            "question": problem["question"],
            "ground_truth": problem["ground_truth"],
            "response": problem["wrong_response"],
            "check_answer": _make_code_checker(problem["keywords"]),
        })

    return items


def generate_logic_wrong(n: int = 40, seed: int = 88) -> list[dict[str, Any]]:
    """Generate logic questions with wrong reasoning.

    **Detailed explanation for engineers:**
        The wrong responses contain logically flawed reasoning that uses
        implicit causal language ("since", "therefore", "because") rather
        than explicit "if/then" structures the LogicExtractor can parse.
    """
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []

    logic_templates = [
        {
            "question": "All dogs are mammals. All mammals are animals. Is Rex (a dog) an animal?",
            "wrong_response": (
                "Since Rex is a dog, and dogs are a subset of mammals, "
                "therefore Rex must be a mammal. However, not all mammals "
                "are animals, so the answer is no."
            ),
            "ground_truth": "yes",
        },
        {
            "question": "If it rains, the ground gets wet. The ground is dry. Did it rain?",
            "wrong_response": (
                "Because the ground is dry, we cannot conclude anything about "
                "rain. The ground could be dry for many reasons. Therefore "
                "the answer is yes, it could have rained."
            ),
            "ground_truth": "no",
        },
        {
            "question": "All squares are rectangles. This shape is a rectangle. Is it necessarily a square?",
            "wrong_response": (
                "Since all squares are rectangles, and this shape is a rectangle, "
                "it follows that this shape must be a square. The answer is yes."
            ),
            "ground_truth": "no",
        },
        {
            "question": "No fish can fly. Tuna is a fish. Can tuna fly?",
            "wrong_response": (
                "Tuna is a fish. Since some fish have been known to glide above "
                "water (flying fish), and tuna is similar, therefore tuna can fly. "
                "The answer is yes."
            ),
            "ground_truth": "no",
        },
        {
            "question": "If A implies B, and B implies C, does A imply C?",
            "wrong_response": (
                "A implies B means whenever A is true, B is true. B implies C "
                "means whenever B is true, C is true. However, since A might "
                "imply B through a different mechanism than B implies C, "
                "transitivity does not hold. The answer is no."
            ),
            "ground_truth": "yes",
        },
    ]

    for i in range(n):
        template = logic_templates[i % len(logic_templates)]
        items.append({
            "domain": "logic",
            "question": template["question"],
            "ground_truth": template["ground_truth"],
            "response": template["wrong_response"],
            "check_answer": lambda ans, gt=template["ground_truth"]: gt in ans.lower(),
        })

    return items


def generate_factual_wrong(n: int = 40, seed: int = 88) -> list[dict[str, Any]]:
    """Generate factual questions with wrong answers containing world knowledge claims.

    **Detailed explanation for engineers:**
        Wrong responses contain plausible-sounding factual claims that require
        external knowledge to verify (years, names, properties). The pipeline
        can parse "X is Y" but can't check if the claim is actually true.
    """
    items: list[dict[str, Any]] = []

    factual_wrongs = [
        {
            "question": "What is the capital of France?",
            "wrong_response": (
                "The capital of France is Lyon. Lyon was established as the "
                "capital in 1792 during the French Revolution. It is the "
                "largest city in France with a population of 3 million."
            ),
            "ground_truth": "paris",
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "wrong_response": (
                "Romeo and Juliet was written by Christopher Marlowe in 1594. "
                "Marlowe is the author of many famous plays. He was born in "
                "Canterbury in 1564."
            ),
            "ground_truth": "shakespeare",
        },
        {
            "question": "What year did World War II end?",
            "wrong_response": (
                "World War II ended in 1944. The war concluded after the "
                "invasion of Normandy, which was the decisive battle. "
                "Therefore, 1944 is the correct answer."
            ),
            "ground_truth": "1945",
        },
        {
            "question": "What is the chemical symbol for gold?",
            "wrong_response": (
                "The chemical symbol for gold is Go. Gold is a precious metal "
                "discovered by ancient civilizations. It has an atomic number "
                "of 79."
            ),
            "ground_truth": "au",
        },
        {
            "question": "How many planets are in our solar system?",
            "wrong_response": (
                "There are 9 planets in our solar system. Since Pluto was "
                "discovered in 1930 by Clyde Tombaugh, it has been counted "
                "as the ninth planet."
            ),
            "ground_truth": "8",
        },
    ]

    for i in range(n):
        template = factual_wrongs[i % len(factual_wrongs)]
        items.append({
            "domain": "factual",
            "question": template["question"],
            "ground_truth": template["ground_truth"],
            "response": template["wrong_response"],
            "check_answer": lambda ans, gt=template["ground_truth"]: gt in ans.lower(),
        })

    return items


def generate_scheduling_wrong(n: int = 40, seed: int = 88) -> list[dict[str, Any]]:
    """Generate scheduling/ordering questions with wrong answers.

    **Detailed explanation for engineers:**
        Scheduling questions involve temporal ordering, comparisons, and
        multi-step reasoning — all areas where the pipeline has gaps.
        The wrong responses use comparison and negation language that
        the pipeline can't extract as constraints.
    """
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []

    for i in range(n):
        tasks = rng.randint(3, 6)
        hours_per_task = [rng.randint(1, 5) for _ in range(tasks)]
        total = sum(hours_per_task)
        wrong_total = total + rng.choice([-2, -1, 1, 2])

        task_desc = ", ".join(f"Task {j+1} takes {h} hours" for j, h in enumerate(hours_per_task))
        question = f"You have {tasks} tasks: {task_desc}. How many total hours?"

        if i % 3 == 0:
            # Comparison-heavy wrong response.
            response = (
                f"The total time is {wrong_total} hours. This is less than "
                f"a standard work day of 8 hours, so it can be completed in "
                f"one day. Task 1 takes the most time at {max(hours_per_task)} hours."
            )
        elif i % 3 == 1:
            # Negation-heavy wrong response.
            response = (
                f"Adding up all tasks gives {wrong_total} hours. There are "
                f"no dependencies between tasks, and none of them can be "
                f"parallelized. Without any breaks, the total is {wrong_total}."
            )
        else:
            # Chain reasoning wrong response.
            response = (
                f"First, tasks 1 and 2 take {hours_per_task[0]} + {hours_per_task[1]} "
                f"which gives us {hours_per_task[0] + hours_per_task[1]}. "
                f"Then adding the remaining tasks results in {wrong_total}."
            )

        items.append({
            "domain": "scheduling",
            "question": question,
            "ground_truth": str(total),
            "response": response,
            "check_answer": _make_number_checker(total),
        })

    return items


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _extract_number(text: str) -> float | None:
    """Pull the last number from a string."""
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if not numbers:
        return None
    try:
        val = float(numbers[-1])
        return int(val) if val == int(val) else val
    except (ValueError, OverflowError):
        return None


def _make_number_checker(expected: int | float) -> Any:
    """Create a checker that extracts the last number and compares."""
    def checker(ans: str) -> bool:
        extracted = _extract_number(ans)
        if extracted is None:
            return False
        return int(extracted) == int(expected)
    return checker


def _make_code_checker(keywords: list[str]) -> Any:
    """Create a checker that verifies code response has def + keywords."""
    def checker(ans: str) -> bool:
        if "def " not in ans:
            return False
        ans_lower = ans.lower()
        return any(kw.lower() in ans_lower for kw in keywords)
    return checker


# ---------------------------------------------------------------------------
# 2. Main experiment
# ---------------------------------------------------------------------------


def run_experiment() -> dict[str, Any]:
    """Run the full failure mining experiment.

    **Detailed explanation for engineers:**
        Generates 200 wrong-answer test cases (40 per domain), runs them
        through the VerifyRepairPipeline, identifies false negatives using
        FailureAnalyzer, and reports the results.

        The key output is:
        1. How many false negatives exist (wrong answers pipeline missed)
        2. What categories of claims are uncovered in those false negatives
        3. What new patterns would catch the most false negatives
        4. Estimated coverage improvement from adding those patterns

    Returns:
        Dict with full experiment results for JSON serialization.

    Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-005
    """
    print("=" * 70)
    print("Experiment 88: Failure Mining — Discovering Missing Extractors")
    print("=" * 70)
    print()

    t0 = time.time()

    # Phase 1: Generate test data.
    print("Phase 1: Generating 200 questions with known-wrong responses...")
    all_items: list[dict[str, Any]] = []
    all_items.extend(generate_arithmetic_wrong(40))
    all_items.extend(generate_code_wrong(40))
    all_items.extend(generate_logic_wrong(40))
    all_items.extend(generate_factual_wrong(40))
    all_items.extend(generate_scheduling_wrong(40))
    print(f"  Generated {len(all_items)} test cases across 5 domains.")
    print()

    # Phase 2: Run pipeline verification via FailureAnalyzer.
    print("Phase 2: Running pipeline verification + failure analysis...")
    pipeline = VerifyRepairPipeline()
    analyzer = FailureAnalyzer(pipeline=pipeline)

    questions = [item["question"] for item in all_items]
    responses = [item["response"] for item in all_items]
    ground_truths = [item["ground_truth"] for item in all_items]
    checkers = [item["check_answer"] for item in all_items]

    report = analyzer.analyze(questions, responses, ground_truths, checkers)
    t1 = time.time()

    # Phase 3: Print summary.
    print(f"  Total questions: {report.total_questions}")
    print(f"  Total wrong answers: {report.total_wrong}")
    print(f"  False negatives (pipeline missed): {len(report.false_negatives)}")
    print(f"  False negative rate: {report.false_negative_rate:.1%}")
    print()

    # Category breakdown.
    print("Phase 3: Uncovered claim categories in false negatives")
    print("-" * 50)
    print(f"  {'Category':<25} {'Count':>8} {'% of FN':>10}")
    print("  " + "-" * 45)
    total_fn = max(len(report.false_negatives), 1)
    for cat in CLAIM_CATEGORIES:
        count = report.category_counts.get(cat, 0)
        pct = count / total_fn * 100
        print(f"  {cat:<25} {count:>8} {pct:>9.1f}%")
    print()

    # Domain breakdown of false negatives.
    domain_fn_counts: dict[str, int] = {}
    domain_totals: dict[str, int] = {}
    for item in all_items:
        d = item["domain"]
        domain_totals[d] = domain_totals.get(d, 0) + 1
    for fn in report.false_negatives:
        # Find domain from the original items by matching question.
        for item in all_items:
            if item["question"] == fn.question:
                d = item["domain"]
                domain_fn_counts[d] = domain_fn_counts.get(d, 0) + 1
                break

    print("  Domain false negative breakdown:")
    print(f"  {'Domain':<15} {'FN':>6} {'Total':>8} {'FN Rate':>10}")
    print("  " + "-" * 40)
    for domain in sorted(domain_totals.keys()):
        fn_count = domain_fn_counts.get(domain, 0)
        total = domain_totals[domain]
        rate = fn_count / total * 100 if total > 0 else 0
        print(f"  {domain:<15} {fn_count:>6} {total:>8} {rate:>9.1f}%")
    print()

    # Phase 4: Suggested patterns.
    print("Phase 4: Suggested new extraction patterns")
    print("-" * 50)
    if report.suggested_patterns:
        for sp in report.suggested_patterns[:6]:
            print(f"  [{sp['category']}] {sp['name']}")
            print(f"    Pattern: {sp['pattern'][:60]}...")
            print(f"    Est. catch: {sp['estimated_catch_count']}/{total_fn} "
                  f"({sp['estimated_catch_rate']:.1%})")
            print()
    else:
        print("  No patterns suggested (no false negatives found).")
    print()

    # Phase 5: Test suggested patterns on false negatives.
    print("Phase 5: Pattern effectiveness test")
    print("-" * 50)
    if report.suggested_patterns:
        # How many false negatives would be caught by ANY suggested pattern?
        caught_by_any = set()
        for sp in report.suggested_patterns:
            compiled = re.compile(sp["pattern"], re.IGNORECASE)
            for j, fn in enumerate(report.false_negatives):
                if compiled.search(fn.response):
                    caught_by_any.add(j)

        coverage_improvement = len(caught_by_any) / total_fn * 100 if total_fn > 0 else 0
        print(f"  False negatives caught by ANY suggested pattern: "
              f"{len(caught_by_any)}/{len(report.false_negatives)}")
        print(f"  Estimated coverage improvement: {coverage_improvement:.1f}%")
    else:
        print("  No patterns to test.")
    print()

    elapsed = t1 - t0
    print(f"Total time: {elapsed:.2f}s")
    print("=" * 70)

    # Build JSON-serializable results.
    results = {
        "experiment": "88_failure_mining",
        "description": (
            "Analyze verify-repair pipeline false negatives to discover "
            "missing constraint extractors"
        ),
        "total_questions": report.total_questions,
        "total_wrong": report.total_wrong,
        "false_negative_count": len(report.false_negatives),
        "false_negative_rate": round(report.false_negative_rate, 4),
        "category_counts": report.category_counts,
        "domain_false_negatives": domain_fn_counts,
        "domain_totals": domain_totals,
        "suggested_patterns": [
            {
                "category": sp["category"],
                "name": sp["name"],
                "pattern": sp["pattern"],
                "description": sp["description"],
                "estimated_catch_count": sp["estimated_catch_count"],
                "estimated_catch_rate": round(sp["estimated_catch_rate"], 4),
            }
            for sp in report.suggested_patterns
        ],
        "coverage_improvement_estimate": round(
            len(caught_by_any) / total_fn * 100, 2
        ) if report.suggested_patterns and total_fn > 0 else 0.0,
        "elapsed_seconds": round(elapsed, 2),
        "spec": "REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-005",
    }

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    results = run_experiment()

    # Save results.
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "experiment_88_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
