#!/usr/bin/env python3
"""Experiment 787 — S* Energy Pre-Ranking for Code Candidate Selection.

WHY THIS EXPERIMENT (arXiv 2502.14382, REQ-RANK-001, REQ-RANK-002):
    S* (arXiv 2502.14382) achieves GPT-4o-mini parity with 3B models by
    generating N=4 candidates and using distinguishing execution tests as a
    selection oracle. Each execution test is expensive: run code + collect output.
    Exp 773 showed Carnot has oracle_call_ratio=6.0 vs SETS (6x fewer oracle calls).

    This experiment tests a hypothesis: can Carnot's STATIC energy (bag-of-tokens
    embedding, no code execution) pre-rank candidates to identify the best one
    BEFORE running any execution tests? If energy correctly identifies the best
    candidate >=60% of the time, we save 30-50% of test suite runs.

    CPU-ONLY: Uses no LLM. Synthetic problems only. Deterministic and reproducible.
    For production pass@1 improvement on real benchmarks, see Exp 785 (GPU required).

HONEST VERDICT RULES:
    - "energy_prefilter_efficient"  if energy_correct_rank_pct >= 0.60
                                    (energy is reliable enough to save tests)
    - "energy_prefilter_marginal"   if 0.40 <= energy_correct_rank_pct < 0.60
                                    (energy adds some signal; marginal savings)
    - "energy_prefilter_random"     if energy_correct_rank_pct < 0.40
                                    (energy uncorrelated with correctness)

TESTS-SAVED INTERPRETATION:
    If energy_correct_rank_pct >= 0.60:
        tests_saved_pct = 1 - (1 / n_candidates)
        For n_candidates=4: tests_saved_pct = 0.75 (skip 3/4 tests on 60%+ problems)
    Otherwise: tests_saved_pct = 0.0 (not reliable enough to skip tests)

Spec: REQ-RANK-001, REQ-RANK-002, SCENARIO-RANK-001, SCENARIO-RANK-002
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from experiment_template import ExperimentTemplate  # noqa: E402
from carnot.pipeline.experiment_watchdog import ExperimentTimeoutWatchdog  # noqa: E402
from carnot.pipeline.sstar_energy_ranker import SStarConfig, SStarEnergyRanker  # noqa: E402
from carnot.verify.python_types import safe_exec_function  # noqa: E402

DELIVERABLE = "results/experiment_787_sstar_energy_ranked_code_selection.json"

tmpl = ExperimentTemplate(
    exp_id=787,
    title=(
        "S* Energy Pre-Ranking for Code Candidate Selection"
        " — arXiv 2502.14382, CPU-only synthetic benchmark"
    ),
    deliverable=DELIVERABLE,
)


# ---------------------------------------------------------------------------
# Synthetic problem corpus
# ---------------------------------------------------------------------------
# Each problem has:
#   - func_name: the function to call
#   - test_cases: list of (args_tuple, expected_output) for correctness check
#   - candidates: list of 4 code strings; index 0 is always the correct one
# ---------------------------------------------------------------------------

def _make_candidates(func_name: str, correct_body: str, wrong_bodies: list[str]) -> list[str]:
    """Build 4 candidate function strings from a function name + body variants."""
    assert len(wrong_bodies) == 3, "Must provide exactly 3 wrong candidates"  # noqa: S101

    def _wrap(body: str) -> str:
        return f"def {func_name}(a, b):\n    {body}"

    return [_wrap(correct_body)] + [_wrap(b) for b in wrong_bodies]


def _make_unary_candidates(func_name: str, correct_body: str, wrong_bodies: list[str]) -> list[str]:
    """Build 4 candidate function strings for unary (single-arg) functions."""
    assert len(wrong_bodies) == 3  # noqa: S101

    def _wrap(body: str) -> str:
        return f"def {func_name}(a):\n    {body}"

    return [_wrap(correct_body)] + [_wrap(b) for b in wrong_bodies]


SYNTHETIC_PROBLEMS: list[dict] = [
    # --- Binary arithmetic (a, b) ---
    {
        "func_name": "add",
        "test_cases": [((1, 2), 3), ((0, 0), 0), ((-1, 1), 0), ((10, 5), 15)],
        "candidates": _make_candidates(
            "add",
            "return a + b",
            ["return a - b", "return a * b", "return a + b + 1"],
        ),
    },
    {
        "func_name": "subtract",
        "test_cases": [((5, 3), 2), ((0, 0), 0), ((10, 10), 0), ((7, 2), 5)],
        "candidates": _make_candidates(
            "subtract",
            "return a - b",
            ["return a + b", "return a * b", "return b - a"],
        ),
    },
    {
        "func_name": "multiply",
        "test_cases": [((3, 4), 12), ((0, 5), 0), ((2, 2), 4), ((6, 7), 42)],
        "candidates": _make_candidates(
            "multiply",
            "return a * b",
            ["return a + b", "return a - b", "return a * b + 1"],
        ),
    },
    {
        "func_name": "integer_divide",
        "test_cases": [((10, 2), 5), ((9, 3), 3), ((7, 2), 3), ((6, 3), 2)],
        "candidates": _make_candidates(
            "integer_divide",
            "return a // b",
            ["return a / b", "return a % b", "return a + b"],
        ),
    },
    {
        "func_name": "modulo",
        "test_cases": [((10, 3), 1), ((9, 3), 0), ((7, 2), 1), ((6, 4), 2)],
        "candidates": _make_candidates(
            "modulo",
            "return a % b",
            ["return a // b", "return a - b", "return a + b"],
        ),
    },
    {
        "func_name": "max_of_two",
        "test_cases": [((3, 5), 5), ((7, 2), 7), ((4, 4), 4), ((0, -1), 0)],
        "candidates": _make_candidates(
            "max_of_two",
            "return a if a > b else b",
            ["return a if a < b else b", "return a + b", "return a"],
        ),
    },
    {
        "func_name": "min_of_two",
        "test_cases": [((3, 5), 3), ((7, 2), 2), ((4, 4), 4), ((0, -1), -1)],
        "candidates": _make_candidates(
            "min_of_two",
            "return a if a < b else b",
            ["return a if a > b else b", "return a + b", "return b"],
        ),
    },
    {
        "func_name": "power",
        "test_cases": [((2, 3), 8), ((3, 2), 9), ((2, 0), 1), ((5, 1), 5)],
        "candidates": _make_candidates(
            "power",
            "return a ** b",
            ["return a * b", "return a + b", "return a ** b + 1"],
        ),
    },
    {
        "func_name": "sum_of_squares",
        "test_cases": [((3, 4), 25), ((1, 1), 2), ((0, 5), 25), ((2, 2), 8)],
        "candidates": _make_candidates(
            "sum_of_squares",
            "return a * a + b * b",
            ["return a + b * b", "return a * a + b", "return (a + b) * (a + b)"],
        ),
    },
    {
        "func_name": "average",
        "test_cases": [((4, 6), 5.0), ((3, 7), 5.0), ((0, 0), 0.0), ((1, 3), 2.0)],
        "candidates": _make_candidates(
            "average",
            "return (a + b) / 2",
            ["return a + b / 2", "return a / 2 + b", "return (a - b) / 2"],
        ),
    },
    # --- Unary functions (single arg) ---
    {
        "func_name": "square",
        "test_cases": [((3,), 9), ((4,), 16), ((0,), 0), ((5,), 25)],
        "candidates": _make_unary_candidates(
            "square",
            "return a * a",
            ["return a + a", "return a * 2", "return a * a + 1"],
        ),
    },
    {
        "func_name": "cube",
        "test_cases": [((2,), 8), ((3,), 27), ((1,), 1), ((0,), 0)],
        "candidates": _make_unary_candidates(
            "cube",
            "return a * a * a",
            ["return a * a", "return a ** 2", "return a * a * a + 1"],
        ),
    },
    {
        "func_name": "double",
        "test_cases": [((5,), 10), ((3,), 6), ((0,), 0), ((7,), 14)],
        "candidates": _make_unary_candidates(
            "double",
            "return a * 2",
            ["return a + 2", "return a / 2", "return a * 2 + 1"],
        ),
    },
    {
        "func_name": "negate",
        "test_cases": [((5,), -5), ((-3,), 3), ((0,), 0), ((7,), -7)],
        "candidates": _make_unary_candidates(
            "negate",
            "return -a",
            ["return a", "return a - 1", "return abs(a)"],
        ),
    },
    {
        "func_name": "is_positive",
        "test_cases": [((1,), True), ((-1,), False), ((0,), False), ((5,), True)],
        "candidates": _make_unary_candidates(
            "is_positive",
            "return a > 0",
            ["return a >= 0", "return a < 0", "return a != 0"],
        ),
    },
    # --- String operations (a is a string, b is an int or string) ---
    {
        "func_name": "repeat_string",
        "test_cases": [(("ab", 3), "ababab"), (("x", 2), "xx"), (("", 5), ""), (("hi", 1), "hi")],
        "candidates": _make_candidates(
            "repeat_string",
            "return a * b",
            ["return a + b", "return a * (b + 1)", "return str(a) * b"],
        ),
    },
    {
        "func_name": "string_length",
        "test_cases": [(("hello", 0), 5), (("", 0), 0), (("abc", 0), 3), (("x", 0), 1)],
        "candidates": _make_candidates(
            "string_length",
            "return len(a)",
            ["return len(a) + 1", "return len(a) - 1", "return a.count(a[0]) if a else 0"],
        ),
    },
    # --- List operations ---
    {
        "func_name": "list_sum",
        "test_cases": [
            (([1, 2, 3], 0), 6),
            (([0], 0), 0),
            (([-1, 1], 0), 0),
            (([10, 20], 0), 30),
        ],
        "candidates": _make_candidates(
            "list_sum",
            "return sum(a)",
            ["return sum(a) + 1", "return len(a)", "return max(a)"],
        ),
    },
    {
        "func_name": "list_length",
        "test_cases": [
            (([1, 2, 3], 0), 3),
            (([], 0), 0),
            (([42], 0), 1),
            (([1, 2, 3, 4], 0), 4),
        ],
        "candidates": _make_candidates(
            "list_length",
            "return len(a)",
            ["return sum(a)", "return len(a) + 1", "return len(a) - 1"],
        ),
    },
    {
        "func_name": "first_element",
        "test_cases": [
            (([1, 2, 3], 0), 1),
            (([42], 0), 42),
            (([7, 8, 9], 0), 7),
            (([0, 1], 0), 0),
        ],
        "candidates": _make_candidates(
            "first_element",
            "return a[0]",
            ["return a[-1]", "return a[1]", "return a[0] + 1"],
        ),
    },
]


def _execute_candidate(code: str, func_name: str, test_cases: list[tuple]) -> bool:
    """Return True if the candidate passes ALL test cases.

    **Detailed explanation for engineers:**
        This is the "oracle" in S* terminology — the expensive execution check
        we are trying to pre-filter with energy ranking. It runs the code on
        every test case and checks the output. Energy ranking tries to identify
        the passing candidate WITHOUT running this function.

    Args:
        code: Python source code defining the function.
        func_name: Name of the function to invoke.
        test_cases: List of (args_tuple, expected_output) pairs.

    Returns:
        True if the function returns the correct output on all test cases.
    """
    for args, expected in test_cases:
        result, error = safe_exec_function(code, func_name, args)
        if error is not None or result != expected:
            return False
    return True


def run() -> None:
    """Execute S* energy pre-ranking experiment and write deliverable artifact."""
    tmpl.setup()

    n_candidates = 4
    config = SStarConfig(n_candidates=n_candidates, energy_top_k=1)
    ranker = SStarEnergyRanker(config=config)

    n_problems = len(SYNTHETIC_PROBLEMS)
    energy_correct_rank_results: list[bool] = []
    per_problem_results: list[dict] = []

    for problem in SYNTHETIC_PROBLEMS:
        func_name: str = problem["func_name"]
        test_cases: list = problem["test_cases"]
        candidates: list[str] = problem["candidates"]

        # Compute energy for each candidate (static — no code execution).
        energies = [ranker.compute_energy(c) for c in candidates]

        # Select the lowest-energy candidate (energy-predicted best).
        energy_selected_idx = int(energies.index(min(energies)))

        # Run execution tests to find the first passing candidate.
        passing_indices = [
            i for i, c in enumerate(candidates)
            if _execute_candidate(c, func_name, test_cases)
        ]
        first_passing_idx = passing_indices[0] if passing_indices else -1

        # Did energy select the correct candidate?
        energy_correct = (energy_selected_idx == first_passing_idx)
        energy_correct_rank_results.append(energy_correct)

        per_problem_results.append({
            "func_name": func_name,
            "energies": energies,
            "energy_selected_idx": energy_selected_idx,
            "first_passing_idx": first_passing_idx,
            "energy_correct": energy_correct,
        })

    # Compute aggregate metrics.
    energy_correct_rank_pct = sum(energy_correct_rank_results) / n_problems

    # tests_saved_pct: how many execution tests we'd skip if we trust energy.
    # We only claim savings if energy is reliable (>=60% correct rank).
    if energy_correct_rank_pct >= 0.60:
        tests_saved_pct = 1.0 - (1.0 / n_candidates)
    else:
        tests_saved_pct = 0.0

    # Honest verdict.
    if energy_correct_rank_pct >= 0.60:
        honest_verdict = "energy_prefilter_efficient"
    elif energy_correct_rank_pct >= 0.40:
        honest_verdict = "energy_prefilter_marginal"
    else:
        honest_verdict = "energy_prefilter_random"

    artifact = tmpl.build_result(
        {
            "n_problems": n_problems,
            "n_candidates": n_candidates,
            "energy_correct_rank_pct": round(energy_correct_rank_pct, 4),
            "tests_saved_pct": round(tests_saved_pct, 4),
            "honest_verdict": honest_verdict,
            "per_problem": per_problem_results,
            "reference": "arXiv 2502.14382 S* test-time scaling",
            "note": (
                "CPU-only static energy (bag-of-tokens L1 norm). "
                "No code execution during energy ranking. "
                "See Exp 785 for GPU+LLM production version."
            ),
        },
        status="success",
    )

    out_path = Path(_REPO) / DELIVERABLE
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)

    print(f"Exp 787: energy_correct_rank_pct={energy_correct_rank_pct:.3f}")
    print(f"Exp 787: tests_saved_pct={tests_saved_pct:.3f}")
    print(f"Exp 787: honest_verdict={honest_verdict}")
    print(f"Exp 787: deliverable written to {out_path}")

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    watchdog = ExperimentTimeoutWatchdog(
        experiment_id=787,
        timeout_minutes=20,
        result_path=str(Path(_REPO) / DELIVERABLE),
    )
    with watchdog:
        try:
            run()
        except Exception:
            traceback.print_exc()
            sys.exit(1)
