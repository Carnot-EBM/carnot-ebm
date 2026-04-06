#!/usr/bin/env python3
"""Demo: Real code verification — LLM generates code, EBM verifies it.

Asks an LLM to implement Python functions, then runs them through:
1. Execution-based verification (type checks, test cases)
2. Learned energy model (pattern-based bug detection)

This is the path from toy SAT domains to real code verification.

Usage:
    python scripts/demo_code_verification.py
    CARNOT_API_BASE=http://localhost:8080/v1 python scripts/demo_code_verification.py
    python scripts/demo_code_verification.py --api-base $CARNOT_API_BASE
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Task definitions: (description, function_name, test_cases, expected_type)
# test_cases = [(args_tuple, expected_output), ...]
TASKS_EASY = [
    {
        "description": "Write a Python function called `fibonacci` that takes an integer n and returns the nth Fibonacci number (0-indexed). fibonacci(0)=0, fibonacci(1)=1, fibonacci(6)=8.",
        "func_name": "fibonacci",
        "test_cases": [
            ((0,), 0),
            ((1,), 1),
            ((2,), 1),
            ((5,), 5),
            ((6,), 8),
            ((10,), 55),
        ],
        "expected_type": int,
    },
    {
        "description": "Write a Python function called `is_palindrome` that takes a string and returns True if it's a palindrome (case-insensitive, ignoring spaces), False otherwise.",
        "func_name": "is_palindrome",
        "test_cases": [
            (("racecar",), True),
            (("hello",), False),
            (("A man a plan a canal Panama",), True),
            (("",), True),
            (("ab",), False),
        ],
        "expected_type": bool,
    },
    {
        "description": "Write a Python function called `flatten` that takes a nested list and returns a flat list. E.g., flatten([1, [2, [3]], 4]) returns [1, 2, 3, 4].",
        "func_name": "flatten",
        "test_cases": [
            (([1, [2, [3]], 4],), [1, 2, 3, 4]),
            (([],), []),
            (([[1, 2], [3, 4]],), [1, 2, 3, 4]),
            (([1, 2, 3],), [1, 2, 3]),
        ],
        "expected_type": list,
    },
    {
        "description": "Write a Python function called `gcd` that takes two positive integers and returns their greatest common divisor using the Euclidean algorithm.",
        "func_name": "gcd",
        "test_cases": [
            ((12, 8), 4),
            ((7, 13), 1),
            ((100, 75), 25),
            ((1, 1), 1),
            ((0, 5), 5),
        ],
        "expected_type": int,
    },
    {
        "description": "Write a Python function called `roman_to_int` that converts a Roman numeral string to an integer. E.g., roman_to_int('XIV') returns 14.",
        "func_name": "roman_to_int",
        "test_cases": [
            (("I",), 1),
            (("IV",), 4),
            (("IX",), 9),
            (("XIV",), 14),
            (("XLII",), 42),
            (("MCMXCIV",), 1994),
        ],
        "expected_type": int,
    },
]


def ask_llm_for_code(api_base: str, model: str, api_key: str, task: dict) -> str:
    """Ask the LLM to implement a function. Returns raw response text."""
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise Python programmer. Write ONLY the function "
                    "definition — no explanation, no imports, no test code. "
                    "Just the def statement and its body."
                ),
            },
            {"role": "user", "content": task["description"]},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content or ""


def extract_code(response: str) -> str:
    """Extract Python code from LLM response (handles code blocks)."""
    # Try to find a Python code block
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try generic code block
    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Assume the whole response is code
    return response.strip()


def verify_code(code: str, task: dict) -> dict:
    """Run the code through Carnot's verification pipeline."""
    from carnot.verify.python_types import build_code_energy, safe_exec_function

    func_name = task["func_name"]
    test_cases = task["test_cases"]
    expected_type = task["expected_type"]

    # Build energy function from test cases
    energy = build_code_energy(
        code, func_name, test_cases,
        expected_type=expected_type,
    )

    # Get embedding for code
    from carnot.verify.python_types import code_to_embedding

    embedding = code_to_embedding(code)

    # Verify against constraints
    result = energy.verify(embedding)

    # Also run individual test cases for detailed reporting
    test_results = []
    for args, expected in test_cases:
        actual, error = safe_exec_function(code, func_name, args)
        passed = error is None and actual == expected
        test_results.append({
            "input": args,
            "expected": expected,
            "actual": actual,
            "error": str(error) if error else None,
            "passed": passed,
        })

    n_passed = sum(1 for t in test_results if t["passed"])
    return {
        "verified": result.verdict.verified,
        "total_energy": float(result.total_energy),
        "n_passed": n_passed,
        "n_total": len(test_cases),
        "test_results": test_results,
    }


TASKS_HARD = [
    {
        "description": "Write a Python function called `eval_rpn` that evaluates a reverse Polish notation expression given as a list of strings. Operators are '+', '-', '*', '/'. Division should truncate toward zero (like int(a/b) for positive results, but -7/2 should be -3 not -4). E.g., eval_rpn(['2','1','+','3','*']) = 9.",
        "func_name": "eval_rpn",
        "test_cases": [
            ((["2", "1", "+", "3", "*"],), 9),
            ((["4", "13", "5", "/", "+"],), 6),
            ((["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"],), 22),
            ((["3"],), 3),
        ],
        "expected_type": int,
    },
    {
        "description": "Write a Python function called `longest_common_subseq` that takes two strings and returns the length of their longest common subsequence. E.g., longest_common_subseq('abcde', 'ace') = 3.",
        "func_name": "longest_common_subseq",
        "test_cases": [
            (("abcde", "ace"), 3),
            (("abc", "abc"), 3),
            (("abc", "def"), 0),
            (("", "abc"), 0),
            (("abcba", "abcbcba"), 5),
        ],
        "expected_type": int,
    },
    {
        "description": "Write a Python function called `balanced_parens` that takes an integer n and returns a list of all valid combinations of n pairs of parentheses, sorted lexicographically.",
        "func_name": "balanced_parens",
        "test_cases": [
            ((1,), ["()"]),
            ((2,), ["(())", "()()"]),
            ((3,), ["((()))", "(()())", "(())()", "()(())", "()()()"]),
            ((0,), [""]),
        ],
        "expected_type": list,
    },
    {
        "description": "Write a Python function called `matrix_rotate` that takes a 2D list (square matrix) and rotates it 90 degrees clockwise IN-PLACE. It should modify the input and return None.",
        "func_name": "matrix_rotate",
        "test_cases": [
            (([[1, 2], [3, 4]],), None),
        ],
        "expected_type": type(None),
    },
    {
        "description": "Write a Python function called `atoi` that converts a string to a 32-bit signed integer. Rules: ignore leading whitespace, optional +/- sign, read digits until non-digit. Clamp to [-2^31, 2^31-1]. Return 0 for invalid input.",
        "func_name": "atoi",
        "test_cases": [
            (("42",), 42),
            (("   -42",), -42),
            (("4193 with words",), 4193),
            (("words and 987",), 0),
            (("-91283472332",), -2147483648),
            (("",), 0),
            (("+1",), 1),
        ],
        "expected_type": int,
    },
    {
        "description": "Write a Python function called `count_inversions` that takes a list of integers and returns the number of inversions (pairs i<j where list[i]>list[j]). Must be O(n log n), not O(n^2).",
        "func_name": "count_inversions",
        "test_cases": [
            (([2, 4, 1, 3, 5],), 3),
            (([1, 2, 3, 4, 5],), 0),
            (([5, 4, 3, 2, 1],), 10),
            (([1],), 0),
            (([],), 0),
        ],
        "expected_type": int,
    },
]

TASKS = TASKS_EASY + TASKS_HARD


def main() -> int:
    parser = argparse.ArgumentParser(description="Real code verification demo")
    parser.add_argument("--api-base", default="http://localhost:8080/v1")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--api-key", default="not-needed")
    args = parser.parse_args()

    print("=" * 60)
    print("CARNOT REAL CODE VERIFICATION DEMO")
    print("LLM writes code → EBM verifies correctness")
    print(f"Model: {args.model}")
    print("=" * 60)

    results = []
    for i, task in enumerate(TASKS):
        print(f"\n{'─'*60}")
        print(f"Task {i+1}/{len(TASKS)}: {task['func_name']}")
        print(f"{'─'*60}")
        print(f"  Prompt: {task['description'][:80]}...")

        # Get code from LLM
        try:
            raw_response = ask_llm_for_code(
                args.api_base, args.model, args.api_key, task
            )
            code = extract_code(raw_response)
            print(f"\n  LLM Code:")
            for line in code.split("\n"):
                print(f"    {line}")
        except Exception as e:
            print(f"  LLM Error: {e}")
            results.append({"task": task["func_name"], "status": "llm_error"})
            continue

        # Verify
        try:
            vr = verify_code(code, task)
            print(f"\n  Verification:")
            print(f"    Tests passed: {vr['n_passed']}/{vr['n_total']}")
            print(f"    Energy: {vr['total_energy']:.4f}")
            print(f"    Verified: {vr['verified']}")

            # Show failed tests
            for tr in vr["test_results"]:
                if not tr["passed"]:
                    print(f"    FAIL: {task['func_name']}{tr['input']} "
                          f"expected {tr['expected']}, got {tr['actual']}"
                          f"{' (error: ' + tr['error'] + ')' if tr['error'] else ''}")

            results.append({
                "task": task["func_name"],
                "status": "ok",
                "n_passed": vr["n_passed"],
                "n_total": vr["n_total"],
                "verified": vr["verified"],
                "energy": vr["total_energy"],
            })
        except Exception as e:
            print(f"  Verification Error: {e}")
            import traceback
            traceback.print_exc()
            results.append({"task": task["func_name"], "status": "verify_error"})

    # Summary
    ok_results = [r for r in results if r["status"] == "ok"]
    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(ok_results)}/{len(TASKS)} tasks completed)")
    print(f"{'='*60}")

    if ok_results:
        all_correct = sum(1 for r in ok_results if r["verified"])
        total_tests = sum(r["n_total"] for r in ok_results)
        passed_tests = sum(r["n_passed"] for r in ok_results)

        print(f"  Functions fully correct: {all_correct}/{len(ok_results)}")
        print(f"  Total test cases passed: {passed_tests}/{total_tests}")
        print(f"  Average energy: {sum(r['energy'] for r in ok_results) / len(ok_results):.4f}")

        print(f"\n  Per-function:")
        for r in ok_results:
            icon = "✓" if r["verified"] else "✗"
            print(f"    {icon} {r['task']}: {r['n_passed']}/{r['n_total']} tests, energy={r['energy']:.4f}")

    print(f"\n{'='*60}")
    print("EBM verification provides deterministic correctness checks")
    print("that the LLM cannot provide about its own output.")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
