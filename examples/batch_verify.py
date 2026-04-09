#!/usr/bin/env python3
"""Batch-verify a JSON file of question/answer pairs using Carnot.

Reads a JSON file where each entry has "question" and "response" fields,
runs Carnot verification on each pair, and prints a summary report showing
which responses passed and which had constraint violations.

Use case: You have a dataset of LLM outputs (e.g., from an eval run or
a production log) and want to audit them for correctness at scale.

Usage:
    # Use built-in sample data:
    JAX_PLATFORMS=cpu python examples/batch_verify.py

    # Verify your own file:
    JAX_PLATFORMS=cpu python examples/batch_verify.py my_qa_pairs.json

JSON format:
    [
      {"question": "What is 2 + 2?", "response": "2 + 2 = 4"},
      {"question": "...", "response": "..."}
    ]
"""

from __future__ import annotations

import json
import sys


# Built-in sample data so the example runs without any files.
SAMPLE_DATA = [
    {
        "question": "What is 15 + 27?",
        "response": "15 + 27 = 42",
    },
    {
        "question": "What is 100 - 37?",
        "response": "100 - 37 = 63",
    },
    {
        "question": "What is 8 + 5?",
        "response": "8 + 5 = 14",
    },
    {
        "question": "If it rains, the streets get wet. What happens when it rains?",
        "response": "If it rains, then the streets get wet.",
    },
    {
        "question": "Write a Python function to double a number",
        "response": '```python\ndef double(x: int) -> int:\n    return x * 2\n```',
    },
    {
        "question": "What is 50 + 50?",
        "response": "50 + 50 = 100. The sum of fifty and fifty is one hundred.",
    },
]


def main() -> int:
    try:
        from carnot.pipeline import VerifyRepairPipeline
    except ImportError:
        print("ERROR: carnot is not installed. Run: pip install -e '.[dev]'")
        return 1

    # Load data from file or use built-in samples.
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        try:
            with open(filepath) as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"ERROR: File not found: {filepath}")
            return 1
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in {filepath}: {e}")
            return 1
        print(f"Loaded {len(data)} entries from {filepath}")
    else:
        data = SAMPLE_DATA
        print(f"Using {len(data)} built-in sample entries (pass a JSON file as argument)")

    # Validate structure.
    for i, entry in enumerate(data):
        if "question" not in entry or "response" not in entry:
            print(f"ERROR: Entry {i} missing 'question' or 'response' key")
            return 1

    # Run verification.
    pipeline = VerifyRepairPipeline()

    print()
    print("=" * 70)
    print(f"{'#':>3}  {'Verified':>8}  {'Constraints':>11}  {'Violations':>10}  Question")
    print("-" * 70)

    n_verified = 0
    n_failed = 0
    all_violations: list[dict] = []

    for i, entry in enumerate(data):
        result = pipeline.verify(entry["question"], entry["response"])
        status = "PASS" if result.verified else "FAIL"

        if result.verified:
            n_verified += 1
        else:
            n_failed += 1
            for v in result.violations:
                all_violations.append({
                    "index": i,
                    "question": entry["question"][:50],
                    "violation": v.description,
                    "type": v.constraint_type,
                })

        question_short = entry["question"][:40]
        if len(entry["question"]) > 40:
            question_short += "..."
        print(
            f"{i:>3}  {status:>8}  {len(result.constraints):>11}  "
            f"{len(result.violations):>10}  {question_short}"
        )

    # Summary.
    print()
    print("=" * 70)
    print(f"  Total:    {len(data)}")
    print(f"  Passed:   {n_verified}")
    print(f"  Failed:   {n_failed}")
    if data:
        print(f"  Pass rate: {n_verified / len(data):.0%}")

    if all_violations:
        print()
        print("Violations detail:")
        for v in all_violations:
            print(f"  Entry {v['index']}: [{v['type']}] {v['violation']}")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
