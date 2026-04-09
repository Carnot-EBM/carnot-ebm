#!/usr/bin/env python3
"""Verify LLM API responses using Carnot's constraint extraction pipeline.

This example shows how to check whether an LLM's answers contain correct
arithmetic, valid logic, and consistent factual claims -- without needing
a GPU or a local model. The pipeline extracts verifiable constraints from
the response text and checks each one, returning a structured result with
an energy score and per-constraint details.

Use case: You call an LLM API (OpenAI, Anthropic, etc.) and want to
programmatically verify the response before showing it to a user or
feeding it into a downstream system.

Usage:
    JAX_PLATFORMS=cpu python examples/verify_api_responses.py
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from carnot.pipeline import VerifyRepairPipeline
    except ImportError:
        print("ERROR: carnot is not installed. Run: pip install -e '.[dev]'")
        return 1

    # --- Example 1: Arithmetic verification ---
    print("=" * 60)
    print("Example 1: Verify arithmetic in an LLM response")
    print("=" * 60)

    pipeline = VerifyRepairPipeline()

    question = "What is 47 + 28?"
    # Simulate an LLM response with a correct answer.
    good_response = "47 + 28 = 75. This is because 47 + 28 = 75."
    result = pipeline.verify(question, good_response, domain="arithmetic")

    print(f"  Question: {question}")
    print(f"  Response: {good_response}")
    print(f"  Verified: {result.verified}")
    print(f"  Constraints found: {len(result.constraints)}")
    for c in result.constraints:
        satisfied = c.metadata.get("satisfied", "N/A")
        print(f"    [{satisfied}] {c.description}")

    # Now try a response with a wrong answer.
    print()
    bad_response = "47 + 28 = 73. Simple addition gives us 73."
    result = pipeline.verify(question, bad_response, domain="arithmetic")

    print(f"  Response: {bad_response}")
    print(f"  Verified: {result.verified}")
    print(f"  Violations: {len(result.violations)}")
    for v in result.violations:
        print(f"    FAIL: {v.description}")
        print(f"          Correct answer: {v.metadata.get('correct_result')}")

    # --- Example 2: Multi-domain verification ---
    print()
    print("=" * 60)
    print("Example 2: Multi-domain verification (arithmetic + logic)")
    print("=" * 60)

    pipeline_multi = VerifyRepairPipeline(domains=["arithmetic", "logic"])

    question = "If it rains, the ground gets wet. Also, what is 10 - 3?"
    response = "If it rains, then the ground gets wet. 10 - 3 = 7."
    result = pipeline_multi.verify(question, response)

    print(f"  Question: {question}")
    print(f"  Response: {response}")
    print(f"  Verified: {result.verified}")
    print(f"  Constraints found: {len(result.constraints)}")
    for c in result.constraints:
        print(f"    [{c.constraint_type}] {c.description}")

    # --- Example 3: Using verify_and_repair without a model ---
    print()
    print("=" * 60)
    print("Example 3: Verify-and-repair (verification-only mode)")
    print("=" * 60)

    question = "What is 99 + 1?"
    response = "99 + 1 = 99"
    repair_result = pipeline.verify_and_repair(question, response, domain="arithmetic")

    print(f"  Question: {question}")
    print(f"  Response: {response}")
    print(f"  Verified: {repair_result.verified}")
    print(f"  Repaired: {repair_result.repaired}")
    print(f"  Iterations: {repair_result.iterations}")
    if repair_result.history:
        for v in repair_result.history[0].violations:
            print(f"    FAIL: {v.description}")

    print()
    print("Done. All examples completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
