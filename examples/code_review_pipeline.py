#!/usr/bin/env python3
"""Verify LLM-generated Python code using Carnot's code constraint extractor.

This example shows how to check LLM-generated code for structural issues:
- Undefined variables (used before assignment)
- Type annotation consistency
- Return type mismatches
- Loop bound verification

Use case: An LLM generates a Python function in response to a coding
question. Before executing or shipping that code, run it through Carnot
to catch common mistakes that the LLM might introduce.

Usage:
    JAX_PLATFORMS=cpu python examples/code_review_pipeline.py
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from carnot.pipeline import CodeExtractor, VerifyRepairPipeline
    except ImportError:
        print("ERROR: carnot is not installed. Run: pip install -e '.[dev]'")
        return 1

    # --- Example 1: Good code passes verification ---
    print("=" * 60)
    print("Example 1: Well-formed code passes all checks")
    print("=" * 60)

    good_code = '''```python
def factorial(n: int) -> int:
    """Return n factorial."""
    result = 1
    for i in range(2, n + 1):
        result = result * i
    return result
```'''

    pipeline = VerifyRepairPipeline(domains=["code"])
    result = pipeline.verify(
        question="Write a factorial function",
        response=good_code,
        domain="code",
    )

    print(f"  Verified: {result.verified}")
    print(f"  Constraints found: {len(result.constraints)}")
    for c in result.constraints:
        satisfied = c.metadata.get("satisfied", True)
        icon = "PASS" if satisfied is not False else "FAIL"
        print(f"    [{icon}] {c.description}")

    # --- Example 2: Buggy code with undefined variable ---
    print()
    print("=" * 60)
    print("Example 2: Code with undefined variable caught")
    print("=" * 60)

    buggy_code = '''```python
def compute_average(numbers: list) -> float:
    total = sum(numbers)
    return total / count
```'''

    result = pipeline.verify(
        question="Write a function to compute the average",
        response=buggy_code,
        domain="code",
    )

    print(f"  Verified: {result.verified}")
    print(f"  Violations: {len(result.violations)}")
    for v in result.violations:
        print(f"    FAIL: {v.description}")

    # --- Example 3: Return type mismatch ---
    print()
    print("=" * 60)
    print("Example 3: Return type mismatch detected")
    print("=" * 60)

    type_mismatch_code = '''```python
def get_name(user_id: int) -> int:
    return "Alice"
```'''

    result = pipeline.verify(
        question="Write a function that returns a user name",
        response=type_mismatch_code,
        domain="code",
    )

    print(f"  Verified: {result.verified}")
    for c in result.constraints:
        satisfied = c.metadata.get("satisfied", True)
        if satisfied is False:
            print(f"    FAIL: {c.description}")
        else:
            print(f"    INFO: {c.description}")

    # --- Example 4: Using CodeExtractor directly ---
    print()
    print("=" * 60)
    print("Example 4: Using CodeExtractor directly for fine-grained control")
    print("=" * 60)

    extractor = CodeExtractor()
    raw_code = """
def process_items(items: list) -> int:
    total = 0
    for i in range(len(items)):
        total += items[i]
    return total
"""

    constraints = extractor.extract(raw_code, domain="code")
    print(f"  Constraints extracted: {len(constraints)}")
    for c in constraints:
        print(f"    [{c.constraint_type}] {c.description}")
        if c.metadata.get("kind") == "loop_bound":
            print(f"      Variable '{c.metadata['variable']}' "
                  f"bounded by {c.metadata.get('lower', '?')} <= "
                  f"{c.metadata['variable']} < {c.metadata.get('upper_expr', '?')}")

    print()
    print("Done. All examples completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
