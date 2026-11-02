#!/usr/bin/env python3
"""Experiment 886: ConstrainedDecodingPreFilter — AST-guided token masking viability.

**Goal:**
    Measure whether the ConstrainedDecodingPreFilter reduces CodeExtractor
    false-positive rate by >= 0.20 (20 pp) on 30 synthetic code generation
    outputs, validating the claim from arXiv 2508.15866.

**What this experiment does:**
    Generates 30 synthetic code samples in three categories:
    - 10 correct Python functions (type annotations, loops, return values)
    - 10 functions with semantic errors (wrong logic, off-by-one) but valid syntax
    - 10 functions with syntactic errors (missing colon, bad indent, truncated)

    Measures:
    - fp_rate_without_filter: CodeExtractor FP rate on the raw corpus
    - fp_rate_with_filter: CodeExtractor FP rate after filtering out syntactically
      broken samples (simulating what pre-filter guarantees in production)
    - fp_rate_delta: fp_rate_without - fp_rate_with (target: >= 0.20)
    - syntactically_clean_fraction: fraction of samples that parse cleanly

    honest_verdict:
    - "fp_reduction_achieved"   if fp_rate_delta >= 0.20
    - "partial_fp_reduction"    if 0.05 < fp_rate_delta < 0.20
    - "no_fp_reduction"         if fp_rate_delta <= 0.05

Spec: REQ-VERIFY-147, SCENARIO-VERIFY-175, SCENARIO-VERIFY-176
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Add scripts/ to path so ExperimentTemplate is importable.
sys.path.insert(0, os.path.dirname(__file__))

from experiment_template import ExperimentTemplate  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic corpus generators
# ---------------------------------------------------------------------------


def _make_correct_functions() -> list[str]:
    """Generate 10 syntactically and semantically correct Python functions.

    Each function has type annotations, a loop, and a return value — giving
    CodeExtractor multiple constraint-extraction opportunities.
    """
    templates = [
        # 0: sum of range
        '''\
def compute_sum(n: int) -> int:
    """Return sum of 1..n."""
    total = 0
    for i in range(n):
        total += i
    return total
''',
        # 1: product
        '''\
def compute_product(items: list) -> float:
    """Return product of all items."""
    result = 1.0
    for item in items:
        result *= item
    return result
''',
        # 2: string repeat
        '''\
def repeat_string(s: str, n: int) -> str:
    """Return s repeated n times."""
    output = ""
    for _ in range(n):
        output += s
    return output
''',
        # 3: max in list
        '''\
def find_max(values: list) -> float:
    """Return the maximum value in values."""
    best = float("-inf")
    for v in values:
        if v > best:
            best = v
    return best
''',
        # 4: count occurrences
        '''\
def count_occurrences(items: list, target: int) -> int:
    """Count how many times target appears in items."""
    count = 0
    for item in items:
        if item == target:
            count += 1
    return count
''',
        # 5: flatten list
        '''\
def flatten(nested: list) -> list:
    """Flatten one level of nesting."""
    result = []
    for sublist in nested:
        for item in sublist:
            result.append(item)
    return result
''',
        # 6: running average
        '''\
def running_average(values: list) -> list:
    """Return list of running averages."""
    result = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        result.append(total / (i + 1))
    return result
''',
        # 7: reverse string
        '''\
def reverse_string(s: str) -> str:
    """Return the string reversed character by character."""
    result = ""
    for ch in s:
        result = ch + result
    return result
''',
        # 8: clamp
        '''\
def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value
''',
        # 9: is palindrome
        '''\
def is_palindrome(s: str) -> bool:
    """Return True if s reads the same forwards and backwards."""
    for i in range(len(s) // 2):
        if s[i] != s[len(s) - 1 - i]:
            return False
    return True
''',
    ]
    return templates


def _make_semantic_error_functions() -> list[str]:
    """Generate 10 functions with semantic errors but valid syntax.

    These have off-by-one errors, wrong operators, or incorrect logic.
    They must parse cleanly via ast.parse (no SyntaxError).
    CodeExtractor should still extract constraints from them — no FP here.
    """
    templates = [
        # Off-by-one in range (skips last element)
        """\
def compute_sum_buggy(n: int) -> int:
    total = 0
    for i in range(n - 1):
        total += i
    return total
""",
        # Wrong operator (subtraction instead of addition)
        """\
def compute_product_buggy(items: list) -> float:
    result = 1.0
    for item in items:
        result -= item
    return result
""",
        # Returns wrong type (int instead of str)
        """\
def repeat_string_buggy(s: str, n: int) -> str:
    output = 0
    for _ in range(n):
        output += 1
    return output
""",
        # Wrong comparison (< instead of >)
        """\
def find_max_buggy(values: list) -> float:
    best = float("inf")
    for v in values:
        if v < best:
            best = v
    return best
""",
        # Off-by-one in divisor
        """\
def running_average_buggy(values: list) -> list:
    result = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        result.append(total / (i + 2))
    return result
""",
        # Missing initialization reset
        """\
def count_occurrences_buggy(items: list, target: int) -> int:
    count = 1
    for item in items:
        if item == target:
            count += 1
    return count
""",
        # Appends to wrong variable
        """\
def flatten_buggy(nested: list) -> list:
    result = []
    other = []
    for sublist in nested:
        for item in sublist:
            other.append(item)
    return result
""",
        # Builds reverse incorrectly
        """\
def reverse_string_buggy(s: str) -> str:
    result = ""
    for ch in s:
        result = result + ch
    return result
""",
        # Clamp logic inverted
        """\
def clamp_buggy(value: float, lo: float, hi: float) -> float:
    if value > lo:
        return lo
    if value < hi:
        return hi
    return value
""",
        # Palindrome check uses wrong index
        """\
def is_palindrome_buggy(s: str) -> bool:
    for i in range(len(s) // 2):
        if s[i] != s[i]:
            return False
    return True
""",
    ]
    return templates


def _make_syntactic_error_functions() -> list[str]:
    """Generate 10 functions with irrecoverable syntax errors.

    These simulate outputs from an LLM that emitted broken Python.
    ast.parse fails on all of them — CodeExtractor returns empty (FP).
    """
    templates = [
        # Missing colon after def
        """\
def compute_sum_broken(n: int) -> int
    total = 0
    for i in range(n):
        total += i
    return total
""",
        # Bad indentation (dedent doesn't match)
        """\
def compute_product_broken(items: list) -> float:
    result = 1.0
    for item in items:
      result *= item
        result += 0
    return result
""",
        # Missing colon after for
        """\
def repeat_string_broken(s: str, n: int) -> str:
    output = ""
    for _ in range(n)
        output += s
    return output
""",
        # Unterminated string literal in the middle
        """\
def find_max_broken(values: list) -> float:
    best = float("-inf
    for v in values:
        if v > best:
            best = v
    return best
""",
        # Invalid keyword usage
        """\
def count_broken(items: list, target: int) -> int:
    count = 0
    foreach item in items:
        if item == target:
            count += 1
    return count
""",
        # Wrong indentation at function body start
        """\
def flatten_broken(nested: list) -> list:
result = []
for sublist in nested:
    for item in sublist:
        result.append(item)
return result
""",
        # Missing closing parenthesis in mid-body call
        """\
def running_average_broken(values: list) -> list:
    result = []
    total = 0.0
    for i, v in enumerate(values:
        total += v
        result.append(total / (i + 1))
    return result
""",
        # Double assignment operator
        """\
def reverse_broken(s: str) -> str:
    result == ""
    for ch in s:
        result = ch + result
    return result
""",
        # Mismatched bracket
        """\
def clamp_broken(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo]
    if value > hi:
        return hi
    return value
""",
        # Truncated mid-expression (simulates LLM stopping early)
        """\
def is_palindrome_broken(s: str) -> bool:
    for i in range(len(s
""",
    ]
    return templates


# ---------------------------------------------------------------------------
# Main experiment logic
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Exp 886: measure FP rate reduction from ConstrainedDecodingPreFilter."""
    import ast

    tmpl = ExperimentTemplate(
        886,
        "ConstrainedDecodingPreFilter — AST-guided token masking FP-rate measurement",
        "results/experiment_886_constrained_decoding_prefilter.json",
        requires_gpu=False,
    )
    tmpl.setup()

    # Import the new module we just built.
    from carnot.pipeline.constrained_decoding import (  # noqa: E402
        ASTValidator,
        ConstrainedDecodingPreFilter,
    )

    # -----------------------------------------------------------------------
    # Build the 30-sample corpus.
    # -----------------------------------------------------------------------
    correct_samples = _make_correct_functions()  # 10 correct
    semantic_samples = _make_semantic_error_functions()  # 10 semantic-error (syntax OK)
    syntactic_samples = _make_syntactic_error_functions()  # 10 syntax-broken

    all_samples = correct_samples + semantic_samples + syntactic_samples
    assert len(all_samples) == 30

    # -----------------------------------------------------------------------
    # Step 1: Measure syntactically_clean_fraction.
    # -----------------------------------------------------------------------
    validator = ASTValidator()
    clean_count = sum(1 for s in all_samples if validator.is_recoverable_partial(s))
    syntactically_clean_fraction = clean_count / len(all_samples)

    # -----------------------------------------------------------------------
    # Step 2: Measure fp_rate_without_filter.
    # All 30 samples fed to measure_fp_rate without pre-filtering.
    # The syntactic_samples will cause FPs (valid code with funcs that
    # CodeExtractor can't extract from due to parse failure).
    # -----------------------------------------------------------------------
    pre_filter = ConstrainedDecodingPreFilter(validator)
    fp_rate_without_filter = pre_filter.measure_fp_rate(all_samples)

    # -----------------------------------------------------------------------
    # Step 3: Measure fp_rate_with_filter.
    # Simulate what the pre-filter guarantees: only pass syntactically valid
    # samples through to CodeExtractor. Filter out syntax-broken inputs.
    # -----------------------------------------------------------------------
    filtered_samples = [s for s in all_samples if validator.is_recoverable_partial(s)]
    fp_rate_with_filter = pre_filter.measure_fp_rate(filtered_samples)

    # -----------------------------------------------------------------------
    # Step 4: Compute delta and determine verdict.
    # -----------------------------------------------------------------------
    fp_rate_delta = fp_rate_without_filter - fp_rate_with_filter

    if fp_rate_delta >= 0.20:
        honest_verdict = "fp_reduction_achieved"
    elif fp_rate_delta > 0.05:
        honest_verdict = "partial_fp_reduction"
    else:
        honest_verdict = "no_fp_reduction"

    # -----------------------------------------------------------------------
    # Step 5: Write deliverable.
    # -----------------------------------------------------------------------
    artifact = tmpl.build_result(
        {
            "n_correct_samples": len(correct_samples),
            "n_semantic_error_samples": len(semantic_samples),
            "n_syntactic_error_samples": len(syntactic_samples),
            "n_total_samples": len(all_samples),
            "syntactically_clean_fraction": round(float(syntactically_clean_fraction), 4),
            "fp_rate_without_filter": round(float(fp_rate_without_filter), 4),
            "fp_rate_with_filter": round(float(fp_rate_with_filter), 4),
            "fp_rate_delta": round(float(fp_rate_delta), 4),
            "n_filtered_samples": len(filtered_samples),
        },
        status="success",
        decision_class="detect",
        honest_verdict=honest_verdict,
    )

    import json

    os.makedirs("results", exist_ok=True)
    with open("results/experiment_886_constrained_decoding_prefilter.json", "w") as f:
        json.dump(artifact, f, indent=2)

    print(
        f"fp_rate_without={fp_rate_without_filter:.4f}  "
        f"fp_rate_with={fp_rate_with_filter:.4f}  "
        f"fp_rate_delta={fp_rate_delta:.4f}  "
        f"honest_verdict={honest_verdict}"
    )

    tmpl.assert_deliverable_written()


if __name__ == "__main__":
    main()
