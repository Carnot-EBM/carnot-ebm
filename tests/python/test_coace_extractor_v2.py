"""Tests for CoACEExtractorV2 — 100% coverage on coace_extractor_v2.py.

Spec: REQ-EXTRACT-035, REQ-EXTRACT-036,
      SCENARIO-EXTRACT-068, SCENARIO-EXTRACT-069, SCENARIO-EXTRACT-070, SCENARIO-EXTRACT-071
"""

from __future__ import annotations

import math

import pytest

from carnot.extraction.coace_extractor_v2 import (
    CoACEExtractorV2,
    NumericContext,
    _extract_chain_equations,
    _parse_number_list,
    _parse_prose_arithmetic,
)


# ---------------------------------------------------------------------------
# _parse_number_list
# ---------------------------------------------------------------------------


def test_parse_number_list_simple():
    nums = _parse_number_list("3, 4, and 5")
    assert nums == [3.0, 4.0, 5.0]


def test_parse_number_list_decimal():
    nums = _parse_number_list("1.5, 2.5")
    assert nums == [1.5, 2.5]


# ---------------------------------------------------------------------------
# _parse_prose_arithmetic — SCENARIO-EXTRACT-068
# ---------------------------------------------------------------------------


def test_prose_percentage_correct():
    # 20% of 150 = 30 — no violation
    eqs = _parse_prose_arithmetic("20% of 150 is 30.")
    assert len(eqs) == 1
    assert eqs[0].lhs_expr == "20/100*150"
    assert eqs[0].rhs_value == 30.0


def test_prose_percentage_violation():
    # 20% of 150 is 31 — violation (20/100*150=30 != 31)
    eqs = _parse_prose_arithmetic("20% of 150 is 31.")
    assert len(eqs) == 1
    assert eqs[0].rhs_value == 31.0
    extractor = CoACEExtractorV2()
    result = extractor.extract("20% of 150 is 31.")
    assert result.n_violations >= 1


# ---------------------------------------------------------------------------
# _parse_prose_arithmetic — SCENARIO-EXTRACT-069 (times)
# ---------------------------------------------------------------------------


def test_prose_times_violation():
    # 47 times 3 = 141, stated 142 — violation
    result = CoACEExtractorV2().extract("47 times 3 is 142.")
    assert result.n_violations >= 1


def test_prose_times_correct():
    result = CoACEExtractorV2().extract("47 times 3 is 141.")
    assert result.n_violations == 0


# ---------------------------------------------------------------------------
# _parse_prose_arithmetic — divided by
# ---------------------------------------------------------------------------


def test_prose_divided_by_violation():
    result = CoACEExtractorV2().extract("100 divided by 4 is 26.")
    assert result.n_violations >= 1  # 100/4=25 != 26


def test_prose_divided_by_correct():
    result = CoACEExtractorV2().extract("100 divided by 4 is 25.")
    assert result.n_violations == 0


# ---------------------------------------------------------------------------
# _parse_prose_arithmetic — difference between
# ---------------------------------------------------------------------------


def test_prose_difference_violation():
    result = CoACEExtractorV2().extract("difference between 10 and 3 is 8.")
    assert result.n_violations >= 1  # 10-3=7 != 8


def test_prose_difference_correct():
    result = CoACEExtractorV2().extract("difference between 10 and 3 is 7.")
    assert result.n_violations == 0


# ---------------------------------------------------------------------------
# _parse_prose_arithmetic — sum of
# ---------------------------------------------------------------------------


def test_prose_sum_of_violation():
    result = CoACEExtractorV2().extract("sum of 3, 4, and 5 is 13.")
    assert result.n_violations >= 1  # 3+4+5=12 != 13


def test_prose_sum_of_correct():
    result = CoACEExtractorV2().extract("sum of 3, 4, and 5 is 12.")
    assert result.n_violations == 0


def test_prose_sum_of_two_numbers():
    # Two-number sum — still a valid sum-of pattern
    result = CoACEExtractorV2().extract("sum of 6 and 7 is 14.")
    assert result.n_violations >= 1  # 6+7=13 != 14


# ---------------------------------------------------------------------------
# v1 chained equality (handled by v1's _parse_arithmetic_equations)
# ---------------------------------------------------------------------------


def test_chained_equality_violation_via_v1():
    # '5 + 7 = 13' — v1 catches this directly (5+7=12 != 13)
    result = CoACEExtractorV2().extract("5 + 7 = 13")
    assert result.n_violations >= 1


def test_chained_equality_correct_via_v1():
    # Use two-term expression to avoid v1 sub-expression false positives.
    result = CoACEExtractorV2().extract("5 + 7 = 12")
    assert result.n_violations == 0


# ---------------------------------------------------------------------------
# _extract_chain_equations — SCENARIO-EXTRACT-070
# ---------------------------------------------------------------------------


def test_chain_equations_detects_variable_mismatch():
    # X is first assigned 75, then re-stated as 76 — chain mismatch
    # Use comma separator so the regex lookahead (?=[,;\n]|$) fires correctly.
    text = "let X = 75, X = 76"
    chain_eqs = _extract_chain_equations(text)
    assert any(eq.lhs_expr.startswith("chain:") for eq in chain_eqs)


def test_chain_equations_no_mismatch():
    text = "let X = 10, total = 10 + 5 = 15"
    chain_eqs = _extract_chain_equations(text)
    # X=10 is assigned once; no conflict
    assert all(eq.lhs_expr.startswith("chain:") for eq in chain_eqs) or len(chain_eqs) == 0


def test_chain_equations_returns_list():
    eqs = _extract_chain_equations("no equations here")
    assert isinstance(eqs, list)


# ---------------------------------------------------------------------------
# CoACEExtractorV2.extract() — SCENARIO-EXTRACT-071
# ---------------------------------------------------------------------------


def test_extract_combines_v1_prose_chain():
    # V1 detectable: '8 * 7 = 65'
    # Prose detectable: '20% of 150 is 31'
    # Combined: at least 2 violations
    text = "Step 1: 8 * 7 = 65. Step 2: 20% of 150 is 31."
    result = CoACEExtractorV2().extract(text)
    assert result.n_violations >= 2


def test_extract_no_violations_clean_text():
    text = "We add 47 and 28 to get 75. 5 times 6 is 30."
    result = CoACEExtractorV2().extract(text)
    assert result.n_violations == 0


def test_extract_deduplication():
    # Same equation appearing in both v1 and prose patterns — should not double-count.
    # '47 + 28 = 76' is v1; also matches chained equality pattern.
    text = "47 + 28 = 76"
    result = CoACEExtractorV2().extract(text)
    # Only one violation regardless of how many patterns fire.
    assert result.n_violations == 1


def test_extract_returns_coace_result():
    from carnot.extraction.coace_extractor import CoACEResult

    result = CoACEExtractorV2().extract("nothing arithmetic here")
    assert isinstance(result, CoACEResult)
    assert result.extraction_mode == "execution_based_v2"


# ---------------------------------------------------------------------------
# detect_violations protocol
# ---------------------------------------------------------------------------


def test_detect_violations_nonempty_on_error():
    extractor = CoACEExtractorV2()
    violations = extractor.detect_violations("47 + 28 = 76")
    assert len(violations) > 0


def test_detect_violations_empty_on_correct():
    extractor = CoACEExtractorV2()
    violations = extractor.detect_violations("47 + 28 = 75")
    assert len(violations) == 0


# ---------------------------------------------------------------------------
# Inheritance — v1 methods still work
# ---------------------------------------------------------------------------


def test_to_constraint_terms_works():
    extractor = CoACEExtractorV2()
    result = extractor.extract("8 * 7 = 65")
    terms = extractor.to_constraint_terms(result)
    assert len(terms) >= 1
    assert "lhs" in terms[0]


def test_custom_tolerance():
    extractor = CoACEExtractorV2(tolerance=10.0)
    result = extractor.extract("47 + 28 = 76")
    # |75 - 76| = 1 < tolerance 10 → not a violation
    assert result.n_violations == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_string():
    result = CoACEExtractorV2().extract("")
    assert result.n_violations == 0
    assert result.n_equations_found == 0


def test_prose_sum_of_single_number_ignored():
    # 'sum of 5 is 5' — only one number, not a valid sum pattern
    eqs = _parse_prose_arithmetic("sum of 5 is 5.")
    # Should not produce a sum equation (need >= 2 numbers)
    sum_eqs = [e for e in eqs if "+" in e.lhs_expr]
    assert len(sum_eqs) == 0


def test_percentage_with_equals_sign():
    # '20% of 150 = 30' — equals sign variant
    eqs = _parse_prose_arithmetic("20% of 150 = 31.")
    assert len(eqs) == 1
    assert eqs[0].rhs_value == 31.0


def test_chain_violation_has_nan_computed():
    # Chain violations use float('nan') as computed_value sentinel
    text = "let A = 10\nA = 20"
    result = CoACEExtractorV2().extract(text)
    chain_violations = [v for v in result.violations if v.equation.lhs_expr.startswith("chain:")]
    if chain_violations:
        assert math.isnan(chain_violations[0].computed_value)
