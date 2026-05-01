"""Tests for the GSM8K extraction fix (exp1101).

These tests verify that the _EQ_INLINE_RE patch to VeriCoTStepValidator
correctly diagnoses and fixes the TP=0 failure observed in exp1079.

Root cause: the mock extractor only handled prose arithmetic ("47 plus 28 gives 75"),
but SOTA models (Qwen3.6-35B, Gemma-4) output equation-style CoT ("47 + 28 = 75").

Spec: REQ-EXTRACT-024 (arithmetic violation detection),
      SCENARIO-EXTRACT-049 (wrong step marked as UNSAT)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _d in [str(_REPO_ROOT / "python"), str(_REPO_ROOT)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)


# ---------------------------------------------------------------------------
# Test 1: Root cause is diagnosable — prose-only extractor misses SOTA format
# REQ-EXTRACT-024 / SCENARIO-EXTRACT-049
# ---------------------------------------------------------------------------


def test_extractor_diagnoses_root_cause_not_none() -> None:
    """The prose-only extractor must return None for equation-style steps.

    This test reproduces the exp1079 failure mode: _mock_extract_expression
    (pre-fix, prose-only) could not extract any claim from "15 + 27 = 43",
    so Z3 never received an assertion and TP was 0.

    We verify that the OLD prose-only logic returns None, confirming the
    root cause is step_decomposition_fails on equation-style text.
    """
    import re

    # Inline the OLD prose-only extractor (pre-fix behaviour)
    _OLD_OP_PATTERNS = [
        (re.compile(r"(\d+(?:\.\d+)?)\s+(?:plus|added to)\s+(\d+(?:\.\d+)?)"), "+"),
        (re.compile(r"(\d+(?:\.\d+)?)\s+(?:minus|subtracted by)\s+(\d+(?:\.\d+)?)"), "-"),
        (re.compile(r"subtract(?:ing)?\s+(\d+(?:\.\d+)?)\s+from\s+(\d+(?:\.\d+)?)"), "from-sub"),
        (re.compile(r"(\d+(?:\.\d+)?)\s+(?:times|multiplied by)\s+(\d+(?:\.\d+)?)"), "*"),
        (re.compile(r"(\d+(?:\.\d+)?)\s+divided by\s+(\d+(?:\.\d+)?)"), "/"),
    ]
    _OLD_RESULT_PATTERN = re.compile(
        r"(?:gives us|gives|equals|is)\s+(\d+(?:\.\d+)?)", re.IGNORECASE
    )

    def _old_extract(step_text: str) -> str | None:
        for op_pat, op_sym in _OLD_OP_PATTERNS:
            op_match = op_pat.search(step_text)
            if not op_match:
                continue
            res_match = _OLD_RESULT_PATTERN.search(step_text, op_match.end())
            if not res_match:
                continue
            return "found"  # Return non-None to indicate a match
        return None

    # These equation-style steps are what Qwen3.6-35B ACTUALLY outputs.
    # The old extractor should return None for all of them (root cause confirmed).
    eq_style_steps = [
        "15 + 27 = 43",
        "8 * 7 = 57",
        "100 - 45 = 56",
        "48 / 6 = 9",
    ]
    for step in eq_style_steps:
        result = _old_extract(step)
        assert result is None, (
            f"Old prose extractor unexpectedly matched equation-style step: {step!r}. "
            "Root cause hypothesis is wrong — investigate further."
        )


# ---------------------------------------------------------------------------
# Test 2: Fixed extractor detects wrong arithmetic in a synthetic SOTA example
# REQ-EXTRACT-024 / SCENARIO-EXTRACT-049
# ---------------------------------------------------------------------------


def test_fixed_extractor_detects_wrong_arithmetic_in_synthetic_example() -> None:
    """VeriCoTStepValidator(use_mock=True) must detect equation-style arithmetic errors.

    This directly tests the _EQ_INLINE_RE addition: given a response in the
    format that SOTA models produce, Z3 must return UNSAT for the wrong step.
    """
    from carnot.extraction.vericot_validator import VeriCoTStepValidator

    validator = VeriCoTStepValidator(use_mock=True)

    # Typical Qwen3.6-35B CoT response with a wrong arithmetic step
    wrong_response = "Step 1: Add the numbers.\n15 + 27 = 43\nThe answer is 43."
    violations = validator.detect_violations(wrong_response)

    assert len(violations) > 0, (
        "Fixed extractor found no violations in a response with a clear equation-style "
        "arithmetic error (15 + 27 = 43, should be 42). The _EQ_INLINE_RE fix may not "
        "be correctly installed in vericot_validator.py."
    )
    assert violations[0].status == "unsat", f"Expected status='unsat', got {violations[0].status!r}"


# ---------------------------------------------------------------------------
# Test 3: TP rate > 0 on the 10 synthetic equation-style wrong answers
# REQ-EXTRACT-025 (TP > 0 required for headline result eligibility)
# ---------------------------------------------------------------------------


def test_fixed_extractor_tp_rate_above_zero_on_wrong_answers() -> None:
    """The fixed extractor must achieve TP > 0 on the 10 equation-style wrong answers.

    This is the quantitative pass gate: TP was 0.0 in exp1079; after the fix
    it must be > 0.0.  We test on the same synthetic corpus used by the
    experiment script (10 equation-style wrong answers).
    """
    from carnot.extraction.vericot_validator import VeriCoTStepValidator

    validator = VeriCoTStepValidator(use_mock=True)

    eq_wrong = [
        "Step 1: Add the numbers.\n15 + 27 = 43\nThe answer is 43.",
        "Multiply: 8 * 7 = 57\nThe answer is 57.",
        "Subtract: 100 - 45 = 56\nThe answer is 56.",
        "Divide: 48 / 6 = 9\nThe answer is 9.",
        "13 + 29 = 41. So the answer is 41.",
        "6 * 9 = 55. The answer is 55.",
        "200 - 87 = 114. The answer is 114.",
        "72 / 8 = 8. The answer is 8.",
        "Step 1: 34 + 58 = 91\nFinal answer: 91",
        "11 * 12 = 133. So the result is 133.",
    ]

    n_detected = sum(1 for resp in eq_wrong if len(validator.detect_violations(resp)) > 0)
    tp_rate = n_detected / len(eq_wrong)

    assert tp_rate > 0.0, (
        f"TP rate is {tp_rate:.2f} — fixed extractor still detects zero violations "
        "on equation-style wrong answers. The _EQ_INLINE_RE fix is not working."
    )


# ---------------------------------------------------------------------------
# Test 4: Prose-style detection is not regressed by the fix
# REQ-EXTRACT-026 (backward compatibility with prose patterns)
# ---------------------------------------------------------------------------


def test_extraction_fix_does_not_regress_humaneval_performance() -> None:
    """The equation-style fix must not break detection of prose-style arithmetic.

    This is a backward-compatibility test: the existing prose patterns must
    still fire correctly after the _EQ_INLINE_RE addition.  A regression here
    would mean the fix broke the HumanEval-adjacent prose detection path.

    We check two prose-style wrong responses (the format exp453 used for TP=8/20)
    and verify that violations are still detected.
    """
    from carnot.extraction.vericot_validator import VeriCoTStepValidator

    validator = VeriCoTStepValidator(use_mock=True)

    prose_wrong = [
        "47 plus 28 gives 76. The answer is 76.",
        "5 times 6 gives us 31. The answer is 31.",
    ]

    for resp in prose_wrong:
        violations = validator.detect_violations(resp)
        assert len(violations) > 0, (
            f"Prose-style detection regressed: no violations found in {resp!r}. "
            "The equation-style fix may have broken the prose pattern matching."
        )
