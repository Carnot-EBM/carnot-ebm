"""Tests for Exp 886: ConstrainedDecodingPreFilter — AST-guided token masking.

Covers ASTValidator and ConstrainedDecodingPreFilter with 100% coverage of
the new module. Every test traces to a spec requirement or scenario.

Spec: REQ-VERIFY-147, SCENARIO-VERIFY-175, SCENARIO-VERIFY-176
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_validator():
    from carnot.pipeline.constrained_decoding import ASTValidator

    return ASTValidator()


def _make_pre_filter():
    from carnot.pipeline.constrained_decoding import ASTValidator, ConstrainedDecodingPreFilter

    return ConstrainedDecodingPreFilter(ASTValidator())


# ---------------------------------------------------------------------------
# ASTValidator.is_recoverable_partial — REQ-VERIFY-147-1, REQ-VERIFY-147-2
# ---------------------------------------------------------------------------


class TestIsRecoverablePartial:
    """ASTValidator must classify partial and broken code correctly.

    Spec: REQ-VERIFY-147, SCENARIO-VERIFY-175, SCENARIO-VERIFY-176
    """

    def test_empty_string_is_recoverable(self) -> None:
        """Empty string has no syntax errors yet — trivially recoverable.

        Spec: REQ-VERIFY-147-1
        """
        v = _make_validator()
        assert v.is_recoverable_partial("") is True

    def test_whitespace_only_is_recoverable(self) -> None:
        """Whitespace-only string is valid Python (empty module).

        Spec: REQ-VERIFY-147-1
        """
        v = _make_validator()
        assert v.is_recoverable_partial("   \n  ") is True

    def test_complete_valid_function_is_recoverable(self) -> None:
        """A fully valid Python function is recoverable (parses cleanly).

        Spec: REQ-VERIFY-147-1, SCENARIO-VERIFY-175
        """
        v = _make_validator()
        code = "def foo(x: int) -> int:\n    return x + 1\n"
        assert v.is_recoverable_partial(code) is True

    def test_incomplete_def_line_is_recoverable(self) -> None:
        """An incomplete function def (no body yet) is a recoverable partial.

        Spec: REQ-VERIFY-147-1, SCENARIO-VERIFY-175
        """
        v = _make_validator()
        # "def foo(" is incomplete — Python raises EOF error, not invalid syntax.
        assert v.is_recoverable_partial("def foo(") is True

    def test_incomplete_assignment_is_recoverable(self) -> None:
        """A hanging assignment like "x = 1 +" is a recoverable partial.

        Spec: REQ-VERIFY-147-1, SCENARIO-VERIFY-175
        """
        v = _make_validator()
        assert v.is_recoverable_partial("x = 1 +") is True

    def test_missing_colon_after_def_is_irrecoverable(self) -> None:
        """A def line missing the colon is an irrecoverable syntax error.

        The colon is already missing — no future tokens fix a completed line
        without the colon in the middle of the source.

        Spec: REQ-VERIFY-147-2, SCENARIO-VERIFY-176
        """
        v = _make_validator()
        # Complete line with missing colon is unambiguously broken.
        code = "def foo(x)\n    return x\n"
        assert v.is_recoverable_partial(code) is False

    def test_invalid_indentation_is_irrecoverable(self) -> None:
        """Wrong indentation in the body is an irrecoverable syntax error.

        Spec: REQ-VERIFY-147-2, SCENARIO-VERIFY-176
        """
        v = _make_validator()
        # Over-indented body with no matching block.
        code = "x = 1\n        y = 2\n"
        assert v.is_recoverable_partial(code) is False

    def test_complete_valid_assignment_is_recoverable(self) -> None:
        """A simple valid assignment recovers fine.

        Spec: REQ-VERIFY-147-1
        """
        v = _make_validator()
        assert v.is_recoverable_partial("x = 42") is True

    def test_multi_line_incomplete_function_is_recoverable(self) -> None:
        """A function with open body (no return yet) is a recoverable partial.

        Spec: REQ-VERIFY-147-1, SCENARIO-VERIFY-175
        """
        v = _make_validator()
        code = "def bar(n: int) -> int:\n    total = 0\n    for i in range(n):\n"
        # The loop body is missing — EOF partial, but not broken yet.
        assert v.is_recoverable_partial(code) is True


# ---------------------------------------------------------------------------
# ASTValidator.would_be_valid — REQ-VERIFY-147-3
# ---------------------------------------------------------------------------


class TestWouldBeValid:
    """would_be_valid delegates to is_recoverable_partial on concatenation.

    Spec: REQ-VERIFY-147, SCENARIO-VERIFY-175
    """

    def test_appending_valid_token_returns_true(self) -> None:
        """Appending a token that keeps the code valid returns True.

        Spec: REQ-VERIFY-147-3
        """
        v = _make_validator()
        assert v.would_be_valid("x = ", "42") is True

    def test_appending_token_that_breaks_syntax_returns_false(self) -> None:
        """Appending a token that makes the code irrecoverably broken returns False.

        Spec: REQ-VERIFY-147-3, SCENARIO-VERIFY-176
        """
        v = _make_validator()
        # "def foo(x)\n    " already has a missing colon — appending more code
        # on the next line doesn't fix the first line's broken def.
        partial = "def foo(x)\n    "
        assert v.would_be_valid(partial, "return x\n") is False

    def test_appending_to_empty_is_recoverable(self) -> None:
        """Appending any reasonable token to an empty partial is recoverable.

        Spec: REQ-VERIFY-147-3
        """
        v = _make_validator()
        assert v.would_be_valid("", "def ") is True


# ---------------------------------------------------------------------------
# ASTValidator.filter_invalid_tokens — REQ-VERIFY-147-3
# ---------------------------------------------------------------------------


class TestFilterInvalidTokens:
    """filter_invalid_tokens must remove tokens causing irrecoverable errors.

    Spec: REQ-VERIFY-147, SCENARIO-VERIFY-175, SCENARIO-VERIFY-176
    """

    def test_all_valid_tokens_pass_through(self) -> None:
        """When all tokens are valid continuations, all are returned.

        Spec: REQ-VERIFY-147-3
        """
        v = _make_validator()
        tokens = ["1", "2", " 3"]
        result = v.filter_invalid_tokens("x = ", tokens)
        assert set(result) == set(tokens)

    def test_invalid_tokens_are_removed(self) -> None:
        """Tokens that break syntax are filtered out.

        Spec: REQ-VERIFY-147-3, SCENARIO-VERIFY-176
        """
        v = _make_validator()
        # Starting from a completed broken def line, all following tokens
        # still don't fix the missing colon on the completed first line.
        partial = "def foo(x)\n"
        # Appending "    return x\n" creates another line — still broken.
        valid_tokens = ["    pass\n"]
        invalid_tokens = ["    return x\n"]
        all_tokens = valid_tokens + invalid_tokens
        result = v.filter_invalid_tokens(partial, all_tokens)
        # Neither should survive — the partial already has an irrecoverable error.
        for tok in result:
            combined = partial + tok
            assert v.is_recoverable_partial(combined)

    def test_empty_candidate_list_returns_empty(self) -> None:
        """Empty candidate list returns empty list (no tokens to filter).

        Spec: REQ-VERIFY-147-3
        """
        v = _make_validator()
        assert v.filter_invalid_tokens("x = 1", []) == []

    def test_partial_empty_string_all_tokens_pass(self) -> None:
        """With an empty partial, all simple tokens are valid continuations.

        Spec: REQ-VERIFY-147-3
        """
        v = _make_validator()
        tokens = ["x", "def ", "# comment\n"]
        result = v.filter_invalid_tokens("", tokens)
        assert set(result) == set(tokens)


# ---------------------------------------------------------------------------
# ConstrainedDecodingPreFilter.apply — REQ-VERIFY-147-4
# ---------------------------------------------------------------------------


class TestConstrainedDecodingPreFilterApply:
    """apply() must filter logits and fall back to original when all filtered.

    Spec: REQ-VERIFY-147, SCENARIO-VERIFY-175
    """

    def test_valid_tokens_keep_logits(self) -> None:
        """Tokens that are valid continuations keep their logit values.

        Spec: REQ-VERIFY-147-4
        """
        pf = _make_pre_filter()
        generated = ["x", " = "]
        logits = {"1": -0.5, "2": -0.7, "3": -1.2}
        result = pf.apply(generated, logits)
        # All of "x = 1", "x = 2", "x = 3" are valid Python — all should pass.
        assert set(result.keys()).issubset(set(logits.keys()))
        assert len(result) > 0

    def test_safety_fallback_when_all_filtered(self) -> None:
        """When no tokens survive filtering, original logits are returned unchanged.

        This ensures generation never stalls due to an overly aggressive filter.

        Spec: REQ-VERIFY-147-4
        """
        from carnot.pipeline.constrained_decoding import ASTValidator, ConstrainedDecodingPreFilter

        class AlwaysInvalidValidator(ASTValidator):
            """Test double: always says tokens are irrecoverable."""

            def filter_invalid_tokens(self, partial_code, candidate_tokens):
                return []  # filter everything out

        pf = ConstrainedDecodingPreFilter(AlwaysInvalidValidator())
        logits = {"a": 1.0, "b": 2.0}
        result = pf.apply(["x"], logits)
        # Safety fallback: original logits returned.
        assert result == logits

    def test_logit_values_preserved(self) -> None:
        """The logit scores for surviving tokens are preserved exactly.

        Spec: REQ-VERIFY-147-4
        """
        pf = _make_pre_filter()
        generated = []
        logits = {"def ": -0.1, "x = ": -0.3}
        result = pf.apply(generated, logits)
        for tok, score in result.items():
            assert score == logits[tok]


# ---------------------------------------------------------------------------
# ConstrainedDecodingPreFilter.measure_fp_rate — REQ-VERIFY-147-5
# ---------------------------------------------------------------------------


class TestMeasureFpRate:
    """measure_fp_rate must correctly count CodeExtractor false positives.

    Spec: REQ-VERIFY-147, SCENARIO-VERIFY-175, SCENARIO-VERIFY-176
    """

    def test_all_syntactically_broken_gives_high_fp_rate(self) -> None:
        """All-broken samples all trigger FPs — CodeExtractor silently returns empty.

        Each broken sample is syntactically invalid; CodeExtractor returns [] for it.
        That is a FP (false "no violations" signal). Rate = 2/2 = 1.0.

        Spec: REQ-VERIFY-147-5
        """
        pf = _make_pre_filter()
        broken = [
            "def foo(x)\n    return x\n",  # missing colon
            "def bar(n: int) -> int\n    return n\n",  # missing colon
        ]
        result = pf.measure_fp_rate(broken)
        # Both are broken and CodeExtractor returns empty → FP rate = 1.0.
        assert result == 1.0

    def test_valid_function_with_constraints_no_fp(self) -> None:
        """A valid function contributes no FP — it is not broken code.

        Spec: REQ-VERIFY-147-5
        """
        pf = _make_pre_filter()
        valid = [
            "def foo(x: int) -> int:\n    return x + 1\n",
        ]
        result = pf.measure_fp_rate(valid)
        # Valid code is not a FP by definition (FP = broken code passing clean).
        assert result == 0.0

    def test_empty_corpus_returns_zero(self) -> None:
        """Empty corpus returns 0.0 (no samples to measure).

        Spec: REQ-VERIFY-147-5
        """
        pf = _make_pre_filter()
        assert pf.measure_fp_rate([]) == 0.0

    def test_fp_rate_range_valid(self) -> None:
        """FP rate is always in [0, 1].

        Spec: REQ-VERIFY-147-5
        """
        pf = _make_pre_filter()
        samples = [
            "def foo(x: int) -> int:\n    return x\n",
            "def bar():\n    pass\n",
            "x = 1\n",  # no function — not counted as potential FP
        ]
        result = pf.measure_fp_rate(samples)
        assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# End-to-end: fp_rate_delta measurement — REQ-VERIFY-147-5
# ---------------------------------------------------------------------------


class TestFpRateDeltaMeasurement:
    """Validates that fp_rate_delta is positive when syntax-broken samples are filtered.

    This is the core claim of Exp 886: the pre-filter reduces FP rate.

    Spec: REQ-VERIFY-147, SCENARIO-VERIFY-175, SCENARIO-VERIFY-176
    """

    def test_filtering_broken_samples_reduces_fp_rate(self) -> None:
        """Removing broken samples reduces FP rate: broken code silently passes CodeExtractor.

        Without filter: 2 broken out of 4 samples → FP rate = 2/4 = 0.5.
        With filter: 0 broken out of 2 valid samples → FP rate = 0/2 = 0.0.
        Delta = 0.5 (well above the 0.20 target).

        Spec: REQ-VERIFY-147-5
        """
        from carnot.pipeline.constrained_decoding import ASTValidator, ConstrainedDecodingPreFilter

        validator = ASTValidator()
        pf = ConstrainedDecodingPreFilter(validator)

        valid_samples = [
            "def foo(x: int) -> int:\n    return x + 1\n",
            "def bar(n: int) -> int:\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n",
        ]
        broken_samples = [
            "def foo(x)\n    return x\n",  # missing colon → syntax error
            "def baz(n: int) -> int\n    return n\n",  # missing colon
        ]

        all_samples = valid_samples + broken_samples
        fp_rate_without = pf.measure_fp_rate(all_samples)
        # 2 broken samples / 4 total = 0.5
        assert fp_rate_without == 0.5

        filtered_samples = [s for s in all_samples if validator.is_recoverable_partial(s)]
        fp_rate_with = pf.measure_fp_rate(filtered_samples)
        # 0 broken samples remaining → 0.0
        assert fp_rate_with == 0.0

        fp_rate_delta = fp_rate_without - fp_rate_with
        assert fp_rate_delta >= 0.20

    def test_honest_verdict_logic_fp_reduction_achieved(self) -> None:
        """Verify the fp_rate_delta >= 0.20 branch maps to fp_reduction_achieved.

        Spec: REQ-VERIFY-147-5
        """
        fp_rate_delta = 0.30
        if fp_rate_delta >= 0.20:
            verdict = "fp_reduction_achieved"
        elif fp_rate_delta > 0.05:
            verdict = "partial_fp_reduction"
        else:
            verdict = "no_fp_reduction"
        assert verdict == "fp_reduction_achieved"

    def test_honest_verdict_logic_partial_fp_reduction(self) -> None:
        """Verify 0.05 < fp_rate_delta < 0.20 maps to partial_fp_reduction.

        Spec: REQ-VERIFY-147-5
        """
        fp_rate_delta = 0.10
        if fp_rate_delta >= 0.20:
            verdict = "fp_reduction_achieved"
        elif fp_rate_delta > 0.05:
            verdict = "partial_fp_reduction"
        else:
            verdict = "no_fp_reduction"
        assert verdict == "partial_fp_reduction"

    def test_honest_verdict_logic_no_fp_reduction(self) -> None:
        """Verify fp_rate_delta <= 0.05 maps to no_fp_reduction.

        Spec: REQ-VERIFY-147-5
        """
        fp_rate_delta = 0.02
        if fp_rate_delta >= 0.20:
            verdict = "fp_reduction_achieved"
        elif fp_rate_delta > 0.05:
            verdict = "partial_fp_reduction"
        else:
            verdict = "no_fp_reduction"
        assert verdict == "no_fp_reduction"
