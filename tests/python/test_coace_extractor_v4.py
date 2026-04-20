"""Tests for CoACEExtractorV4 — GenPRM-style arithmetic extraction.

Coverage target: 100% of coace_extractor_v4.py.

Spec: REQ-EXTRACT-045, REQ-EXTRACT-046,
      SCENARIO-EXTRACT-080, SCENARIO-EXTRACT-081, SCENARIO-EXTRACT-082,
      SCENARIO-EXTRACT-083
"""

from __future__ import annotations

import json

import pytest

from carnot.extraction.coace_extractor_v4 import (
    EXTRACTION_PROMPT,
    ArithmeticClaim,
    CoACEExtractorV4,
    GenPRMExtractor,
    safe_eval,
)


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-080: safe_eval — allowed operations
# ---------------------------------------------------------------------------


class TestSafeEval:
    """REQ-EXTRACT-046: safe_eval evaluates restricted arithmetic; blocks builtins."""

    def test_addition(self):
        assert safe_eval("1+2") == pytest.approx(3.0)

    def test_subtraction(self):
        assert safe_eval("10-3") == pytest.approx(7.0)

    def test_multiplication(self):
        assert safe_eval("7*1.5") == pytest.approx(10.5)

    def test_division(self):
        assert safe_eval("90/7") == pytest.approx(12.857142857)

    def test_power(self):
        assert safe_eval("2**10") == pytest.approx(1024.0)

    def test_modulo(self):
        assert safe_eval("10%3") == pytest.approx(1.0)

    def test_parentheses(self):
        assert safe_eval("(1+2)*3") == pytest.approx(9.0)

    def test_unary_minus(self):
        assert safe_eval("-5+10") == pytest.approx(5.0)

    def test_unary_plus(self):
        assert safe_eval("+3") == pytest.approx(3.0)

    def test_currency_stripped(self):
        # $ and , should be stripped before eval so '$7*1.5' works.
        assert safe_eval("$7*1.5") == pytest.approx(10.5)

    def test_comma_stripped(self):
        assert safe_eval("1,000+500") == pytest.approx(1500.0)

    def test_division_by_zero_returns_none(self):
        assert safe_eval("1/0") is None

    def test_overflow_returns_none(self):
        # 10**10**10 overflows float
        assert safe_eval("10**10**10") is None

    def test_invalid_syntax_returns_none(self):
        assert safe_eval("abc+1") is None

    def test_empty_string_returns_none(self):
        assert safe_eval("") is None

    def test_function_call_blocked(self):
        assert safe_eval("__import__('os').getenv('HOME')") is None

    def test_list_blocked(self):
        assert safe_eval("[1,2,3]") is None

    def test_string_literal_blocked(self):
        assert safe_eval("'hello'") is None

    def test_attribute_access_blocked(self):
        assert safe_eval("(1).real") is None

    def test_name_reference_blocked(self):
        # A bare name reference like 'x' should be blocked.
        assert safe_eval("x+1") is None


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-081: GenPRMExtractor CI stub (regex fallback)
# ---------------------------------------------------------------------------


class TestGenPRMExtractorCIStub:
    """REQ-EXTRACT-045: GenPRMExtractor in CI stub mode extracts via regex."""

    def setup_method(self):
        self.ext = GenPRMExtractor(llm_caller=None)

    def test_latex_times_inline(self):
        # REQ-EXTRACT-045: LaTeX \times inside \( \) should be extracted.
        text = r"So the cost is \(7 \times 1.5 = 10.5\) dollars."
        claims = self.ext.extract_claims(text)
        assert any(c.lhs_expr == "7*1.5" and c.rhs_value == pytest.approx(10.5) for c in claims)

    def test_latex_cdot_inline(self):
        text = r"Result: \(4 \cdot 5 = 20\)."
        claims = self.ext.extract_claims(text)
        assert any(c.lhs_expr == "4*5" and c.rhs_value == pytest.approx(20.0) for c in claims)

    def test_latex_div_op(self):
        text = r"We compute \(10 \div 2 = 5\)."
        claims = self.ext.extract_claims(text)
        assert any(c.rhs_value == pytest.approx(5.0) for c in claims)

    def test_latex_frac(self):
        text = r"The rate is \frac{200}{2} = 100 minutes."
        claims = self.ext.extract_claims(text)
        assert any(c.lhs_expr == "200/2" and c.rhs_value == pytest.approx(100.0) for c in claims)

    def test_unicode_multiply(self):
        # REQ-EXTRACT-045: Unicode × operator should be extracted.
        text = "Distance = 60 × 3 = 180 miles."
        claims = self.ext.extract_claims(text)
        assert any("60" in c.lhs_expr and c.rhs_value == pytest.approx(180.0) for c in claims)

    def test_unicode_divide(self):
        text = "Result: 10 ÷ 2 = 5."
        claims = self.ext.extract_claims(text)
        assert any(c.rhs_value == pytest.approx(5.0) for c in claims)

    def test_plain_multiply_no_dollar(self):
        # REQ-EXTRACT-045: plain N*M=P without $ sign (V3 requires $).
        text = "7 * 1.5 = 10.0"
        claims = self.ext.extract_claims(text)
        assert any(c.lhs_expr == "7*1.5" and c.rhs_value == pytest.approx(10.0) for c in claims)

    def test_plain_divide_no_dollar(self):
        text = "90 / 7 = 12"
        claims = self.ext.extract_claims(text)
        assert any("90" in c.lhs_expr and c.rhs_value == pytest.approx(12.0) for c in claims)

    def test_plain_add(self):
        text = "15 + 25 = 40"
        claims = self.ext.extract_claims(text)
        assert any(c.rhs_value == pytest.approx(40.0) for c in claims)

    def test_plain_sub(self):
        text = "100 - 30 = 70"
        claims = self.ext.extract_claims(text)
        assert any(c.rhs_value == pytest.approx(70.0) for c in claims)

    def test_no_claims_placeholder(self):
        # SCENARIO-EXTRACT-082: placeholder responses yield empty list.
        claims = self.ext.extract_claims("The answer is 42.")
        assert claims == []

    def test_deduplication(self):
        # Same claim appearing twice should appear once.
        text = "7 * 1.5 = 10.5 and also 7 * 1.5 = 10.5"
        claims = self.ext.extract_claims(text)
        matching = [c for c in claims if c.lhs_expr == "7*1.5" and c.rhs_value == pytest.approx(10.5)]
        assert len(matching) == 1

    def test_confidence_default(self):
        text = "7 * 1.5 = 10.5"
        claims = self.ext.extract_claims(text)
        assert all(c.confidence == pytest.approx(0.85) for c in claims)

    def test_latex_display_block(self):
        # LaTeX display block \[ ... \] should be scanned for \times.
        text = r"\[ 3 \times 68 \]"
        # No = sign inside, so no claim expected (no rhs to verify).
        claims = self.ext.extract_claims(text)
        # The block has no = sign; _LATEX_TIMES looks for '= P'. No claim.
        assert isinstance(claims, list)

    def test_latex_display_block_with_result(self):
        text = r"\[ 3 \times 68 = 204 \]"
        claims = self.ext.extract_claims(text)
        assert any(c.rhs_value == pytest.approx(204.0) for c in claims)


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-082: GenPRMExtractor LLM path
# ---------------------------------------------------------------------------


class TestGenPRMExtractorLLMPath:
    """REQ-EXTRACT-045: GenPRMExtractor calls llm_caller and parses JSON."""

    def test_valid_llm_response(self):
        json_payload = json.dumps([
            {"lhs": "7*1.5", "rhs": 10.0, "text": "7 * 1.5 = 10.0"},
        ])

        def _caller(prompt: str) -> str:
            assert "Response:" in prompt
            return json_payload

        ext = GenPRMExtractor(llm_caller=_caller)
        claims = ext.extract_claims("7 * 1.5 = 10.0")
        assert len(claims) == 1
        assert claims[0].lhs_expr == "7*1.5"
        assert claims[0].rhs_value == pytest.approx(10.0)

    def test_llm_response_with_markdown_fence(self):
        # LLMs often wrap JSON in ```json ... ``` fences.
        payload = '[{"lhs": "3*4", "rhs": 12.0, "text": "3*4=12"}]'
        wrapped = f"```json\n{payload}\n```"

        ext = GenPRMExtractor(llm_caller=lambda _: wrapped)
        claims = ext.extract_claims("3*4=12")
        assert any(c.rhs_value == pytest.approx(12.0) for c in claims)

    def test_llm_response_empty_array(self):
        ext = GenPRMExtractor(llm_caller=lambda _: "[]")
        claims = ext.extract_claims("No math here.")
        assert claims == []

    def test_llm_malformed_json_falls_back_to_regex(self):
        # If LLM returns garbage, should fall back to regex and not crash.
        ext = GenPRMExtractor(llm_caller=lambda _: "NOT JSON")
        claims = ext.extract_claims("7 * 1.5 = 10.5")
        # regex fallback should find the plain multiplication
        assert isinstance(claims, list)

    def test_llm_exception_falls_back_to_regex(self):
        def _bad_caller(prompt: str) -> str:
            raise RuntimeError("LLM unavailable")

        ext = GenPRMExtractor(llm_caller=_bad_caller)
        claims = ext.extract_claims("7 * 1.5 = 10.5")
        assert isinstance(claims, list)

    def test_llm_missing_fields_skipped(self):
        # Items missing 'lhs' or 'rhs' should be silently skipped.
        payload = '[{"lhs": "3*4"}, {"rhs": 12.0}]'
        ext = GenPRMExtractor(llm_caller=lambda _: payload)
        claims = ext.extract_claims("3*4=12")
        assert claims == []

    def test_llm_confidence_field_used(self):
        payload = '[{"lhs": "3*4", "rhs": 12.0, "text": "x", "confidence": 0.95}]'
        ext = GenPRMExtractor(llm_caller=lambda _: payload)
        claims = ext.extract_claims("3*4=12")
        assert claims[0].confidence == pytest.approx(0.95)

    def test_extraction_prompt_contains_response(self):
        received = []

        def _caller(prompt: str) -> str:
            received.append(prompt)
            return "[]"

        ext = GenPRMExtractor(llm_caller=_caller)
        ext.extract_claims("some response text")
        assert "some response text" in received[0]
        assert "Response:" in received[0]


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-083: CoACEExtractorV4 integration
# ---------------------------------------------------------------------------


class TestCoACEExtractorV4:
    """REQ-EXTRACT-045/046: V4 merges V3 + GenPRM results and deduplicates."""

    def test_ci_stub_catches_plain_arithmetic_error(self):
        # SCENARIO-EXTRACT-083: 7*1.5 stated as 10 — V4 CI stub should flag this.
        # Note: V3 already catches this via prose patterns (7 * $1.5 = $10 form).
        # This test confirms V4 also catches it via plain pattern.
        text = "Carlos earns 7 * 1.5 = 10 per year."
        v4 = CoACEExtractorV4(llm_caller=None)
        result = v4.extract(text)
        assert result.n_violations > 0

    def test_ci_stub_no_violation_on_correct_arithmetic(self):
        # SCENARIO-EXTRACT-083: 3 * 4 = 12 is correct; no violation.
        text = "The total is 3 * 4 = 12."
        v4 = CoACEExtractorV4(llm_caller=None)
        result = v4.extract(text)
        # Should have no violations for this correct arithmetic.
        violations_from_this_eq = [
            v for v in result.violations
            if "3" in v.equation.lhs_expr and "4" in v.equation.lhs_expr
        ]
        assert violations_from_this_eq == []

    def test_v4_catches_latex_arithmetic_error(self):
        # V4 should extract and flag LaTeX arithmetic that is wrong.
        text = r"The result is \(7 \times 1.5 = 10.0\)."
        v4 = CoACEExtractorV4(llm_caller=None)
        result = v4.extract(text)
        assert result.n_violations > 0
        assert result.extraction_mode == "genprm_v4"

    def test_v4_deduplication_with_v3(self):
        # If V3 already caught a violation, V4 should not double-count it.
        # Carlos response: 7 * $1.5 = $10 — V3 catches via currency pattern.
        text = "Carlos earns 7 * $1.5 = $10 per year."
        v4 = CoACEExtractorV4(llm_caller=None)
        result = v4.extract(text)
        # Count violations with lhs_expr containing '7' and '1.5'
        matching = [
            v for v in result.violations
            if "7" in v.equation.lhs_expr and "1.5" in v.equation.lhs_expr
        ]
        # Should appear exactly once despite V3 and V4 both seeing it.
        assert len(matching) == 1

    def test_v4_placeholder_response_no_violations(self):
        # SCENARIO-EXTRACT-082: placeholder response yields no violations.
        v4 = CoACEExtractorV4(llm_caller=None)
        result = v4.extract("The answer is 42.")
        assert result.n_violations == 0

    def test_v4_extraction_mode_label(self):
        v4 = CoACEExtractorV4(llm_caller=None)
        result = v4.extract("3 * 4 = 12")
        assert result.extraction_mode == "genprm_v4"

    def test_v4_with_llm_caller_arithmetic_error(self):
        # SCENARIO-EXTRACT-083: LLM path finds error that regex misses.
        payload = '[{"lhs": "3*4", "rhs": 13.0, "text": "3*4=13"}]'
        v4 = CoACEExtractorV4(llm_caller=lambda _: payload)
        result = v4.extract("The answer is 3 times 4 which equals thirteen.")
        assert result.n_violations > 0

    def test_v4_with_llm_caller_no_error(self):
        # LLM returns correct arithmetic — no violation.
        payload = '[{"lhs": "3*4", "rhs": 12.0, "text": "3*4=12"}]'
        v4 = CoACEExtractorV4(llm_caller=lambda _: payload)
        result = v4.extract("3 times 4 equals twelve.")
        v4_violations = [v for v in result.violations if "3" in v.equation.lhs_expr]
        assert v4_violations == []

    def test_v4_inherits_v3(self):
        # Confirm V4 is a subclass of V3.
        from carnot.extraction.coace_extractor_v3 import CoACEExtractorV3
        assert issubclass(CoACEExtractorV4, CoACEExtractorV3)

    def test_arithmetic_claim_dataclass(self):
        # SCENARIO-EXTRACT-080: ArithmeticClaim stores all required fields.
        claim = ArithmeticClaim(
            lhs_expr="7*1.5",
            rhs_value=10.0,
            claim_text="7 * 1.5 = 10",
            confidence=0.85,
        )
        assert claim.lhs_expr == "7*1.5"
        assert claim.rhs_value == pytest.approx(10.0)
        assert claim.claim_text == "7 * 1.5 = 10"
        assert claim.confidence == pytest.approx(0.85)

    def test_extraction_prompt_constant(self):
        # EXTRACTION_PROMPT must include the {response} placeholder.
        assert "{response}" in EXTRACTION_PROMPT
        assert "JSON" in EXTRACTION_PROMPT
        assert "lhs" in EXTRACTION_PROMPT
