"""Tests for LLMAsExtractorV1 — three-strategy LLM-based arithmetic extractor.

Coverage target: 100% of llm_extractor_v1.py.

Spec: REQ-EXTRACT-050, REQ-EXTRACT-051, REQ-EXTRACT-052,
      SCENARIO-EXTRACT-085, SCENARIO-EXTRACT-086, SCENARIO-EXTRACT-087,
      SCENARIO-EXTRACT-088, SCENARIO-EXTRACT-089
"""

from __future__ import annotations

import json

import pytest

from carnot.extraction.llm_extractor_v1 import (
    ArithmeticClaim,
    JsonClaimExtractor,
    LLMAsExtractorV1,
    StepSegmentEvalChain,
    SymCodeExtractor,
    safe_eval,
)


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-085: safe_eval — allowed and blocked operations
# ---------------------------------------------------------------------------


class TestSafeEval:
    """REQ-EXTRACT-050: safe_eval evaluates restricted arithmetic; blocks unsafe code."""

    def test_addition(self):
        assert safe_eval("1+2") == pytest.approx(3.0)

    def test_subtraction(self):
        assert safe_eval("10-3") == pytest.approx(7.0)

    def test_multiplication(self):
        assert safe_eval("3*16.50") == pytest.approx(49.50)

    def test_division(self):
        assert safe_eval("100/4") == pytest.approx(25.0)

    def test_power(self):
        assert safe_eval("2**8") == pytest.approx(256.0)

    def test_modulo(self):
        assert safe_eval("10%3") == pytest.approx(1.0)

    def test_parentheses(self):
        assert safe_eval("(1+2)*3") == pytest.approx(9.0)

    def test_unary_minus(self):
        assert safe_eval("-5+10") == pytest.approx(5.0)

    def test_unary_plus(self):
        assert safe_eval("+3") == pytest.approx(3.0)

    def test_currency_stripped(self):
        # Dollar sign and comma must be stripped before evaluation.
        assert safe_eval("$3*16.50") == pytest.approx(49.50)

    def test_comma_stripped(self):
        assert safe_eval("1,000+500") == pytest.approx(1500.0)

    def test_underscore_stripped(self):
        assert safe_eval("1_000+0") == pytest.approx(1000.0)

    def test_division_by_zero_returns_none(self):
        assert safe_eval("1/0") is None

    def test_overflow_returns_none(self):
        assert safe_eval("10**400") is None

    def test_invalid_syntax_returns_none(self):
        assert safe_eval("abc+1") is None

    def test_empty_string_returns_none(self):
        assert safe_eval("") is None

    def test_function_call_blocked(self):
        # exec() and other builtins must be blocked by AST walk.
        assert safe_eval("__import__('os')") is None

    def test_attribute_access_blocked(self):
        assert safe_eval("(1).real") is None

    def test_list_literal_blocked(self):
        # List literals are not numeric operations.
        assert safe_eval("[1,2]") is None

    def test_string_literal_blocked(self):
        assert safe_eval("'hello'") is None

    def test_nested_expression(self):
        assert safe_eval("(10+5)*2/5") == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-086: JsonClaimExtractor — LLM-emitted JSON array parsing
# ---------------------------------------------------------------------------


class TestJsonClaimExtractor:
    """REQ-EXTRACT-052: JsonClaimExtractor calls LLM and parses JSON claim array."""

    def _make_llm(self, response: str):
        """Build a stub llm_caller that returns a fixed string."""
        def llm_caller(prompt: str) -> str:
            return response
        return llm_caller

    def test_basic_claim_extracted(self):
        # LLM returns a valid JSON array with one claim.
        raw = json.dumps([{"lhs": "3*16.50", "rhs": 49.50, "text": "3 * $16.50 = $49.50"}])
        extractor = JsonClaimExtractor()
        claims = extractor.extract_claims("3 * $16.50 = $49.50", self._make_llm(raw))
        assert len(claims) == 1
        assert claims[0].lhs_expr == "3*16.50"
        assert claims[0].rhs_value == pytest.approx(49.50)
        assert claims[0].strategy == "json_claim"
        assert claims[0].confidence == pytest.approx(0.85)

    def test_multiple_claims(self):
        raw = json.dumps([
            {"lhs": "35*20", "rhs": 700, "text": "35 hours × $20 = $700"},
            {"lhs": "700*50", "rhs": 35000, "text": "$700/wk × 50wk = $35000"},
        ])
        extractor = JsonClaimExtractor()
        claims = extractor.extract_claims("...", self._make_llm(raw))
        assert len(claims) == 2
        assert claims[0].lhs_expr == "35*20"
        assert claims[1].rhs_value == pytest.approx(35000.0)

    def test_empty_array_returns_empty(self):
        extractor = JsonClaimExtractor()
        claims = extractor.extract_claims("No math here.", self._make_llm("[]"))
        assert claims == []

    def test_malformed_json_returns_empty(self):
        extractor = JsonClaimExtractor()
        claims = extractor.extract_claims("...", self._make_llm("not json at all"))
        assert claims == []

    def test_no_json_array_returns_empty(self):
        # LLM returned a JSON object, not array.
        extractor = JsonClaimExtractor()
        claims = extractor.extract_claims("...", self._make_llm('{"key": "value"}'))
        assert claims == []

    def test_markdown_fenced_json_parsed(self):
        # LLM sometimes wraps JSON in ```json ... ``` fences.
        raw = '```json\n[{"lhs": "2+2", "rhs": 4, "text": "2+2=4"}]\n```'
        extractor = JsonClaimExtractor()
        claims = extractor.extract_claims("2+2=4", self._make_llm(raw))
        assert len(claims) == 1
        assert claims[0].lhs_expr == "2+2"

    def test_item_missing_lhs_skipped(self):
        raw = json.dumps([{"rhs": 42, "text": "missing lhs"}])
        extractor = JsonClaimExtractor()
        claims = extractor.extract_claims("...", self._make_llm(raw))
        assert claims == []

    def test_item_missing_rhs_skipped(self):
        raw = json.dumps([{"lhs": "2+2", "text": "missing rhs"}])
        extractor = JsonClaimExtractor()
        claims = extractor.extract_claims("...", self._make_llm(raw))
        assert claims == []

    def test_non_dict_item_skipped(self):
        raw = json.dumps(["not a dict", {"lhs": "1+1", "rhs": 2, "text": "ok"}])
        extractor = JsonClaimExtractor()
        claims = extractor.extract_claims("...", self._make_llm(raw))
        assert len(claims) == 1

    def test_invalid_rhs_skipped(self):
        raw = json.dumps([{"lhs": "2+2", "rhs": "not_a_number", "text": "bad rhs"}])
        extractor = JsonClaimExtractor()
        claims = extractor.extract_claims("...", self._make_llm(raw))
        assert claims == []

    def test_llm_exception_returns_empty(self):
        def bad_llm(prompt: str) -> str:
            raise RuntimeError("GPU OOM")
        extractor = JsonClaimExtractor()
        claims = extractor.extract_claims("...", bad_llm)
        assert claims == []

    def test_claim_text_truncated_to_120(self):
        long_text = "x" * 200
        raw = json.dumps([{"lhs": "1+1", "rhs": 2, "text": long_text}])
        extractor = JsonClaimExtractor()
        claims = extractor.extract_claims("...", self._make_llm(raw))
        assert len(claims[0].claim_text) == 120


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-087: SymCodeExtractor — LLM-synthesised Python expression
# ---------------------------------------------------------------------------


class TestSymCodeExtractor:
    """REQ-EXTRACT-051: SymCodeExtractor prompts LLM to emit executable Python."""

    def _make_llm(self, response: str):
        def llm_caller(prompt: str) -> str:
            return response
        return llm_caller

    def test_correct_arithmetic_no_violation(self):
        # LLM synthesises "35*20", step states result is 700.
        extractor = SymCodeExtractor()
        claims = extractor.extract_claims(
            "She earns 35 hours × $20/hour = $700 per week.",
            self._make_llm("35*20"),
        )
        # 35*20 = 700, stated = 700 → no violation → returns the claim (rhs matches)
        assert len(claims) == 1
        assert claims[0].strategy == "symcode"
        assert claims[0].lhs_expr == "35*20"
        assert claims[0].rhs_value == pytest.approx(700.0)

    def test_wrong_arithmetic_returns_claim(self):
        # LLM synthesises "3*16.50" which is 49.50, but step says "= $54.50".
        extractor = SymCodeExtractor()
        claims = extractor.extract_claims(
            "Mishka bought 3 pairs for $16.50 each = $54.50 on shorts.",
            self._make_llm("3*16.50"),
        )
        # 3*16.50 = 49.50, stated result = 54.50 → violation candidate
        assert len(claims) == 1
        assert claims[0].rhs_value == pytest.approx(54.50)

    def test_llm_returns_none_literal(self):
        extractor = SymCodeExtractor()
        claims = extractor.extract_claims("unclear step", self._make_llm("None"))
        assert claims == []

    def test_llm_returns_empty_string(self):
        extractor = SymCodeExtractor()
        claims = extractor.extract_claims("...", self._make_llm(""))
        assert claims == []

    def test_no_stated_result_returns_empty(self):
        # Step has no "= N" or "is N" pattern.
        extractor = SymCodeExtractor()
        claims = extractor.extract_claims(
            "She went to the store.", self._make_llm("3*5")
        )
        assert claims == []

    def test_llm_exception_returns_empty(self):
        def bad_llm(prompt: str) -> str:
            raise ValueError("timeout")
        extractor = SymCodeExtractor()
        claims = extractor.extract_claims("...", bad_llm)
        assert claims == []

    def test_markdown_fenced_code_stripped(self):
        # LLM wraps expression in code fence.
        extractor = SymCodeExtractor()
        claims = extractor.extract_claims(
            "Total is 100.", self._make_llm("```python\n50+50\n```")
        )
        assert len(claims) == 1
        assert claims[0].lhs_expr == "50+50"

    def test_unevaluable_expression_returns_empty(self):
        # LLM returns something safe_eval can't handle.
        extractor = SymCodeExtractor()
        claims = extractor.extract_claims(
            "Result = 42.", self._make_llm("os.system('ls')")
        )
        assert claims == []

    def test_strategy_label(self):
        extractor = SymCodeExtractor()
        claims = extractor.extract_claims(
            "Cost is 25.", self._make_llm("5*5")
        )
        if claims:
            assert claims[0].strategy == "symcode"

    def test_null_literal_returns_empty(self):
        extractor = SymCodeExtractor()
        claims = extractor.extract_claims("step", self._make_llm("null"))
        assert claims == []


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-088: StepSegmentEvalChain — regex/eval baseline
# ---------------------------------------------------------------------------


class TestStepSegmentEvalChain:
    """REQ-EXTRACT-050: StepSegmentEvalChain detects symbolic arithmetic errors."""

    def test_symbolic_multiplication_correct(self):
        extractor = StepSegmentEvalChain()
        claims = extractor.extract_claims("She spent 3*16.50=49.50 on shorts.")
        # 3*16.50 = 49.50 → correct, but the claim is still returned (violation check is caller's job)
        assert any(c.lhs_expr == "3*16.50" for c in claims)

    def test_symbolic_wrong_result(self):
        extractor = StepSegmentEvalChain()
        # 3*16.50 = 54.50 is incorrect: 3*16.50 = 49.50
        claims = extractor.extract_claims("She spent 3*16.50=54.50 on shorts.")
        assert any(c.lhs_expr == "3*16.50" and c.rhs_value == pytest.approx(54.50) for c in claims)

    def test_addition(self):
        extractor = StepSegmentEvalChain()
        claims = extractor.extract_claims("Total: 15+25=40.")
        assert any(c.lhs_expr == "15+25" for c in claims)

    def test_subtraction(self):
        extractor = StepSegmentEvalChain()
        claims = extractor.extract_claims("Remaining: 100-30=70.")
        assert any(c.lhs_expr == "100-30" for c in claims)

    def test_division(self):
        extractor = StepSegmentEvalChain()
        claims = extractor.extract_claims("Per item: 100/4=25.")
        assert any(c.lhs_expr == "100/4" for c in claims)

    def test_prose_multiplication(self):
        extractor = StepSegmentEvalChain()
        claims = extractor.extract_claims("3 times 16.50 gives 49.50.")
        assert any("3" in c.lhs_expr and "16.50" in c.lhs_expr for c in claims)

    def test_prose_addition(self):
        extractor = StepSegmentEvalChain()
        claims = extractor.extract_claims("15 plus 25 equals 40.")
        assert any("15" in c.lhs_expr and "25" in c.lhs_expr for c in claims)

    def test_no_arithmetic_returns_empty(self):
        extractor = StepSegmentEvalChain()
        claims = extractor.extract_claims("The answer is 42.")
        assert claims == []

    def test_dedup_same_equation(self):
        extractor = StepSegmentEvalChain()
        # Same equation appears twice in the text.
        claims = extractor.extract_claims("3*5=15 and also 3*5=15.")
        lhs_matches = [c for c in claims if c.lhs_expr == "3*5"]
        assert len(lhs_matches) == 1

    def test_strategy_label(self):
        extractor = StepSegmentEvalChain()
        claims = extractor.extract_claims("2+2=4.")
        for c in claims:
            assert c.strategy == "step_segment_eval"

    def test_currency_handled(self):
        extractor = StepSegmentEvalChain()
        claims = extractor.extract_claims("Cost: $3*$5=$15.")
        assert any(c.lhs_expr == "3*5" for c in claims)


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-089: LLMAsExtractorV1 — integration and CI mode
# ---------------------------------------------------------------------------


class TestLLMAsExtractorV1:
    """REQ-EXTRACT-050/051/052: LLMAsExtractorV1 selects strategy and detects violations."""

    def test_ci_mode_uses_chain_only(self):
        # No llm_caller → only StepSegmentEvalChain runs.
        extractor = LLMAsExtractorV1(llm_caller=None)
        # Arithmetic error: 3*16.50 ≠ 54.50
        violations = extractor.extract("She spent 3*16.50=54.50 on shorts.")
        assert len(violations) == 1
        assert violations[0].lhs_expr == "3*16.50"

    def test_ci_mode_correct_arithmetic_no_violation(self):
        extractor = LLMAsExtractorV1(llm_caller=None)
        violations = extractor.extract("Total: 3*16.50=49.50.")
        assert violations == []

    def test_ci_mode_placeholder_response_no_violation(self):
        extractor = LLMAsExtractorV1(llm_caller=None)
        violations = extractor.extract("The answer is 42.")
        assert violations == []

    def test_live_mode_unions_strategies(self):
        # LLM returns a JSON claim that StepSegmentEvalChain would miss.
        json_raw = json.dumps([{
            "lhs": "3*16.50",
            "rhs": 54.50,
            "text": "3 pairs at $16.50 each = $54.50",
        }])
        def llm_caller(prompt: str) -> str:
            # Return JSON claim for JsonClaimExtractor, "None" for SymCode.
            if "JSON array" in prompt:
                return json_raw
            return "None"
        extractor = LLMAsExtractorV1(llm_caller=llm_caller)
        violations = extractor.extract("3 pairs of shorts at $16.50 each is $54.50 on shorts.")
        # 3*16.50 = 49.50 ≠ 54.50 → violation found by JSON strategy
        json_violations = [v for v in violations if v.strategy == "json_claim"]
        assert len(json_violations) >= 1

    def test_live_mode_dedup(self):
        # Same claim from both JSON and chain strategies → dedup to one.
        json_raw = json.dumps([{"lhs": "3*16.50", "rhs": 54.50, "text": "test"}])
        def llm_caller(prompt: str) -> str:
            if "JSON array" in prompt:
                return json_raw
            return "None"
        extractor = LLMAsExtractorV1(llm_caller=llm_caller)
        violations = extractor.extract("3*16.50=54.50")
        # Even if both strategies find it, dedup ensures only one.
        lhs_matches = [v for v in violations if v.lhs_expr == "3*16.50"]
        assert len(lhs_matches) == 1

    def test_filter_violations_rhs_none_kept(self):
        # Claims with rhs_value=None are kept as potential violations.
        extractor = LLMAsExtractorV1(llm_caller=None)
        claim = ArithmeticClaim(
            lhs_expr="3*16.50",
            rhs_value=None,
            claim_text="test",
            strategy="json_claim",
            confidence=0.85,
        )
        result = extractor._filter_violations([claim])
        assert len(result) == 1

    def test_filter_violations_unevaluable_dropped(self):
        # Claims whose lhs_expr safe_eval returns None are silently dropped.
        extractor = LLMAsExtractorV1(llm_caller=None)
        claim = ArithmeticClaim(
            lhs_expr="os.system('ls')",
            rhs_value=0.0,
            claim_text="test",
            strategy="json_claim",
            confidence=0.85,
        )
        result = extractor._filter_violations([claim])
        assert result == []

    def test_default_best_strategy_is_chain(self):
        extractor = LLMAsExtractorV1(llm_caller=None)
        assert extractor._best_strategy == "step_segment_eval"

    def test_arithmetic_claim_dataclass_fields(self):
        claim = ArithmeticClaim(
            lhs_expr="2+2",
            rhs_value=5.0,
            claim_text="2+2=5",
            strategy="json_claim",
            confidence=0.9,
        )
        assert claim.lhs_expr == "2+2"
        assert claim.rhs_value == pytest.approx(5.0)
        assert claim.claim_text == "2+2=5"
        assert claim.strategy == "json_claim"
        assert claim.confidence == pytest.approx(0.9)

    def test_arithmetic_claim_rhs_value_optional(self):
        # rhs_value may be None for SymCode claims without explicit stated result.
        claim = ArithmeticClaim(
            lhs_expr="2+2",
            rhs_value=None,
            claim_text="two plus two",
            strategy="symcode",
            confidence=0.80,
        )
        assert claim.rhs_value is None
