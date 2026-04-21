"""Tests for TrustAgentsExtractor — three-agent arithmetic claim extractor.

Coverage target: 100% of trust_agents_extractor.py.

Spec: REQ-EXTRACT-053, SCENARIO-EXTRACT-090, SCENARIO-EXTRACT-091
"""

from __future__ import annotations

import json

import pytest

from carnot.extraction.trust_agents_extractor import (
    Agent1NER,
    Agent2ClaimFormer,
    Agent3Verifier,
    TrustAgentsExtractor,
)
from carnot.extraction.llm_extractor_v1 import ArithmeticClaim


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _stub(response: str):
    """Return a stub llm_caller that always returns the given string."""
    def llm_caller(prompt: str) -> str:
        return response
    return llm_caller


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-090: Agent1NER
# ---------------------------------------------------------------------------


class TestAgent1NER:
    """REQ-EXTRACT-053: Agent1NER extracts numeric entity strings from text."""

    def test_returns_list_of_strings(self):
        raw = json.dumps(["3", "16.50", "49.50"])
        entities = Agent1NER("3 * $16.50 = $49.50", _stub(raw))
        assert entities == ["3", "16.50", "49.50"]

    def test_empty_array_returns_empty(self):
        entities = Agent1NER("No numbers here.", _stub("[]"))
        assert entities == []

    def test_malformed_json_returns_empty(self):
        entities = Agent1NER("text", _stub("not json"))
        assert entities == []

    def test_no_array_in_response_returns_empty(self):
        entities = Agent1NER("text", _stub('{"key": "val"}'))
        assert entities == []

    def test_llm_exception_returns_empty(self):
        def bad_llm(prompt: str) -> str:
            raise RuntimeError("GPU OOM")
        assert Agent1NER("text", bad_llm) == []

    def test_items_stripped(self):
        raw = json.dumps(["  42  ", " 3.14 "])
        entities = Agent1NER("...", _stub(raw))
        assert entities == ["42", "3.14"]

    def test_non_string_items_coerced(self):
        # LLM might return numbers instead of strings.
        raw = json.dumps([3, 16.5])
        entities = Agent1NER("...", _stub(raw))
        assert "3" in entities
        assert "16.5" in entities

    def test_empty_string_items_excluded(self):
        raw = json.dumps(["", "5", "  "])
        entities = Agent1NER("...", _stub(raw))
        # Empty and whitespace-only items should be dropped.
        assert entities == ["5"]

    def test_array_brackets_but_invalid_json_returns_empty(self):
        # The regex matches [...] but json.loads fails (line 66-67 branch).
        # "[ not valid ]" matches \[.*?\] but is not valid JSON.
        entities = Agent1NER("...", _stub("[ not valid ]"))
        assert entities == []


# ---------------------------------------------------------------------------
# Agent2ClaimFormer
# ---------------------------------------------------------------------------


class TestAgent2ClaimFormer:
    """REQ-EXTRACT-053: Agent2ClaimFormer forms arithmetic claim dicts from entities."""

    def test_basic_claim_returned(self):
        raw = json.dumps([{"lhs": "3*16.50", "rhs": 49.50, "text": "test"}])
        claims = Agent2ClaimFormer(["3", "16.50", "49.50"], "text", _stub(raw))
        assert len(claims) == 1
        assert claims[0]["lhs"] == "3*16.50"
        assert claims[0]["rhs"] == pytest.approx(49.50)

    def test_empty_entities_returns_empty(self):
        claims = Agent2ClaimFormer([], "text", _stub("[]"))
        assert claims == []

    def test_malformed_json_returns_empty(self):
        claims = Agent2ClaimFormer(["3"], "text", _stub("broken"))
        assert claims == []

    def test_no_array_returns_empty(self):
        claims = Agent2ClaimFormer(["3"], "text", _stub('{"k": "v"}'))
        assert claims == []

    def test_llm_exception_returns_empty(self):
        def bad_llm(prompt: str) -> str:
            raise ValueError("timeout")
        assert Agent2ClaimFormer(["3"], "text", bad_llm) == []

    def test_missing_lhs_skipped(self):
        raw = json.dumps([{"rhs": 42, "text": "no lhs"}])
        claims = Agent2ClaimFormer(["42"], "text", _stub(raw))
        assert claims == []

    def test_missing_rhs_skipped(self):
        raw = json.dumps([{"lhs": "2+2", "text": "no rhs"}])
        claims = Agent2ClaimFormer(["2"], "text", _stub(raw))
        assert claims == []

    def test_invalid_rhs_skipped(self):
        raw = json.dumps([{"lhs": "2+2", "rhs": "not_a_number", "text": "bad"}])
        claims = Agent2ClaimFormer(["2"], "text", _stub(raw))
        assert claims == []

    def test_non_dict_item_skipped(self):
        raw = json.dumps(["string", {"lhs": "1+1", "rhs": 2, "text": "ok"}])
        claims = Agent2ClaimFormer(["1"], "text", _stub(raw))
        assert len(claims) == 1

    def test_multiple_claims_returned(self):
        raw = json.dumps([
            {"lhs": "3*16.50", "rhs": 49.50, "text": "a"},
            {"lhs": "49.50+10", "rhs": 59.50, "text": "b"},
        ])
        claims = Agent2ClaimFormer(["3", "16.50", "49.50", "10", "59.50"], "text", _stub(raw))
        assert len(claims) == 2

    def test_array_brackets_but_invalid_json_returns_empty(self):
        # The regex matches [...] but json.loads fails (line 111-112 branch).
        claims = Agent2ClaimFormer(["3"], "text", _stub("[ not valid ]"))
        assert claims == []


# ---------------------------------------------------------------------------
# Agent3Verifier
# ---------------------------------------------------------------------------


class TestAgent3Verifier:
    """REQ-EXTRACT-053: Agent3Verifier converts claim dicts to ArithmeticClaim objects."""

    def test_basic_claim_converted(self):
        claims = Agent3Verifier([{"lhs": "3*16.50", "rhs": 49.50, "text": "test"}])
        assert len(claims) == 1
        assert isinstance(claims[0], ArithmeticClaim)
        assert claims[0].lhs_expr == "3*16.50"
        assert claims[0].rhs_value == pytest.approx(49.50)
        assert claims[0].strategy == "trust_agents"
        assert claims[0].confidence == pytest.approx(0.85)

    def test_empty_input_returns_empty(self):
        assert Agent3Verifier([]) == []

    def test_claim_text_truncated_to_120(self):
        long_text = "x" * 200
        claims = Agent3Verifier([{"lhs": "1+1", "rhs": 2.0, "text": long_text}])
        assert len(claims[0].claim_text) == 120

    def test_multiple_claims(self):
        claim_dicts = [
            {"lhs": "2+2", "rhs": 4.0, "text": "a"},
            {"lhs": "3*3", "rhs": 9.0, "text": "b"},
        ]
        claims = Agent3Verifier(claim_dicts)
        assert len(claims) == 2

    def test_strategy_is_trust_agents(self):
        claims = Agent3Verifier([{"lhs": "1+1", "rhs": 2.0, "text": "ok"}])
        assert claims[0].strategy == "trust_agents"

    def test_missing_text_defaults_to_empty(self):
        claims = Agent3Verifier([{"lhs": "1+1", "rhs": 2.0}])
        assert claims[0].claim_text == ""

    def test_none_rhs_becomes_none(self):
        claims = Agent3Verifier([{"lhs": "1+1", "rhs": None, "text": "ok"}])
        assert claims[0].rhs_value is None


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-090: TrustAgentsExtractor — CI mode
# ---------------------------------------------------------------------------


class TestTrustAgentsExtractorCI:
    """SCENARIO-EXTRACT-090: CI mode returns [] without calling llm_caller."""

    def test_ci_mode_returns_empty(self):
        extractor = TrustAgentsExtractor(llm_caller=None)
        result = extractor.extract("She spent 3*16.50=54.50 on shorts.")
        assert result == []

    def test_ci_mode_placeholder_returns_empty(self):
        extractor = TrustAgentsExtractor(llm_caller=None)
        result = extractor.extract("The answer is 42.")
        assert result == []

    def test_default_tolerance(self):
        extractor = TrustAgentsExtractor()
        assert extractor.tolerance == pytest.approx(1e-6)


# ---------------------------------------------------------------------------
# SCENARIO-EXTRACT-091: TrustAgentsExtractor — live mode
# ---------------------------------------------------------------------------


class TestTrustAgentsExtractorLive:
    """SCENARIO-EXTRACT-091: Live mode runs three-agent pipeline and returns violations."""

    def _make_llm(self, ner_response: str, claim_response: str):
        """Build a stub that returns different responses for NER vs. ClaimFormer prompts."""
        call_count = {"n": 0}

        def llm_caller(prompt: str) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return ner_response
            return claim_response

        return llm_caller

    def test_violation_detected(self):
        # Agent1 finds entities; Agent2 forms a claim; Agent3 finds violation.
        ner_raw = json.dumps(["3", "16.50", "54.50"])
        claim_raw = json.dumps([{"lhs": "3*16.50", "rhs": 54.50, "text": "3 * $16.50 = $54.50"}])
        extractor = TrustAgentsExtractor(llm_caller=self._make_llm(ner_raw, claim_raw))
        # 3*16.50 = 49.50 != 54.50 → violation
        violations = extractor.extract("She spent 3 * $16.50 = $54.50 on shorts.")
        assert len(violations) == 1
        assert violations[0].strategy == "trust_agents"

    def test_no_violation_on_correct_arithmetic(self):
        ner_raw = json.dumps(["3", "16.50", "49.50"])
        claim_raw = json.dumps([{"lhs": "3*16.50", "rhs": 49.50, "text": "correct"}])
        extractor = TrustAgentsExtractor(llm_caller=self._make_llm(ner_raw, claim_raw))
        # 3*16.50 = 49.50 == 49.50 → no violation
        violations = extractor.extract("Total: 3*16.50=49.50.")
        assert violations == []

    def test_agent1_returns_empty_no_violations(self):
        extractor = TrustAgentsExtractor(llm_caller=self._make_llm("[]", "[]"))
        violations = extractor.extract("No numbers.")
        assert violations == []

    def test_agent2_returns_empty_no_violations(self):
        ner_raw = json.dumps(["42"])
        extractor = TrustAgentsExtractor(llm_caller=self._make_llm(ner_raw, "[]"))
        violations = extractor.extract("The answer is 42.")
        assert violations == []

    def test_unevaluable_lhs_dropped(self):
        # Agent2 emits a claim with lhs that safe_eval cannot handle.
        ner_raw = json.dumps(["42"])
        claim_raw = json.dumps([{"lhs": "os.system('ls')", "rhs": 0.0, "text": "unsafe"}])
        extractor = TrustAgentsExtractor(llm_caller=self._make_llm(ner_raw, claim_raw))
        violations = extractor.extract("...")
        assert violations == []

    def test_rhs_none_dropped(self):
        # Agent2 should always produce rhs, but if None slips through it is dropped.
        ner_raw = json.dumps(["42"])
        claim_raw = json.dumps([{"lhs": "2+2", "rhs": None, "text": "ok"}])
        extractor = TrustAgentsExtractor(llm_caller=self._make_llm(ner_raw, claim_raw))
        violations = extractor.extract("...")
        assert violations == []

    def test_filter_violations_within_tolerance(self):
        # 2+2 = 4.0, rhs = 4.0000001 — within default tolerance (1e-6).
        ner_raw = json.dumps(["2", "4"])
        claim_raw = json.dumps([{"lhs": "2+2", "rhs": 4.0000001, "text": "close"}])
        extractor = TrustAgentsExtractor(llm_caller=self._make_llm(ner_raw, claim_raw))
        violations = extractor.extract("2+2 = 4.0000001")
        # 4.0 vs 4.0000001 delta = 1e-7 < 1e-6 → NOT a violation
        assert violations == []

    def test_filter_violations_rhs_none_skipped_directly(self):
        # Directly call _filter_violations with a claim that has rhs_value=None (line 226).
        extractor = TrustAgentsExtractor(llm_caller=_stub("[]"))
        claim = ArithmeticClaim(
            lhs_expr="2+2",
            rhs_value=None,
            claim_text="test",
            strategy="trust_agents",
            confidence=0.85,
        )
        result = extractor._filter_violations([claim])
        assert result == []

    def test_custom_tolerance_respected(self):
        ner_raw = json.dumps(["2", "4"])
        claim_raw = json.dumps([{"lhs": "2+2", "rhs": 4.1, "text": "off by 0.1"}])
        extractor = TrustAgentsExtractor(
            llm_caller=self._make_llm(ner_raw, claim_raw),
            tolerance=0.5,
        )
        # 4.0 vs 4.1, delta = 0.1 < tolerance 0.5 → no violation
        violations = extractor.extract("2+2=4.1")
        assert violations == []
