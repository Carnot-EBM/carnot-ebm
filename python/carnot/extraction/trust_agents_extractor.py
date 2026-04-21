"""TrustAgentsExtractor — three-agent claim extraction pipeline.

**Why this module exists:**

    LLMAsExtractorV1 treats extraction as a single-prompt problem.  The TRUST
    Agents paper (arXiv 2604.12184) decomposes extraction into three specialised
    agents that run in sequence:

        Agent 1 (NER): identify all numeric values in the text.
        Agent 2 (ClaimFormer): given those numbers, find the arithmetic
                               relationships they participate in.
        Agent 3 (Verifier): evaluate each claimed relationship with eval().

    The decomposition improves precision on natural-language CoT responses:
    Agent 1 anchors attention to specific numbers, Agent 2 uses those anchors
    to form structured claims, and Agent 3 mechanically verifies them.

    When llm_caller is None (CI mode): extract() returns [] immediately.
    This is intentional — the three-agent pipeline requires two LLM calls and
    produces no useful signal without a real LLM.

Spec: REQ-EXTRACT-053, SCENARIO-EXTRACT-090, SCENARIO-EXTRACT-091
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

from carnot.extraction.llm_extractor_v1 import ArithmeticClaim, safe_eval

# ---------------------------------------------------------------------------
# Agent 1: Named Entity Recognition — find all numeric values
# ---------------------------------------------------------------------------

_NER_PROMPT = (
    "List all numeric values in this text as a JSON array of strings. "
    "Include integers, decimals, and currency amounts (strip the $ sign). "
    "Output ONLY a JSON array, no prose. Text: {text}"
)


def Agent1NER(
    response: str,
    llm_caller: Callable[[str], str],
) -> list[str]:
    """Extract all numeric entity strings from the response via LLM.

    The LLM is asked for a JSON array of strings, one per numeric value found.
    This gives Agent 2 a concrete set of numbers to build arithmetic claims from.

    Returns [] on LLM error or malformed JSON.
    """
    prompt = _NER_PROMPT.format(text=response)
    try:
        raw = llm_caller(prompt)
    except Exception:  # noqa: BLE001
        return []

    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    return [str(item).strip() for item in items if str(item).strip()]


# ---------------------------------------------------------------------------
# Agent 2: ClaimFormer — build arithmetic claims from entities
# ---------------------------------------------------------------------------

_CLAIM_PROMPT = (
    "Given these numbers: {entities}. "
    "Find arithmetic relationships stated in this text. "
    "Output JSON: [{{'lhs': 'expr', 'rhs': value, 'text': 'excerpt'}}]. "
    "Output ONLY valid JSON array. Text: {text}"
)


def Agent2ClaimFormer(
    entities: list[str],
    response: str,
    llm_caller: Callable[[str], str],
) -> list[dict]:
    """Form arithmetic claim dicts from the numeric entities found by Agent 1.

    The LLM is given the entity list as anchors to identify arithmetic
    relationships in the text (e.g., "3 * 16.50 = 49.50" from entities
    ["3", "16.50", "49.50"]).

    Returns [] on LLM error, empty entity list, or malformed JSON.
    """
    if not entities:
        return []

    prompt = _CLAIM_PROMPT.format(entities=", ".join(entities), text=response)
    try:
        raw = llm_caller(prompt)
    except Exception:  # noqa: BLE001
        return []

    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    claims = []
    for item in items:
        if not isinstance(item, dict):
            continue
        lhs = str(item.get("lhs", "")).strip()
        rhs_raw = item.get("rhs")
        text = str(item.get("text", "")).strip()
        if not lhs or rhs_raw is None:
            continue
        try:
            rhs = float(rhs_raw)
        except (ValueError, TypeError):
            continue
        claims.append({"lhs": lhs, "rhs": rhs, "text": text})
    return claims


# ---------------------------------------------------------------------------
# Agent 3: Verifier — evaluate claims with safe_eval
# ---------------------------------------------------------------------------


def Agent3Verifier(claims: list[dict]) -> list[ArithmeticClaim]:
    """Verify arithmetic claims by evaluating lhs with safe_eval.

    For each claim dict from Agent 2, evaluate lhs with safe_eval() and
    produce an ArithmeticClaim regardless of whether it matches rhs.
    The caller (TrustAgentsExtractor.extract) is responsible for filtering
    violations — Agent 3 only performs the evaluation.

    Claims with unevaluable lhs (safe_eval returns None) are still included
    as ArithmeticClaim objects with no computed value to compare, which lets
    the upstream violation filter handle them uniformly.
    """
    result: list[ArithmeticClaim] = []
    for claim in claims:
        lhs = claim.get("lhs", "")
        rhs = claim.get("rhs")
        text = claim.get("text", "")
        result.append(
            ArithmeticClaim(
                lhs_expr=lhs,
                rhs_value=rhs,
                claim_text=text[:120],
                strategy="trust_agents",
                confidence=0.85,
            )
        )
    return result


# ---------------------------------------------------------------------------
# TrustAgentsExtractor — orchestrator
# ---------------------------------------------------------------------------


class TrustAgentsExtractor:
    """Three-agent arithmetic claim extractor (arXiv 2604.12184 TRUST Agents).

    Pipeline:
        Agent1NER(response) → list of numeric entity strings
        Agent2ClaimFormer(entities, response) → list of claim dicts
        Agent3Verifier(claims) → list[ArithmeticClaim]

    Then filter to violations only (same contract as LLMAsExtractorV1.extract).

    CI mode (llm_caller=None): returns [] immediately.  The two LLM calls in
    the pipeline cannot run without a real LLM, so there is no useful baseline
    output to fall back to.

    Args:
        llm_caller : callable(prompt: str) -> str, or None for CI mode.
        tolerance  : absolute error below which a discrepancy is ignored.

    Spec: REQ-EXTRACT-053, SCENARIO-EXTRACT-090, SCENARIO-EXTRACT-091
    """

    def __init__(
        self,
        llm_caller: Optional[Callable[[str], str]] = None,
        tolerance: float = 1e-6,
    ) -> None:
        self.llm_caller = llm_caller
        self.tolerance = tolerance

    def extract(self, response: str) -> list[ArithmeticClaim]:
        """Run the three-agent pipeline and return violation claims.

        CI mode (llm_caller=None): returns [] — no fallback baseline.
        Live mode: Agent1 → Agent2 → Agent3 → filter violations.

        A violation is a claim where safe_eval(lhs_expr) differs from
        rhs_value by more than self.tolerance.  Claims where safe_eval
        returns None (unevaluable) are silently dropped.
        """
        if self.llm_caller is None:
            return []

        entities = Agent1NER(response, self.llm_caller)
        claim_dicts = Agent2ClaimFormer(entities, response, self.llm_caller)
        claims = Agent3Verifier(claim_dicts)
        return self._filter_violations(claims)

    def _filter_violations(self, claims: list[ArithmeticClaim]) -> list[ArithmeticClaim]:
        """Return only claims where safe_eval(lhs) disagrees with rhs_value.

        Claims with rhs_value=None are dropped (Agent2 always produces an rhs).
        Claims where safe_eval returns None (unevaluable) are silently dropped.
        """
        violations: list[ArithmeticClaim] = []
        for claim in claims:
            if claim.rhs_value is None:
                continue
            computed = safe_eval(claim.lhs_expr)
            if computed is None:
                continue
            if abs(computed - claim.rhs_value) > self.tolerance:
                violations.append(claim)
        return violations
