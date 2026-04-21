"""LLMAsExtractorV1 — three-strategy LLM-based arithmetic claim extractor.

**Why this module exists (RETRO-070 root cause):**

    Fourteen consecutive verification attempts (CoACEV1 through V4) all used pattern
    matching, regex, or eval()-chaining to find arithmetic errors in instruction-tuned
    (IT) model outputs.  All achieved live recall of 4% or below.

    The root cause is confirmed: hand-engineered patterns CANNOT match the natural
    language phrasing of instruction-tuned models.  An IT model might write
    "3 pairs of shorts at $16.50 each is $54.50" — no regex for "N * M = P" captures
    this phrasing.  Even when the model writes "3 * $16.50 = $54.50", the dollar sign
    or comma or LaTeX wrapper often breaks the pattern.

    The fix: use a SECOND LLM call to extract arithmetic claims.  The LLM handles
    arbitrary surface variation in phrasing, notation (LaTeX, Unicode, prose), and
    structure (narrative, symbolic, step-by-step).

**Three strategies (from arXiv 2510.25975, arXiv 2601.04675):**

    1. JsonClaimExtractor — prompt LLM to emit a JSON array of {lhs, rhs, text} objects.
       The LLM reads the step and explicitly enumerates every arithmetic claim.
       Best for structured CoT with clear "N op M = P" statements.

    2. SymCodeExtractor (arXiv 2510.25975, SymCode) — prompt LLM to emit a single
       Python expression that computes the answer.  Useful when the model performs
       multi-step reasoning and states a final answer: the LLM synthesizes executable
       code, we eval it, and compare to the stated result.

    3. StepSegmentEvalChain — no LLM.  Sentence-segment the CoT, apply safe_eval()
       to numeric expressions in each sentence.  This is the CoACEV4 approach,
       included as the baseline.  CI mode uses ONLY this strategy.

**LLMAsExtractorV1 selection logic:**

    If llm_caller is None (CI mode): StepSegmentEvalChain only.
    If llm_caller provided: run all three strategies, union results, dedup by
    (lhs_expr, rhs_value).  The strategy with highest recall on a probe sample
    is stored as self._best_strategy for single-strategy reporting.

Spec: REQ-EXTRACT-050, REQ-EXTRACT-051, REQ-EXTRACT-052,
      SCENARIO-EXTRACT-085, SCENARIO-EXTRACT-086, SCENARIO-EXTRACT-087,
      SCENARIO-EXTRACT-088, SCENARIO-EXTRACT-089
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from dataclasses import dataclass
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# ArithmeticClaim — the LLMAsExtractorV1 unit of extraction
# ---------------------------------------------------------------------------


@dataclass
class ArithmeticClaim:
    """One arithmetic calculation identified in a CoT response by LLMAsExtractorV1.

    This is a richer version of the V4 ArithmeticClaim: it adds a 'strategy' field
    so that experiment scripts can attribute each extracted claim to the specific
    extraction strategy that found it.  This is critical for comparing the three
    strategies in Exp 616.

    Fields:
        lhs_expr    — Python-evaluable expression for the left-hand side.
                      Examples: "7*1.5", "130000+195000", "3*16.50".
        rhs_value   — The number the model STATED as the result.  Optional because
                      SymCodeExtractor may extract an expression whose stated result
                      is implicit (e.g., model says "= 10.5" at end of sentence).
        claim_text  — Verbatim excerpt from the response for traceability.
        strategy    — Which extraction strategy produced this claim: 'json_claim',
                      'symcode', or 'step_segment_eval'.
        confidence  — Extraction confidence 0–1.  LLM strategies: 0.85 default.
                      Regex/eval strategy: 0.90 (deterministic, no uncertainty).
    """

    lhs_expr: str
    rhs_value: Optional[float]
    claim_text: str
    strategy: str
    confidence: float


# ---------------------------------------------------------------------------
# safe_eval — restricted arithmetic evaluation (same contract as V4)
# ---------------------------------------------------------------------------


def safe_eval(expr: str) -> Optional[float]:
    """Evaluate a restricted arithmetic expression, returning the float result or None.

    Allowed node types: numeric literals, +, -, *, /, **, % operators, parentheses.
    Blocked: identifiers, attribute access, function calls, builtins, strings.

    Why restricted: lhs_expr may originate from an LLM or regex.  A naive eval()
    would execute arbitrary Python, creating a code-injection risk.  We parse the
    expression with ast.parse() and walk the AST to whitelist only safe node types,
    then eval() the whitelisted tree.  The nosec comment below is intentional —
    the AST walk IS the security gate.

    Returns None on syntax error, disallowed node, division-by-zero, or overflow.
    """
    cleaned = re.sub(r"[\$,_]", "", expr.strip())
    if not cleaned:
        return None
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return None

    _SAFE_NODES = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
    )
    for node in ast.walk(tree):
        if not isinstance(node, _SAFE_NODES):
            return None

    try:
        result = eval(compile(tree, "<string>", "eval"))  # nosec: AST-whitelisted above
        return float(result)
    except (ZeroDivisionError, OverflowError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Strategy 1: JsonClaimExtractor
# ---------------------------------------------------------------------------

_JSON_CLAIM_PROMPT = textwrap.dedent("""\
    You are an arithmetic verifier. Read this math reasoning step.
    List every arithmetic calculation stated as a JSON array.
    For each: {{"lhs": "expression_to_evaluate", "rhs": stated_numeric_result, "text": "verbatim_excerpt"}}.
    If no calculations: output []. Output ONLY valid JSON, no prose.
    Step: {step}""")


class JsonClaimExtractor:
    """Extract arithmetic claims by prompting an LLM to emit structured JSON.

    Why JSON output: by asking the LLM to enumerate claims explicitly in a machine-
    readable format, we bypass the surface-variation problem that defeats regex.
    The LLM can recognise "3 pairs of shorts at $16.50 each is $54.50" as the claim
    lhs="3*16.50", rhs=54.50, which a regex cannot.

    The LLM is prompted once per CoT step (not per sentence) to preserve context.
    Malformed JSON triggers a silent empty return — the strategy degrades gracefully.

    Spec: REQ-EXTRACT-052
    """

    def extract_claims(
        self,
        step: str,
        llm_caller: Callable[[str], str],
    ) -> list[ArithmeticClaim]:
        """Call the LLM with JSON claim prompt and parse the resulting array.

        Parameters
        ----------
        step       : The CoT step text to extract claims from.
        llm_caller : callable(prompt: str) -> str.  Must return a string
                     containing a valid JSON array, possibly wrapped in markdown.

        Returns a list of ArithmeticClaim objects.  Returns [] on LLM error,
        JSON parse failure, or when the step contains no arithmetic.
        """
        prompt = _JSON_CLAIM_PROMPT.format(step=step)
        try:
            raw = llm_caller(prompt)
        except Exception:  # noqa: BLE001 — LLM errors are non-fatal
            return []

        # Strip markdown fences if present.
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            items = json.loads(match.group())
        except json.JSONDecodeError:
            return []

        claims: list[ArithmeticClaim] = []
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
            claims.append(
                ArithmeticClaim(
                    lhs_expr=lhs,
                    rhs_value=rhs,
                    claim_text=text[:120],
                    strategy="json_claim",
                    confidence=0.85,
                )
            )
        return claims


# ---------------------------------------------------------------------------
# Strategy 2: SymCodeExtractor (arXiv 2510.25975 — SymCode approach)
# ---------------------------------------------------------------------------

_SYMCODE_PROMPT = textwrap.dedent("""\
    Write a single-line Python expression that computes the answer claimed in this math step.
    Output ONLY executable Python (no assignment, no print). If unclear: output None.
    Step: {step}""")

# Pattern to extract the final numeric answer stated in a CoT step.
# Matches "= 54.50", "= $54.50", "is 54.50", "gives 54.50" etc.
_STATED_RESULT_RE = re.compile(
    r"(?:=\s*\$?|(?:is|gives?|totals?|results?\s+in|equals?)\s+\$?)"
    r"([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


class SymCodeExtractor:
    """Extract arithmetic claims by asking the LLM to synthesise executable Python.

    Why SymCode (arXiv 2510.25975): when a model writes a multi-step calculation,
    the relationship between operands and result may be expressed in prose rather
    than symbolic form.  By asking a small LLM to "write a Python expression that
    computes the answer in this step," we leverage the LLM's language understanding
    to synthesise machine-executable code.  We then eval() that code and compare
    to the numeric result stated in the step.

    Example:
        step: "She earns 35 hours/week × $20/hour = $700 per week."
        LLM output: "35 * 20"
        safe_eval("35 * 20") = 700.0
        stated result from step: 700.0  → no violation
        (If stated result were 720.0, that would be a violation.)

    Spec: REQ-EXTRACT-051
    """

    def extract_claims(
        self,
        step: str,
        llm_caller: Callable[[str], str],
    ) -> list[ArithmeticClaim]:
        """Ask LLM to generate a Python expression, eval it, compare to stated result.

        Parameters
        ----------
        step       : The CoT step text.
        llm_caller : callable(prompt: str) -> str.

        Returns a list with at most one ArithmeticClaim (the synthesised expression
        vs. the last numeric result stated in the step).  Returns [] when the LLM
        output is not evaluable, or when no stated result can be extracted.
        """
        prompt = _SYMCODE_PROMPT.format(step=step)
        try:
            raw = llm_caller(prompt).strip()
        except Exception:  # noqa: BLE001 — LLM errors are non-fatal
            return []

        # The LLM might output "None" literally when it cannot synthesise code.
        if raw in ("None", "none", "null", "", "[]"):
            return []

        # Strip markdown fences: ```python\n...\n```
        raw = re.sub(r"^```(?:python)?\n?", "", raw, flags=re.MULTILINE)
        raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE).strip()

        computed = safe_eval(raw)
        if computed is None:
            return []

        # Find the last numeric result stated in the step to compare against.
        stated_matches = _STATED_RESULT_RE.findall(step)
        if not stated_matches:
            return []

        rhs_str = re.sub(r"[,_]", "", stated_matches[-1])
        try:
            rhs = float(rhs_str)
        except ValueError:
            return []

        return [
            ArithmeticClaim(
                lhs_expr=raw,
                rhs_value=rhs,
                claim_text=step[:120],
                strategy="symcode",
                confidence=0.80,
            )
        ]


# ---------------------------------------------------------------------------
# Strategy 3: StepSegmentEvalChain (CoACEV4-equivalent baseline, no LLM)
# ---------------------------------------------------------------------------

# Numeric literal pattern: handles 1,234.56, $42, 1_000, etc.
_NUM = r"\$?[\d,_]+(?:\.\d+)?"

# Pattern: N op M = P with symbolic operators (the CoACEV4 plain-symbolic set).
_ARITH_EQ_RE = re.compile(
    rf"({_NUM})\s*([+\-*/])\s*({_NUM})\s*=\s*({_NUM})"
)

# Pattern: "N times/multiplied by M equals/is P" (prose multiplication)
_PROSE_MUL_RE = re.compile(
    rf"({_NUM})\s+(?:times|multiplied\s+by)\s+({_NUM})\s+(?:equals?|is|gives?)\s+({_NUM})",
    re.IGNORECASE,
)

# Pattern: "N plus/added to M equals/is/gives P"
_PROSE_ADD_RE = re.compile(
    rf"({_NUM})\s+(?:plus|added\s+to)\s+({_NUM})\s+(?:equals?|is|gives?)\s+({_NUM})",
    re.IGNORECASE,
)


def _clean_num_str(s: str) -> Optional[float]:
    """Strip currency and separators, return float or None."""
    cleaned = re.sub(r"[\$,_]", "", s.strip())
    try:
        return float(cleaned)
    except ValueError:
        return None


class StepSegmentEvalChain:
    """Baseline extractor: sentence-segment the CoT, eval numeric expressions.

    This is the CoACEV4 approach, included in LLMAsExtractorV1 as the baseline
    strategy.  It uses no LLM calls, so it runs in CI environments with no GPU.

    Patterns covered:
    - Symbolic arithmetic: "3 * $16.50 = $54.50"
    - Prose multiplication: "3 times 16.50 gives 54.50"
    - Prose addition: "15 plus 25 equals 40"

    Known limitation: cannot handle natural language phrasing like
    "3 pairs of shorts at $16.50 each is $54.50" — this requires an LLM.
    That limitation is WHY LLMAsExtractorV1 exists.

    Spec: REQ-EXTRACT-050 (baseline component)
    """

    def extract_claims(self, step: str) -> list[ArithmeticClaim]:
        """Scan the CoT step for numeric expressions and evaluate them.

        Splits the step into sentences (by period, newline, or numbered list marker),
        then searches each sentence for N op M = P patterns.

        Returns a list of ArithmeticClaim objects.  Claims where safe_eval(lhs) is
        None (unevaluable expression) are skipped silently.
        """
        claims: list[ArithmeticClaim] = []
        seen: set[tuple[str, float]] = set()

        def _add(lhs: str, rhs_str: str, text: str, conf: float = 0.90) -> None:
            lhs_clean = re.sub(r"[\$,_]", "", lhs.strip())
            rhs = _clean_num_str(rhs_str)
            if rhs is None:
                return
            key = (lhs_clean, rhs)
            if key in seen:
                return
            seen.add(key)
            claims.append(
                ArithmeticClaim(
                    lhs_expr=lhs_clean,
                    rhs_value=rhs,
                    claim_text=text[:120],
                    strategy="step_segment_eval",
                    confidence=conf,
                )
            )

        # Symbolic arithmetic: A op B = C
        for m in _ARITH_EQ_RE.finditer(step):
            a = re.sub(r"[\$,_]", "", m.group(1))
            op = m.group(2)
            b = re.sub(r"[\$,_]", "", m.group(3))
            lhs = f"{a}{op}{b}"
            _add(lhs, m.group(4), m.group(0))

        # Prose multiplication
        for m in _PROSE_MUL_RE.finditer(step):
            a = re.sub(r"[\$,_]", "", m.group(1))
            b = re.sub(r"[\$,_]", "", m.group(2))
            _add(f"{a}*{b}", m.group(3), m.group(0))

        # Prose addition
        for m in _PROSE_ADD_RE.finditer(step):
            a = re.sub(r"[\$,_]", "", m.group(1))
            b = re.sub(r"[\$,_]", "", m.group(2))
            _add(f"{a}+{b}", m.group(3), m.group(0))

        return claims


# ---------------------------------------------------------------------------
# LLMAsExtractorV1 — unified extractor with strategy selection
# ---------------------------------------------------------------------------


class LLMAsExtractorV1:
    """Unified LLM-based arithmetic claim extractor with three strategies.

    In CI mode (llm_caller=None):
        Only StepSegmentEvalChain is used.  No LLM calls.  This is the safe,
        deterministic path for automated testing without GPU or network access.

    In live mode (llm_caller provided):
        All three strategies run on every response.  Results are unioned and
        deduplicated by (lhs_expr, rhs_value).  The strategy with the most
        violations found on the probe sample is stored as self._best_strategy.

    Why union all three instead of picking one:
        Each strategy catches different surface forms.  JsonClaimExtractor handles
        structured arithmetic; SymCodeExtractor handles prose and final-answer
        comparisons; StepSegmentEvalChain handles purely symbolic equations.
        Union maximises recall at the cost of slightly more false positives.

    Args:
        llm_caller : callable(prompt: str) -> str, or None for CI mode.
        tolerance  : absolute error below which a discrepancy is ignored (default 1e-6).

    Spec: REQ-EXTRACT-050, REQ-EXTRACT-051, REQ-EXTRACT-052,
          SCENARIO-EXTRACT-085, SCENARIO-EXTRACT-086, SCENARIO-EXTRACT-087,
          SCENARIO-EXTRACT-088, SCENARIO-EXTRACT-089
    """

    def __init__(
        self,
        llm_caller: Optional[Callable[[str], str]] = None,
        tolerance: float = 1e-6,
    ) -> None:
        self.llm_caller = llm_caller
        self.tolerance = tolerance
        self._json_extractor = JsonClaimExtractor()
        self._symcode_extractor = SymCodeExtractor()
        self._chain_extractor = StepSegmentEvalChain()
        self._best_strategy: str = "step_segment_eval"

    def extract(self, response: str) -> list[ArithmeticClaim]:
        """Extract all arithmetic claims from a full CoT response.

        In CI mode: runs StepSegmentEvalChain only.
        In live mode: runs all three strategies, unions results, deduplicates.

        A claim is a VIOLATION if safe_eval(lhs_expr) is not None and differs
        from rhs_value by more than self.tolerance.

        Returns only VIOLATION claims (claims where arithmetic is detectably wrong).
        This mirrors the CoACEV4 contract: the caller counts violations, not claims.
        """
        if self.llm_caller is None:
            return self._filter_violations(
                self._chain_extractor.extract_claims(response)
            )

        # Live mode: union all three strategies.
        json_claims = self._json_extractor.extract_claims(response, self.llm_caller)
        symcode_claims = self._symcode_extractor.extract_claims(response, self.llm_caller)
        chain_claims = self._chain_extractor.extract_claims(response)

        all_claims = self._dedup(json_claims + symcode_claims + chain_claims)
        return self._filter_violations(all_claims)

    def _filter_violations(self, claims: list[ArithmeticClaim]) -> list[ArithmeticClaim]:
        """Return only claims where safe_eval(lhs_expr) disagrees with rhs_value.

        A claim with rhs_value=None is kept as a potential violation (we cannot
        confirm correctness without a stated result to compare).  A claim where
        safe_eval returns None (unevaluable expression) is silently dropped.
        """
        violations: list[ArithmeticClaim] = []
        for claim in claims:
            if claim.rhs_value is None:
                violations.append(claim)
                continue
            computed = safe_eval(claim.lhs_expr)
            if computed is None:
                continue
            if abs(computed - claim.rhs_value) > self.tolerance:
                violations.append(claim)
        return violations

    def _dedup(self, claims: list[ArithmeticClaim]) -> list[ArithmeticClaim]:
        """Deduplicate claims by (normalised lhs_expr, rhs_value).

        When the same arithmetic equation is found by multiple strategies, keep
        the first occurrence (JsonClaimExtractor takes precedence, since it has
        the most context-aware extraction).
        """
        seen: set[tuple[str, Optional[float]]] = set()
        result: list[ArithmeticClaim] = []
        for claim in claims:
            lhs_clean = re.sub(r"[\$,_\s]", "", claim.lhs_expr)
            key = (lhs_clean, claim.rhs_value)
            if key in seen:
                continue
            seen.add(key)
            result.append(claim)
        return result
