"""CoACEExtractorV4 — GenPRM-style data-driven arithmetic extraction.

**Why V4 exists (RETRO-068 root cause):**

    Three consecutive versions of hand-engineered regex patterns (V1, V2, V3) achieved
    recall of 0%, 5.9%, and 4% respectively on live IT-model outputs.  The root cause is
    not missing patterns — it is the offline/live distribution gap:

    - The live test set (Exp 591) is dominated by "The answer is 42" placeholder responses
      (16/25 = 64%) with no arithmetic to extract.  No regex can help here.
    - Of the remaining 9 real responses, the majority contain LOGIC errors (wrong premise)
      with arithmetically correct steps.  The model computes 60 × 3 = 180 correctly but
      applies the wrong multiplier — no arithmetic extractor can flag this without knowing
      the correct setup.
    - A minority of responses use LaTeX math notation (\times, \frac, display blocks) that
      V3's plain-text regexes cannot parse.

    The GenPRM approach (arXiv 2504.00891) takes a fundamentally different stance:
    instead of enumerating surface patterns, a small LLM is prompted to "identify ALL
    arithmetic calculations in this CoT step."  The LLM handles surface variation
    naturally.  V4 implements this as the primary path, with a regex fallback for CI/test.

**Architecture:**

    1. ArithmeticClaim — structured claim extracted from a CoT response.
    2. GenPRMExtractor — claim extractor:
          - LLM path: prompt → JSON → ArithmeticClaim list.
          - CI stub path: regex fallback covering LaTeX + Unicode × + plain prose.
    3. CoACEExtractorV4(CoACEExtractorV3) — runs V3, then GenPRM; merges violations.

**CI stub coverage (regex fallback, no LLM):**

    The fallback adds patterns NOT covered by V3:
    - LaTeX inline math: \(N \times M = P\) or \(N \cdot M = P\)
    - LaTeX display blocks: \[...\] with = sign
    - Unicode × operator: N × M = P
    - Plain symbolic: N * M = P (V3 requires a $ sign; V4 drops that requirement)
    - Division: N / M = P or N ÷ M = P
    - Unicode ÷ operator: N ÷ M = P

Spec: REQ-EXTRACT-045, REQ-EXTRACT-046,
      SCENARIO-EXTRACT-080, SCENARIO-EXTRACT-081, SCENARIO-EXTRACT-082,
      SCENARIO-EXTRACT-083
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Callable, Optional

from carnot.extraction.coace_extractor import (
    CoACEResult,
    CoACEViolation,
    ArithmeticEquation,
)
from carnot.extraction.coace_extractor_v3 import CoACEExtractorV3

# ---------------------------------------------------------------------------
# ArithmeticClaim — the GenPRM unit of extraction
# ---------------------------------------------------------------------------


@dataclass
class ArithmeticClaim:
    """One arithmetic calculation identified in a CoT response.

    Why a separate dataclass from ArithmeticEquation: ArithmeticClaim represents
    a GenPRM extraction result with a verbatim text excerpt and LLM confidence score,
    before the claim is evaluated.  ArithmeticEquation is the V1/V2/V3 internal
    representation that feeds the violation check.  Keeping them separate avoids
    polluting the V3 schema with LLM-specific fields.

    Fields:
        lhs_expr    — Python-evaluable expression for the left-hand side.
                      Examples: "7*1.5", "130000+195000", "60*3".
        rhs_value   — The number the model STATED as the result.
        claim_text  — Verbatim excerpt from the response for traceability.
        confidence  — Extraction confidence 0–1.  LLM path: from the model.
                      Regex path: fixed at 0.85 (regex has no uncertainty).
    """

    lhs_expr: str
    rhs_value: float
    claim_text: str
    confidence: float


# ---------------------------------------------------------------------------
# EXTRACTION_PROMPT — the GenPRM-style LLM prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = (
    "Given the following math reasoning step, identify ALL arithmetic calculations"
    " stated (not just final answers). For each calculation, output JSON:"
    ' {{"lhs": "expression_to_evaluate", "rhs": stated_number_value,'
    ' "text": "verbatim_text_excerpt"}}.'
    " Output only a valid JSON array. If no calculations found: output []."
    "\nResponse:\n{response}"
)

# ---------------------------------------------------------------------------
# Regex building blocks for the CI stub fallback
# ---------------------------------------------------------------------------

_NUM_PLAIN = r"\d+(?:[,_]\d+)*(?:\.\d+)?"  # plain non-currency number
_NUM_CURRENCY = r"\$?" + _NUM_PLAIN  # optional leading $

# LaTeX inline: \(expr = number\)  — capture everything inside \( \)
_LATEX_INLINE = re.compile(r"\\\(([^)]*?)\\\)")

# LaTeX display block: \[...\]
_LATEX_DISPLAY = re.compile(r"\\\[([^\]]*?)\\\]", re.DOTALL)

# Unicode × and ÷ operators between two numbers: N × M = P or N ÷ M = P
_UNICODE_MUL = re.compile(
    rf"({_NUM_CURRENCY})\s*×\s*({_NUM_CURRENCY})\s*=\s*({_NUM_CURRENCY})",
)
_UNICODE_DIV = re.compile(
    rf"({_NUM_CURRENCY})\s*÷\s*({_NUM_CURRENCY})\s*=\s*({_NUM_CURRENCY})",
)

# Plain symbolic without $ requirement: N * M = P, N / M = P, N + M = P, N - M = P
# V3 already covers these when a $ is present.  V4 covers the $ -free forms.
_PLAIN_MUL = re.compile(
    rf"(?<!\$)({_NUM_PLAIN})\s*\*\s*({_NUM_PLAIN})\s*=\s*({_NUM_PLAIN})(?!\d)",
)
_PLAIN_DIV = re.compile(
    rf"(?<!\$)({_NUM_PLAIN})\s*/\s*({_NUM_PLAIN})\s*=\s*({_NUM_PLAIN})(?!\d)",
)
_PLAIN_ADD = re.compile(
    rf"(?<!\$)({_NUM_PLAIN})\s*\+\s*({_NUM_PLAIN})\s*=\s*({_NUM_PLAIN})(?!\d)",
)
_PLAIN_SUB = re.compile(
    rf"(?<!\$)({_NUM_PLAIN})\s*-\s*({_NUM_PLAIN})\s*=\s*({_NUM_PLAIN})(?!\d)",
)

# LaTeX \times and \cdot inside a block: N \times M = P or N \cdot M = P
_LATEX_TIMES = re.compile(
    rf"({_NUM_PLAIN})\s*\\(?:times|cdot)\s*({_NUM_PLAIN})\s*=\s*({_NUM_PLAIN})",
)

# LaTeX \div or \frac: \frac{N}{M} = P  or  N \div M = P
_LATEX_DIV_FRAC = re.compile(
    r"\\frac\s*\{(" + _NUM_PLAIN + r")\}\s*\{(" + _NUM_PLAIN + r")\}\s*=\s*(" + _NUM_PLAIN + r")",
)
_LATEX_DIV_OP = re.compile(
    rf"({_NUM_PLAIN})\s*\\div\s*({_NUM_PLAIN})\s*=\s*({_NUM_PLAIN})",
)


def _clean_num(s: str) -> Optional[float]:
    """Parse a number string, stripping commas and underscores, ignoring currency.

    Returns None if the string cannot be converted to float.  This is a
    permissive parser — it handles '1,234.56', '1_000', '$42.00', etc.

    Why: extracted LaTeX and prose numbers may include thousands separators that
    Python's float() rejects; stripping them first avoids silent extraction failure.
    """
    cleaned = re.sub(r"[\$,_]", "", s.strip())
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# safe_eval — restricted arithmetic evaluation
# ---------------------------------------------------------------------------


def safe_eval(expr: str) -> Optional[float]:
    """Evaluate a restricted arithmetic expression and return the float result.

    Allowed: numeric literals, +, -, *, /, **, % operators, parentheses.
    Blocked: any identifier, attribute access, function call, or builtin.

    Why restricted: the lhs_expr may originate from an LLM response or regex
    capture.  A naive eval() call would execute arbitrary Python.  We parse with
    ast.parse() and walk the AST to whitelist only safe node types before eval.

    Returns None on parse error, unsafe content, division-by-zero, or overflow.
    """
    # Normalise: strip currency and commas so '7*$1.5' evaluates correctly.
    cleaned = re.sub(r"[\$,]", "", expr.strip())
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError:
        return None

    # Walk the AST and reject any node that isn't a literal or numeric operation.
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
# GenPRMExtractor
# ---------------------------------------------------------------------------


class GenPRMExtractor:
    """Extract arithmetic claims from a CoT response using GenPRM-style prompting.

    Two modes:
        LLM mode (llm_caller provided): send EXTRACTION_PROMPT to the LLM, parse
            the returned JSON array of {lhs, rhs, text} objects.  This handles
            arbitrary surface variation in the model's output.

        CI stub mode (llm_caller=None): fall back to a deterministic regex sweep
            covering LaTeX inline/display math, Unicode × / ÷, and plain N*M=P
            patterns not already covered by V3.  No LLM call is made, so this mode
            is safe for CI pipelines with no GPU/network access.

    After extraction, each claim is verified with safe_eval().  Claims where
    safe_eval(lhs_expr) != rhs_value (within tolerance) become violations.

    Args:
        llm_caller: callable(prompt: str) -> str, or None for CI stub mode.
            Receives the full EXTRACTION_PROMPT text; must return a string
            containing a valid JSON array.
        tolerance: absolute error threshold below which a discrepancy is ignored.
            Matches CoACEExtractorV3's default of 1e-6.
    """

    def __init__(
        self,
        llm_caller: Optional[Callable[[str], str]] = None,
        tolerance: float = 1e-6,
    ) -> None:
        self.llm_caller = llm_caller
        self.tolerance = tolerance

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract_claims(self, response: str) -> list[ArithmeticClaim]:
        """Identify all arithmetic claims in a CoT response.

        LLM path: prompt the LLM → parse JSON → return claims.
        Fallback path: regex sweep → return claims.

        Returns an empty list if no claims are found (e.g., placeholder responses
        like "The answer is 42" with no reasoning).
        """
        if self.llm_caller is not None:
            return self._llm_extract(response)
        return self._regex_extract(response)

    # ------------------------------------------------------------------
    # LLM extraction path
    # ------------------------------------------------------------------

    def _llm_extract(self, response: str) -> list[ArithmeticClaim]:
        """Call the LLM with EXTRACTION_PROMPT and parse the JSON result.

        If the LLM returns malformed JSON or an unexpected structure, fall back
        to the regex path rather than crashing.  This defensive pattern ensures
        the pipeline degrades gracefully when the LLM is confused.
        """
        prompt = EXTRACTION_PROMPT.format(response=response)
        try:
            raw = self.llm_caller(prompt)
            # Extract JSON array — the LLM may wrap it in markdown fences.
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return self._regex_extract(response)
            items = json.loads(match.group())
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
                conf = float(item.get("confidence", 0.85))
                claims.append(ArithmeticClaim(
                    lhs_expr=lhs,
                    rhs_value=rhs,
                    claim_text=text,
                    confidence=conf,
                ))
            return claims
        except Exception:  # noqa: BLE001 — LLM errors are non-fatal
            return self._regex_extract(response)

    # ------------------------------------------------------------------
    # Regex fallback (CI stub) extraction path
    # ------------------------------------------------------------------

    def _regex_extract(self, response: str) -> list[ArithmeticClaim]:
        """Sweep the response with regex patterns not covered by V3.

        Patterns covered (V3 already handles these, so we skip duplicates later):
            - LaTeX \times: \(7 \times 1.5 = 10.5\)
            - LaTeX \cdot: \(4 \cdot 5 = 20\)
            - LaTeX \div: \(10 \div 2 = 5\)
            - LaTeX \frac: \frac{200}{2} = 100
            - Unicode ×: 60 × 3 = 180
            - Unicode ÷: 10 ÷ 2 = 5
            - Plain multiplication without $: 7 * 1.5 = 10.5
            - Plain division without $: 90 / 7 = 12
            - Plain addition without $: 15 + 25 = 40
            - Plain subtraction without $: 100 - 30 = 70

        Why extract LaTeX blocks first: \( ... \) and \[ ... \] blocks sometimes
        contain the full equation; applying \times / \cdot patterns inside the
        extracted block avoids false positives from surrounding prose numbers.
        """
        claims: list[ArithmeticClaim] = []
        seen: set[tuple[str, float]] = set()

        def _add(lhs: str, rhs_str: str, text: str, conf: float = 0.85) -> None:
            rhs = _clean_num(rhs_str)
            if rhs is None:
                return
            # Normalise lhs for dedup
            lhs_clean = re.sub(r"[\$,]", "", lhs.strip())
            key = (lhs_clean, rhs)
            if key in seen:
                return
            seen.add(key)
            claims.append(ArithmeticClaim(
                lhs_expr=lhs_clean,
                rhs_value=rhs,
                claim_text=text[:120],
                confidence=conf,
            ))

        # Step 1: extract LaTeX blocks and scan inside them for \times / \cdot / \div
        for block_pat in (_LATEX_INLINE, _LATEX_DISPLAY):
            for bm in block_pat.finditer(response):
                block = bm.group(1)
                raw = bm.group(0)
                for m in _LATEX_TIMES.finditer(block):
                    a, b, r = m.group(1), m.group(2), m.group(3)
                    _add(f"{a}*{b}", r, raw)
                for m in _LATEX_DIV_FRAC.finditer(block):
                    a, b, r = m.group(1), m.group(2), m.group(3)
                    _add(f"{a}/{b}", r, raw)
                for m in _LATEX_DIV_OP.finditer(block):
                    a, b, r = m.group(1), m.group(2), m.group(3)
                    _add(f"{a}/{b}", r, raw)

        # Step 2: scan full text for LaTeX \times outside of block captures
        for m in _LATEX_TIMES.finditer(response):
            a, b, r = m.group(1), m.group(2), m.group(3)
            _add(f"{a}*{b}", r, m.group(0))

        # Step 3: \frac and \div outside blocks
        for m in _LATEX_DIV_FRAC.finditer(response):
            a, b, r = m.group(1), m.group(2), m.group(3)
            _add(f"{a}/{b}", r, m.group(0))
        for m in _LATEX_DIV_OP.finditer(response):
            a, b, r = m.group(1), m.group(2), m.group(3)
            _add(f"{a}/{b}", r, m.group(0))

        # Step 4: Unicode × and ÷
        for m in _UNICODE_MUL.finditer(response):
            a = re.sub(r"\$", "", m.group(1))
            b = re.sub(r"\$", "", m.group(2))
            r = m.group(3)
            _add(f"{a}*{b}", r, m.group(0))
        for m in _UNICODE_DIV.finditer(response):
            a = re.sub(r"\$", "", m.group(1))
            b = re.sub(r"\$", "", m.group(2))
            r = m.group(3)
            _add(f"{a}/{b}", r, m.group(0))

        # Step 5: plain symbolic (no $ required — V3 requires $)
        for m in _PLAIN_MUL.finditer(response):
            _add(f"{m.group(1)}*{m.group(2)}", m.group(3), m.group(0))
        for m in _PLAIN_DIV.finditer(response):
            _add(f"{m.group(1)}/{m.group(2)}", m.group(3), m.group(0))
        for m in _PLAIN_ADD.finditer(response):
            _add(f"{m.group(1)}+{m.group(2)}", m.group(3), m.group(0))
        for m in _PLAIN_SUB.finditer(response):
            _add(f"{m.group(1)}-{m.group(2)}", m.group(3), m.group(0))

        return claims


# ---------------------------------------------------------------------------
# CoACEExtractorV4
# ---------------------------------------------------------------------------


class CoACEExtractorV4(CoACEExtractorV3):
    """V4 of the COACE extractor — GenPRM-style data-driven arithmetic extraction.

    **Why V4 extends V3 rather than replacing it:**
        V3 patterns are precise for the patterns they cover.  V4 uses them as a
        first pass, then runs GenPRMExtractor on the same response to catch claims
        that the V3 regex suite could not express (LaTeX notation, Unicode operators,
        LLM-generated JSON claims).  The two results are merged and deduplicated.

    **CI stub vs. LLM mode:**
        pass llm_caller=None (default) for deterministic CI behaviour.
        pass a callable(str)->str to enable LLM-based claim extraction in live runs.

    Spec: REQ-EXTRACT-045, REQ-EXTRACT-046,
          SCENARIO-EXTRACT-080, SCENARIO-EXTRACT-081, SCENARIO-EXTRACT-082,
          SCENARIO-EXTRACT-083
    """

    def __init__(
        self,
        llm_caller: Optional[Callable[[str], str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._genprm = GenPRMExtractor(
            llm_caller=llm_caller,
            tolerance=self.tolerance,
        )

    def extract(self, response: str) -> CoACEResult:
        """Run V3 extraction then GenPRM extraction; merge and deduplicate results.

        Steps:
        1. Call CoACEExtractorV3.extract() to get V3 violations and seen equations.
        2. Run GenPRMExtractor.extract_claims() on the same response.
        3. For each GenPRM claim not already in the V3 seen-set, evaluate via
           safe_eval() and emit a violation if the arithmetic is wrong.
        4. Merge V3 + V4 violations; return a CoACEResult with mode='genprm_v4'.

        Deduplication key: (normalised lhs_expr, rhs_value) — same as V3.
        """
        # Step 1: V3 base extraction
        v3_result = super().extract(response)

        # Rebuild V3 seen set from reported violations + re-running V3 parsers
        seen_pairs: set[tuple[str, float]] = {
            (re.sub(r"[\$,]", "", v.equation.lhs_expr), v.equation.rhs_value)
            for v in v3_result.violations
        }

        # Also populate seen from all V3-sourced equations (including non-violations)
        from carnot.extraction.coace_extractor import _parse_arithmetic_equations
        from carnot.extraction.coace_extractor_v2 import (
            _parse_prose_arithmetic,
            _extract_chain_equations,
        )
        from carnot.extraction.coace_extractor_v3 import (
            _parse_narrative_arithmetic,
            _parse_percentage_word_problem,
            _parse_unit_conversion,
            _parse_running_total_chain,
        )

        for eq in (
            _parse_arithmetic_equations(response)
            + _parse_prose_arithmetic(response)
            + _extract_chain_equations(response)
            + _parse_narrative_arithmetic(response)
            + _parse_percentage_word_problem(response)
            + _parse_unit_conversion(response)
            + _parse_running_total_chain(response)
        ):
            seen_pairs.add((re.sub(r"[\$,]", "", eq.lhs_expr), eq.rhs_value))

        # Step 2: GenPRM claim extraction
        genprm_claims = self._genprm.extract_claims(response)

        # Step 3: evaluate new claims only
        new_violations: list[CoACEViolation] = []
        new_eq_count = 0
        for claim in genprm_claims:
            if claim.confidence < self.min_confidence:
                continue
            lhs_clean = re.sub(r"[\$,]", "", claim.lhs_expr)
            key = (lhs_clean, claim.rhs_value)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            new_eq_count += 1

            computed = safe_eval(claim.lhs_expr)
            if computed is None:
                continue
            abs_err = abs(computed - claim.rhs_value)
            if abs_err > self.tolerance:
                rel_err = abs_err / max(abs(claim.rhs_value), 1e-12)
                eq = ArithmeticEquation(
                    lhs_expr=claim.lhs_expr,
                    rhs_value=claim.rhs_value,
                    stated_result=str(claim.rhs_value),
                    confidence=claim.confidence,
                )
                new_violations.append(
                    CoACEViolation(
                        equation=eq,
                        computed_value=computed,
                        stated_value=claim.rhs_value,
                        absolute_error=abs_err,
                        relative_error=rel_err,
                        is_violation=True,
                    )
                )

        # Step 4: merge
        all_violations = v3_result.violations + new_violations
        n_eqs = v3_result.n_equations_found + new_eq_count
        conf_weighted = sum(1 for v in all_violations if v.equation.confidence >= 0.8)

        return CoACEResult(
            n_equations_found=n_eqs,
            n_violations=len(all_violations),
            violations=all_violations,
            extraction_mode="genprm_v4",
            confidence_weighted_violations=conf_weighted,
        )
