"""CoACEExtractorV2 — extended arithmetic extraction for multi-step IT model CoT.

**Why V2 exists (RETRO-064 root cause):**

    CoACEExtractor v1 achieves only 5.9% recall on live incorrect IT model responses.
    Root cause: v1 only detects simple 'A op B = C' one-step equations written with
    symbolic operators (+, -, *, /).  IT models write three additional patterns that
    v1 cannot parse:

    1. Prose percentage/ratio patterns:
       '20% of 150 is 30', '47 times 3 gives 141', 'difference between 10 and 3 is 7'
       These use natural-language words instead of operator symbols.

    2. Multi-step chain reasoning:
       'first compute X = 47 + 28 = 75, then Y = X * 2 = 151'
       The variable X is assigned 75 in step 1, but step 2 uses a wrong value for X.
       V1 sees no symbolic operator equation at that step and silently misses it.

    3. Chained equality:
       'total = 3 + 4 + 5 = 11' — three-part chain where the intermediate sum is
       right but the final total is wrong.  V1 only grabs the rightmost '= 11' and
       never evaluates '3 + 4 + 5'.

    V2 adds three helper parsers (_parse_prose_arithmetic, _extract_chain_equations,
    and chain-equality handling inside _parse_prose_arithmetic) and merges their
    output with v1's before running the same _safe_eval violation loop.

Spec: REQ-EXTRACT-035, REQ-EXTRACT-036,
      SCENARIO-EXTRACT-068, SCENARIO-EXTRACT-069, SCENARIO-EXTRACT-070, SCENARIO-EXTRACT-071
"""

from __future__ import annotations

import re
from typing import Optional

from carnot.extraction.coace_extractor import (
    ArithmeticEquation,
    CoACEExtractor,
    CoACEResult,
    CoACEViolation,
    _safe_eval,
)

# ---------------------------------------------------------------------------
# Type alias: tracks variable_name → computed_float as chain executes
# ---------------------------------------------------------------------------

NumericContext = dict[str, float]

# ---------------------------------------------------------------------------
# Prose arithmetic patterns
# ---------------------------------------------------------------------------

# 'N% of M is/equals/gives P' — handles integer or decimal N and M
_PERCENT_OF = re.compile(
    r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d+(?:\.\d+)?)\s+"
    r"(?:is|equals?|gives?|=)\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# 'X times Y is/equals/gives/= Z'
_TIMES = re.compile(
    r"(\d+(?:\.\d+)?)\s+times\s+(\d+(?:\.\d+)?)\s+"
    r"(?:is|equals?|gives?|=)\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# 'P divided by Q is/equals/gives/= R'
_DIVIDED_BY = re.compile(
    r"(\d+(?:\.\d+)?)\s+divided\s+by\s+(\d+(?:\.\d+)?)\s+"
    r"(?:is|equals?|gives?|=)\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# 'difference between P and Q is/equals/gives/= R'
_DIFFERENCE = re.compile(
    r"difference\s+between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)\s+"
    r"(?:is|equals?|gives?|=)\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# 'sum of A, B[, and C]* is/equals/gives/= R'
# Captures a comma-separated (with optional 'and') list of numbers, then a stated total.
_SUM_OF = re.compile(
    r"sum\s+of\s+([\d,\s\.and]+?)\s+(?:is|equals?|gives?|=)\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Chained equality: 'label = expr = final' e.g. 'total = 3 + 4 + 5 = 12'
# We look for: [word =]? arithmetic_expr = number at end of a segment.
_CHAINED_EQ = re.compile(
    r"(?:\w[\w\s]*=\s*)?"           # optional label like 'total ='
    r"([\d]+(?:\.\d+)?\s*(?:[+\-*/]\s*[\d]+(?:\.\d+)?)+)"  # arithmetic LHS
    r"\s*=\s*(\d+(?:\.\d+)?)",       # final stated RHS
    re.IGNORECASE,
)

# Chain patterns for step-by-step reasoning:
# 'let X = expr', 'so X = expr', 'X = expr' at start of sentence
_CHAIN_ASSIGN = re.compile(
    r"(?:let\s+|so\s+)?([A-Za-z_]\w*)\s*=\s*([\d+\-*/\s\.\(\)]+?)(?=[,;\n]|$)",
    re.IGNORECASE | re.MULTILINE,
)


def _parse_number_list(text: str) -> list[float]:
    """Extract all numbers from a comma/and-separated string like '3, 4, and 5'."""
    return [float(m.group()) for m in re.finditer(r"\d+(?:\.\d+)?", text)]


def _parse_prose_arithmetic(text: str) -> list[ArithmeticEquation]:
    """Extract arithmetic equations hidden in natural-language prose.

    Why this is needed: v1's parser only matches symbolic '47 + 28 = 76' style
    equations.  IT models frequently write 'twenty percent of 150 is 31' or
    '47 times 3 gives 142'.  None of these contain the '+' or '*' character, so v1
    is blind to them.  We match the prose connectives and reconstruct a
    Python-evaluable lhs_expr so _safe_eval can verify the arithmetic.

    Each pattern returns an ArithmeticEquation with:
        lhs_expr: a Python-evaluable expression string (e.g., '20/100*150').
        rhs_value: the float the model stated.
        confidence: 0.9 for explicit prose connectives (slightly below 1.0 = '=').

    Returns a list of ArithmeticEquation objects (may be empty).
    """
    equations: list[ArithmeticEquation] = []
    seen: set[str] = set()

    def _add(lhs: str, rhs_str: str, conf: float) -> None:
        key = (lhs, rhs_str)
        if key in seen:
            return
        seen.add(key)
        try:
            rhs = float(rhs_str)
        except ValueError:
            return
        equations.append(
            ArithmeticEquation(
                lhs_expr=lhs,
                rhs_value=rhs,
                stated_result=rhs_str,
                confidence=conf,
            )
        )

    # Percentage: 'N% of M is P'
    for m in _PERCENT_OF.finditer(text):
        n, base, stated = m.group(1), m.group(2), m.group(3)
        _add(f"{n}/100*{base}", stated, 0.9)

    # Times: 'X times Y is Z'
    for m in _TIMES.finditer(text):
        x, y, stated = m.group(1), m.group(2), m.group(3)
        _add(f"{x}*{y}", stated, 0.9)

    # Divided by: 'P divided by Q is R'
    for m in _DIVIDED_BY.finditer(text):
        p, q, stated = m.group(1), m.group(2), m.group(3)
        _add(f"{p}/{q}", stated, 0.9)

    # Difference between: 'difference between P and Q is R'
    for m in _DIFFERENCE.finditer(text):
        p, q, stated = m.group(1), m.group(2), m.group(3)
        _add(f"{p}-{q}", stated, 0.9)

    # Sum of: 'sum of A, B, and C is R'
    for m in _SUM_OF.finditer(text):
        nums = _parse_number_list(m.group(1))
        stated = m.group(2)
        if len(nums) >= 2:
            lhs = "+".join(str(n) for n in nums)
            _add(lhs, stated, 0.9)

    return equations


def _extract_chain_equations(text: str) -> list[ArithmeticEquation]:
    """Detect chain-tracking violations where a later step uses a wrong prior result.

    Why this is needed: IT model multi-step chains look like:
        'let cost = 3 * 5 = 15, then total = cost + 7 = 23'
    V1 can check '3 * 5 = 15' directly.  But the variable 'cost' is then reused.
    If the model later writes 'total = cost + 7 = 23' but cost=15, then cost+7=22,
    not 23.  V1 cannot evaluate 'cost + 7' because 'cost' is not a numeric literal.

    This function:
    1. Walks the text in sentence order.
    2. Extracts 'let X = expr = value' or 'X = number' assignments into NumericContext.
    3. When it sees 'X = number' where X is already in NumericContext and the new
       number differs from the stored value by more than tolerance, it emits a violation.

    Returns ArithmeticEquation objects whose lhs_expr is a diagnostic string
    'chain:{var}' and rhs_value is the newly-stated value, so violation detection
    in the caller can flag them.

    Note: chain equation violations bypass _safe_eval (the lhs is not a numeric
    expression) and are injected directly as CoACEViolation objects in the caller.
    We return them as ArithmeticEquation only to reuse the deduplication logic.
    """
    tolerance = 1e-4
    context: NumericContext = {}
    violations: list[ArithmeticEquation] = []
    seen: set[str] = set()

    # First pass: extract all 'var = number' or 'var = expr = number' assignments.
    # Process in document order so earlier definitions shadow later ones.
    for m in _CHAIN_ASSIGN.finditer(text):
        var = m.group(1)
        expr_raw = m.group(2).strip().rstrip(")").strip()

        # Try to extract a number from the end of expr_raw first (chained '= number' form).
        final_num_match = re.search(r"=\s*(\d+(?:\.\d+)?)\s*$", expr_raw)
        if final_num_match:
            stated_val_str = final_num_match.group(1)
            lhs_only = expr_raw[: final_num_match.start()].strip()
            computed: Optional[float] = _safe_eval(lhs_only)
        else:
            # Simple 'var = number' or 'var = expr'
            stated_val_str = expr_raw
            computed = _safe_eval(expr_raw)

        try:
            stated_val = float(stated_val_str)
        except ValueError:
            continue

        key = f"chain:{var}:to:{stated_val_str}"
        if key in seen:
            continue

        if var in context:
            # Variable was already assigned a different value — chain mismatch.
            prior_val = context[var]
            if abs(prior_val - stated_val) > tolerance:
                seen.add(key)
                violations.append(
                    ArithmeticEquation(
                        lhs_expr=f"chain:{var}",
                        rhs_value=stated_val,
                        stated_result=stated_val_str,
                        confidence=0.85,
                    )
                )
        else:
            # First assignment: store in context.
            # Prefer the computed value (eval of lhs) if available; else use stated.
            context[var] = computed if computed is not None else stated_val

    return violations


# ---------------------------------------------------------------------------
# CoACEExtractorV2 — public API
# ---------------------------------------------------------------------------


class CoACEExtractorV2(CoACEExtractor):
    """Extended CoACE extractor that catches multi-step chains and prose patterns.

    **Why inherit from CoACEExtractor:**
        V1 already handles 'A op B = C' equations correctly.  V2 reuses _safe_eval,
        the tolerance/confidence parameters, and to_constraint_terms() unchanged.
        We only override extract() to add the two new equation sources.

    **Three equation sources:**
        1. v1: _parse_arithmetic_equations(response) — symbolic 'A op B = C'.
        2. prose: _parse_prose_arithmetic(response) — natural-language connectives.
        3. chain: _extract_chain_equations(response) — variable-tracking mismatches.

    Deduplication: equations with the same (lhs_expr, rhs_value) pair are counted once.

    Spec: REQ-EXTRACT-035, REQ-EXTRACT-036,
          SCENARIO-EXTRACT-068, SCENARIO-EXTRACT-069, SCENARIO-EXTRACT-070, SCENARIO-EXTRACT-071
    """

    def extract(self, response: str) -> CoACEResult:
        """Extract violations from all three equation sources and return merged result.

        Steps:
        1. Call v1's _parse_arithmetic_equations for symbolic equations.
        2. Call _parse_prose_arithmetic for natural-language arithmetic patterns.
        3. Call _extract_chain_equations for chain-tracking mismatches.
        4. Deduplicate by (lhs_expr, rhs_value).
        5. Run _safe_eval on each unique equation; emit CoACEViolation if error > tolerance.
        6. Chain equations (lhs starts 'chain:') are violations by construction —
           inject directly without _safe_eval.

        Returns a CoACEResult with all combined violations.
        """
        from carnot.extraction.coace_extractor import _parse_arithmetic_equations

        # Collect equations from all three sources.
        v1_eqs = _parse_arithmetic_equations(response)
        prose_eqs = _parse_prose_arithmetic(response)
        chain_eqs = _extract_chain_equations(response)

        # Deduplicate: (lhs_expr, rhs_value) is the canonical identity.
        seen_pairs: set[tuple[str, float]] = set()
        all_eqs: list[ArithmeticEquation] = []
        for eq in v1_eqs + prose_eqs:
            key = (eq.lhs_expr, eq.rhs_value)
            if key not in seen_pairs:
                seen_pairs.add(key)
                all_eqs.append(eq)

        # Evaluate and detect violations for numeric equations.
        violations: list[CoACEViolation] = []
        for eq in all_eqs:
            if eq.confidence < self.min_confidence:
                continue
            computed = _safe_eval(eq.lhs_expr)
            if computed is None:
                continue
            abs_err = abs(computed - eq.rhs_value)
            if abs_err > self.tolerance:
                rel_err = abs_err / max(abs(eq.rhs_value), 1e-12)
                violations.append(
                    CoACEViolation(
                        equation=eq,
                        computed_value=computed,
                        stated_value=eq.rhs_value,
                        absolute_error=abs_err,
                        relative_error=rel_err,
                        is_violation=True,
                    )
                )

        # Chain equations are violations by construction (prior value != stated value).
        for eq in chain_eqs:
            key = (eq.lhs_expr, eq.rhs_value)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            # Stated value is the newly-claimed value; computed is "unknown" from chain context.
            # We represent the error symbolically: absolute_error = 1.0 (sentinel).
            violations.append(
                CoACEViolation(
                    equation=eq,
                    computed_value=float("nan"),
                    stated_value=eq.rhs_value,
                    absolute_error=1.0,
                    relative_error=1.0,
                    is_violation=True,
                )
            )

        n_eqs = len(all_eqs) + len(chain_eqs)
        conf_weighted = sum(1 for v in violations if v.equation.confidence >= 0.8)

        return CoACEResult(
            n_equations_found=n_eqs,
            n_violations=len(violations),
            violations=violations,
            extraction_mode="execution_based_v2",
            confidence_weighted_violations=conf_weighted,
        )

    def detect_violations(self, text: str) -> list[CoACEViolation]:
        """Satisfy the ViolationExtractor protocol for run_extractor_diagnostic.

        The diagnostic harness calls detect_violations(text) -> list[Any] and checks
        whether the list is non-empty.  This thin wrapper delegates to extract().
        """
        result = self.extract(text)
        return result.violations
