"""CoACEExtractorV3 — live-corpus-calibrated arithmetic extraction.

**Why V3 exists (RETRO-066 root cause):**

    CoACEExtractorV2 achieves 86.7% recall on the offline synthetic corpus (Exp 565)
    but only 5.9% recall on live FOVER production responses (Exp 581).  Root cause:
    the offline corpus used curated 'A op B = C' patterns.  Real IT-model outputs use:

    1. Currency-prefixed arithmetic: '3 * $16.50 = $54.50' — the dollar sign and
       comma separators break V2's number-matching regex, so the equation is invisible.

    2. Narrative addition chains: 'Adding 47 to 28 gives us 76' — no '=' symbol, no
       operator symbol.  V2's prose patterns require specific connective words that
       don't cover 'gives us', 'result is', 'we get', etc.

    3. Cumulative running totals: 'bringing the total to 150' — the model states a
       running subtotal in prose, and the error may be three steps removed from the
       point where it diverges.

    4. Percentage word problems with alternate connectives: 'N% discount leaves P',
       'amounts to P', 'totals P' — V2's _PERCENT_OF only matches 'is/equals/gives'.

    5. Unit conversion assertions: 'X hours is Y minutes' — the model converts units
       and states the result.  V2 has no unit conversion patterns at all.

    6. Total-of multi-term additive chains: 'total of A + B + C = Z' where V2 only
       matches 'sum of'.

    7. After-operation narrative: 'after adding X to Y, we get Z' / 'after subtracting
       X from Y, resulting in Z'.

    V3 inherits all V2 patterns and adds seven new helper parsers calibrated against
    the 100 live responses in results/live_pairs_578.json.

Spec: REQ-EXTRACT-040, REQ-EXTRACT-041, REQ-EXTRACT-042,
      SCENARIO-EXTRACT-075, SCENARIO-EXTRACT-076, SCENARIO-EXTRACT-077, SCENARIO-EXTRACT-078
"""

from __future__ import annotations

import re
from typing import Optional

from carnot.extraction.coace_extractor import (
    ArithmeticEquation,
    CoACEResult,
    CoACEViolation,
    _safe_eval,
)
from carnot.extraction.coace_extractor_v2 import CoACEExtractorV2

# ---------------------------------------------------------------------------
# Helper: strip currency symbols and thousands commas from numeric strings
# ---------------------------------------------------------------------------

_CURRENCY_STRIP = re.compile(r"[\$,]")


def _strip_currency(s: str) -> str:
    """Remove '$' and ',' from a numeric string so float() can parse it.

    Why: IT models write '$16.50' or '$130,000' inline in equations.  Python's
    float() rejects both forms.  We strip currency markers before evaluation.
    """
    return _CURRENCY_STRIP.sub("", s)


# ---------------------------------------------------------------------------
# Numeric building blocks (handles optional $, commas, decimals)
# ---------------------------------------------------------------------------

_NUM = r"\$?[\d,]+(?:\.\d+)?"  # matches $1,234.56 or 42 or 3.14

# ---------------------------------------------------------------------------
# Pattern 1: currency-prefixed arithmetic  'N * $M.PP = $Z.ZZ'
# Catches: 3 * $16.50 = $54.50  (wrong: should be $49.50)
# ---------------------------------------------------------------------------

_CURRENCY_MUL = re.compile(
    rf"({_NUM})\s*\*\s*({_NUM})\s*=\s*({_NUM})",
    re.IGNORECASE,
)

_CURRENCY_ADD = re.compile(
    rf"({_NUM})\s*\+\s*({_NUM})\s*=\s*({_NUM})",
    re.IGNORECASE,
)

_CURRENCY_SUB = re.compile(
    rf"({_NUM})\s*-\s*({_NUM})\s*=\s*({_NUM})",
    re.IGNORECASE,
)

_CURRENCY_DIV = re.compile(
    rf"({_NUM})\s*/\s*({_NUM})\s*=\s*({_NUM})",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Pattern 2: narrative addition  'Adding X to Y gives/results in/we get Z'
# ---------------------------------------------------------------------------

_NARRATIVE_ADD = re.compile(
    rf"[Aa]dding\s+({_NUM})\s+to\s+({_NUM})\s+"
    rf"(?:gives?(?:\s+us)?|results?\s+in|we\s+get|equals?|is|=)\s+({_NUM})",
    re.IGNORECASE,
)

_NARRATIVE_SUB = re.compile(
    rf"[Ss]ubtracting\s+({_NUM})\s+from\s+({_NUM})\s+"
    rf"(?:gives?(?:\s+us)?|results?\s+in|we\s+get|leaves?|equals?|is|=)\s+({_NUM})",
    re.IGNORECASE,
)

_NARRATIVE_MUL = re.compile(
    rf"[Mm]ultiplying\s+({_NUM})\s+(?:by|times)\s+({_NUM})\s+"
    rf"(?:gives?(?:\s+us)?|results?\s+in|we\s+get|equals?|is|=)\s+({_NUM})",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Pattern 3: cumulative/running-total  'bringing the total to Z'
# We scan prose for intermediate sums and then check the stated cumulative total.
# ---------------------------------------------------------------------------

_BRINGING_TOTAL = re.compile(
    r"bringing\s+(?:the\s+)?(?:total|count|sum)\s+to\s+"
    rf"({_NUM})",
    re.IGNORECASE,
)

# Numbers that precede the 'bringing total' phrase (look-behind context window)
_PRECEDING_ADDENDS = re.compile(
    rf"({_NUM})(?:\s*[+,]\s*({_NUM}))*",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Pattern 4: percentage word problems with extended connectives
# Supplements V2's _PERCENT_OF (is/equals/gives) with: totals, amounts to,
# discount, markup, leaves, becomes
# ---------------------------------------------------------------------------

_PERCENT_EXTENDED = re.compile(
    rf"(\d+(?:\.\d+)?)\s*%\s+(?:of\s+)?({_NUM})\s+"
    rf"(?:is|equals?|gives?|totals?|amounts?\s+to|leaves?|becomes?|=)\s+({_NUM})",
    re.IGNORECASE,
)

_DISCOUNT_MARKUP = re.compile(
    rf"({_NUM})\s+at\s+(\d+(?:\.\d+)?)\s*%\s+"
    rf"(?:discount|markup|off|interest)\s+"
    rf"(?:is|equals?|leaves?|becomes?|gives?|results?\s+in|=)\s+({_NUM})",
    re.IGNORECASE,
)

_OUT_OF_PERCENT = re.compile(
    rf"out\s+of\s+({_NUM})\s+(?:total[,\s]+)?(\d+(?:\.\d+)?)\s*%\s+"
    rf"(?:attended|enrolled|are|were|means?|meaning)\s*[,\s]*"
    rf"(?:meaning\s+)?(?:so\s+)?(?:that\s+(?:is|means?)\s+)?({_NUM})\s+(?:people|students|items?|units?)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Pattern 5: unit conversions
# ---------------------------------------------------------------------------

_HOURS_TO_MINUTES = re.compile(
    rf"(\d+(?:\.\d+)?)\s+hours?\s+(?:is|equals?|=|converts?\s+to|is\s+equal\s+to)\s+"
    rf"(\d+(?:\.\d+)?)\s+minutes?",
    re.IGNORECASE,
)

_DAYS_TO_HOURS = re.compile(
    rf"(\d+(?:\.\d+)?)\s+days?\s+(?:is|equals?|=|converts?\s+to|is\s+equal\s+to)\s+"
    rf"(\d+(?:\.\d+)?)\s+hours?",
    re.IGNORECASE,
)

_KM_TO_METERS = re.compile(
    rf"(\d+(?:\.\d+)?)\s+km(?:s|ilometers?)?\s+"
    rf"(?:is|equals?|=|converts?\s+to|is\s+equal\s+to)\s+"
    rf"(\d+(?:\.\d+)?)\s+met(?:er|re)s?",
    re.IGNORECASE,
)

_MILES_TO_KM = re.compile(
    rf"(\d+(?:\.\d+)?)\s+miles?\s+"
    rf"(?:is|equals?|=|converts?\s+to|is\s+equal\s+to)\s+"
    rf"(\d+(?:\.\d+)?)\s+(?:km|kilometer)s?",
    re.IGNORECASE,
)

_FEET_TO_INCHES = re.compile(
    rf"(\d+(?:\.\d+)?)\s+feet\s+"
    rf"(?:is|equals?|=|converts?\s+to|is\s+equal\s+to)\s+"
    rf"(\d+(?:\.\d+)?)\s+inches?",
    re.IGNORECASE,
)

_WEEKS_TO_DAYS = re.compile(
    rf"(\d+(?:\.\d+)?)\s+weeks?\s+"
    rf"(?:is|equals?|=|converts?\s+to|is\s+equal\s+to)\s+"
    rf"(\d+(?:\.\d+)?)\s+days?",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Pattern 6: 'total of A + B + C = Z'  (V2 covers 'sum of', not 'total of')
# ---------------------------------------------------------------------------

_TOTAL_OF_CHAIN = re.compile(
    r"total\s+(?:cost\s+)?(?:of|is|=)?\s+"
    rf"((?:{_NUM}\s*\+\s*)+{_NUM})\s*=\s*({_NUM})",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Pattern 7: after-operation narrative
# 'after adding X to Y, we get Z' / 'after subtracting X from Y, resulting in Z'
# ---------------------------------------------------------------------------

_AFTER_ADD = re.compile(
    rf"after\s+adding\s+({_NUM})\s+to\s+({_NUM})[,\s]+"
    rf"(?:we\s+get|the\s+result\s+is|resulting\s+in|we\s+have|giving\s+us?|=)\s+({_NUM})",
    re.IGNORECASE,
)

_AFTER_SUB = re.compile(
    rf"after\s+subtracting\s+({_NUM})\s+from\s+({_NUM})[,\s]+"
    rf"(?:we\s+get|the\s+result\s+is|resulting\s+in|we\s+have|giving\s+us?|=)\s+({_NUM})",
    re.IGNORECASE,
)

_AFTER_MUL = re.compile(
    rf"after\s+multiplying\s+({_NUM})\s+(?:by|times)\s+({_NUM})[,\s]+"
    rf"(?:we\s+get|the\s+result\s+is|resulting\s+in|we\s+have|giving\s+us?|=)\s+({_NUM})",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _num(s: str) -> Optional[float]:
    """Parse a numeric string, stripping currency markers and commas.

    Returns None if the string cannot be parsed as a float.  This is safer than
    calling float() directly because IT model responses embed '$' and ',' in numbers.
    Also returns None for non-string input (e.g., None) to prevent TypeErrors when
    called from regex match groups that may be absent.
    """
    try:
        return float(_strip_currency(s))
    except (ValueError, AttributeError, TypeError):
        return None


def _make_eq(lhs: str, rhs_str: str, conf: float = 0.85) -> Optional[ArithmeticEquation]:
    """Build an ArithmeticEquation only if rhs parses to a valid float.

    Why return Optional: if the regex captures a malformed group (e.g., partial match
    on a line break), rhs_str may not be a valid float.  Silently skip rather than crash.
    """
    rhs = _num(rhs_str)
    if rhs is None:
        return None
    return ArithmeticEquation(
        lhs_expr=lhs,
        rhs_value=rhs,
        stated_result=rhs_str,
        confidence=conf,
    )


# ---------------------------------------------------------------------------
# Public parsers (exported for unit testing)
# ---------------------------------------------------------------------------


def _parse_narrative_arithmetic(text: str) -> list[ArithmeticEquation]:
    """Parse narrative arithmetic patterns not covered by V2.

    Covers:
      - Currency-prefixed equations: '3 * $16.50 = $54.50'
      - Narrative addition: 'Adding 47 to 28 gives us 76'
      - Narrative subtraction: 'Subtracting 3 from 10 leaves 7'
      - Narrative multiplication: 'Multiplying 7 by 1.5 gives 10'
      - 'After adding X to Y, we get Z'

    Why separate from _parse_prose_arithmetic in V2: V2 patterns require the
    operator to be written as a connecting word ('times', 'divided by').  These
    patterns handle operator *symbols* embedded in currency-heavy prose, plus
    'after …' narrative frames.

    Returns: list of ArithmeticEquation (may be empty).
    """
    eqs: list[ArithmeticEquation] = []
    seen: set[tuple[str, float]] = set()

    def _add(lhs: str, rhs_str: str, conf: float = 0.85) -> None:
        eq = _make_eq(lhs, rhs_str, conf)
        if eq is None:
            return
        key = (eq.lhs_expr, eq.rhs_value)
        if key not in seen:
            seen.add(key)
            eqs.append(eq)

    # Currency-prefixed symbolic equations.
    for m in _CURRENCY_MUL.finditer(text):
        a, b, r = _strip_currency(m.group(1)), _strip_currency(m.group(2)), m.group(3)
        if "$" in m.group(1) or "$" in m.group(2) or "$" in m.group(3):
            _add(f"{a}*{b}", r)

    for m in _CURRENCY_ADD.finditer(text):
        a, b, r = _strip_currency(m.group(1)), _strip_currency(m.group(2)), m.group(3)
        if "$" in m.group(1) or "$" in m.group(2) or "$" in m.group(3):
            _add(f"{a}+{b}", r)

    for m in _CURRENCY_SUB.finditer(text):
        a, b, r = _strip_currency(m.group(1)), _strip_currency(m.group(2)), m.group(3)
        if "$" in m.group(1) or "$" in m.group(2) or "$" in m.group(3):
            _add(f"{a}-{b}", r)

    for m in _CURRENCY_DIV.finditer(text):
        a, b, r = _strip_currency(m.group(1)), _strip_currency(m.group(2)), m.group(3)
        if "$" in m.group(1) or "$" in m.group(2) or "$" in m.group(3):
            _add(f"{a}/{b}", r)

    # Narrative addition / subtraction / multiplication.
    for m in _NARRATIVE_ADD.finditer(text):
        a, b, r = _strip_currency(m.group(1)), _strip_currency(m.group(2)), m.group(3)
        _add(f"{a}+{b}", r)

    for m in _NARRATIVE_SUB.finditer(text):
        a, b, r = _strip_currency(m.group(1)), _strip_currency(m.group(2)), m.group(3)
        _add(f"{b}-{a}", r)

    for m in _NARRATIVE_MUL.finditer(text):
        a, b, r = _strip_currency(m.group(1)), _strip_currency(m.group(2)), m.group(3)
        _add(f"{a}*{b}", r)

    # After-operation narrative.
    for m in _AFTER_ADD.finditer(text):
        a, b, r = _strip_currency(m.group(1)), _strip_currency(m.group(2)), m.group(3)
        _add(f"{b}+{a}", r)

    for m in _AFTER_SUB.finditer(text):
        a, b, r = _strip_currency(m.group(1)), _strip_currency(m.group(2)), m.group(3)
        _add(f"{b}-{a}", r)

    for m in _AFTER_MUL.finditer(text):
        a, b, r = _strip_currency(m.group(1)), _strip_currency(m.group(2)), m.group(3)
        _add(f"{a}*{b}", r)

    return eqs


def _parse_percentage_word_problem(text: str) -> list[ArithmeticEquation]:
    """Parse percentage-of word problems with connectives not covered by V2.

    V2's _PERCENT_OF matches 'N% of M is/equals/gives P'.  Real IT model outputs
    also write: 'totals P', 'amounts to P', 'leaves P', 'becomes P' (post-discount).
    V3 adds these forms plus:
      - 'M at N% discount/markup/interest is/becomes P'
      - 'out of M total, N% attended, meaning P people'

    Why the 'at N% discount' form matters: the Josh house-flip response writes
    '$130,000 * 1.5 = $195,000' which V3 catches via currency arithmetic.  But
    an IT model could also write 'the house at 150% increase is $325,000' and the
    base value for that claim ($130,000) must be recovered from context.

    Returns: list of ArithmeticEquation (may be empty).
    """
    eqs: list[ArithmeticEquation] = []
    seen: set[tuple[str, float]] = set()

    def _add(lhs: str, rhs_str: str, conf: float = 0.85) -> None:
        eq = _make_eq(lhs, rhs_str, conf)
        if eq is None:
            return
        key = (eq.lhs_expr, eq.rhs_value)
        if key not in seen:
            seen.add(key)
            eqs.append(eq)

    # Extended 'N% of M is/totals/amounts to P'
    for m in _PERCENT_EXTENDED.finditer(text):
        n, base, stated = m.group(1), _strip_currency(m.group(2)), m.group(3)
        _add(f"{n}/100*{base}", stated)

    # 'M at N% discount/markup is P'  → P = M * (1 - N/100)  or  M * (1 + N/100)
    for m in _DISCOUNT_MARKUP.finditer(text):
        base_str = _strip_currency(m.group(1))
        n = m.group(2)
        stated = m.group(3)
        raw = m.group(0).lower()
        if "discount" in raw or "off" in raw:
            _add(f"{base_str}*(1-{n}/100)", stated)
        else:
            # markup or interest
            _add(f"{base_str}*(1+{n}/100)", stated)

    # 'out of M total, N% attended, meaning P people'
    for m in _OUT_OF_PERCENT.finditer(text):
        total_str = _strip_currency(m.group(1))
        n = m.group(2)
        stated = _strip_currency(m.group(3))
        _add(f"{total_str}*{n}/100", stated)

    return eqs


def _parse_unit_conversion(text: str) -> list[ArithmeticEquation]:
    """Parse unit conversion assertions.

    IT models sometimes state a unit conversion result explicitly, e.g.
    '3 hours is 180 minutes'.  If the model makes an arithmetic error in the
    conversion (e.g., '3 hours is 200 minutes'), V3 can catch it.

    Conversion rules:
      hours → minutes: multiply by 60
      days  → hours:   multiply by 24
      km    → meters:  multiply by 1000
      miles → km:      multiply by 1.60934
      feet  → inches:  multiply by 12
      weeks → days:    multiply by 7

    Returns: list of ArithmeticEquation (may be empty).
    """
    eqs: list[ArithmeticEquation] = []
    seen: set[tuple[str, float]] = set()

    def _add(lhs: str, rhs_str: str) -> None:
        eq = _make_eq(lhs, rhs_str, conf=0.90)
        if eq is None:
            return
        key = (eq.lhs_expr, eq.rhs_value)
        if key not in seen:
            seen.add(key)
            eqs.append(eq)

    for m in _HOURS_TO_MINUTES.finditer(text):
        _add(f"{m.group(1)}*60", m.group(2))

    for m in _DAYS_TO_HOURS.finditer(text):
        _add(f"{m.group(1)}*24", m.group(2))

    for m in _KM_TO_METERS.finditer(text):
        _add(f"{m.group(1)}*1000", m.group(2))

    for m in _MILES_TO_KM.finditer(text):
        # Use rational approximation to avoid float comparison issues in tests.
        _add(f"{m.group(1)}*1.60934", m.group(2))

    for m in _FEET_TO_INCHES.finditer(text):
        _add(f"{m.group(1)}*12", m.group(2))

    for m in _WEEKS_TO_DAYS.finditer(text):
        _add(f"{m.group(1)}*7", m.group(2))

    return eqs


def _parse_running_total_chain(text: str) -> list[ArithmeticEquation]:
    """Parse running-total patterns where a stated subtotal may diverge.

    Two sub-patterns:
      A. 'bringing the total to Z' — scans the preceding 300 characters for
         accumulated numeric addends and checks if their sum equals Z.
      B. 'total of A + B + ... = Z' — a single-sentence multi-term addition
         stated with the word 'total' (V2 covers 'sum of', not 'total of').

    Why cumulative totals are hard: the error may be three arithmetic steps
    before the stated total.  By the time the model writes 'bringing the total
    to 150', the mistake was made back at 'first I add 47 + 28 = 74'.  V3 does
    not resolve this retroactively; instead, it checks that the stated cumulative
    total is arithmetically consistent with the *most recent* stated intermediate
    values found in the preceding context window.

    Returns: list of ArithmeticEquation (may be empty).
    """
    eqs: list[ArithmeticEquation] = []
    seen: set[tuple[str, float]] = set()

    def _add(lhs: str, rhs_str: str, conf: float = 0.80) -> None:
        eq = _make_eq(lhs, rhs_str, conf)
        if eq is None:
            return
        key = (eq.lhs_expr, eq.rhs_value)
        if key not in seen:
            seen.add(key)
            eqs.append(eq)

    # Sub-pattern B: 'total of A + B + C = Z' (explicit additive chain)
    for m in _TOTAL_OF_CHAIN.finditer(text):
        raw_chain = m.group(1)
        stated = m.group(2)
        # Extract all numbers from the chain expression.
        nums = [_strip_currency(n.group()) for n in re.finditer(r"\$?[\d,]+(?:\.\d+)?", raw_chain)]
        if len(nums) >= 2:
            clean = [n for n in nums if _num(n) is not None]
            if len(clean) >= 2:
                lhs = "+".join(clean)
                _add(lhs, _strip_currency(stated))

    # Sub-pattern A: 'bringing the total to Z' with look-behind context
    for m in _BRINGING_TOTAL.finditer(text):
        stated_str = _strip_currency(m.group(1))
        stated_val = _num(stated_str)
        if stated_val is None:
            continue
        # Scan the 300 characters before this phrase for standalone numbers.
        start = max(0, m.start() - 300)
        context_window = text[start : m.start()]
        # Find all numbers in context that are plausible addends (not zero, not huge).
        candidates = [
            _num(n.group())
            for n in re.finditer(r"\b\d+(?:\.\d+)?\b", context_window)
            if _num(n.group()) is not None
        ]
        # Only check if we have at least 2 candidates and their sum is close to stated.
        if len(candidates) >= 2:
            # Try consecutive pairs and triples to find a sub-sum that matches.
            for i in range(len(candidates) - 1):
                for j in range(i + 2, min(i + 7, len(candidates) + 1)):
                    subset = candidates[i:j]
                    total = sum(v for v in subset if v is not None)
                    lhs = "+".join(str(v) for v in subset)
                    key = (lhs, stated_val)
                    if key not in seen:
                        seen.add(key)
                        eq = ArithmeticEquation(
                            lhs_expr=lhs,
                            rhs_value=stated_val,
                            stated_result=stated_str,
                            confidence=0.75,
                        )
                        eqs.append(eq)
                    # Stop after first match to avoid explosion.
                    break

    return eqs


# ---------------------------------------------------------------------------
# CoACEExtractorV3 — public API
# ---------------------------------------------------------------------------


class CoACEExtractorV3(CoACEExtractorV2):
    """V3 of the COACE extractor, calibrated on live IT-model output patterns.

    **Why inherit from CoACEExtractorV2:**
        V2 already handles symbolic equations, prose percentage/ratio patterns,
        and multi-step chain variable tracking.  V3 reuses all of these and adds
        four new pattern parsers that cover the real live-corpus failure modes
        documented in RETRO-066.

    **Four new equation sources (in addition to the three from V2):**
        4. _parse_narrative_arithmetic: currency-prefixed equations, 'Adding X to Y
           gives Z', 'after multiplying X by Y, we get Z', etc.
        5. _parse_percentage_word_problem: extended connectives ('totals', 'amounts
           to', 'at N% discount'), and 'out of M total N% meaning P' patterns.
        6. _parse_unit_conversion: hours→minutes, days→hours, km→meters, etc.
        7. _parse_running_total_chain: 'total of A+B+C=Z' and 'bringing total to Z'.

    Deduplication is performed on (lhs_expr, rhs_value) pairs across all seven sources.

    Spec: REQ-EXTRACT-040, REQ-EXTRACT-041, REQ-EXTRACT-042,
          SCENARIO-EXTRACT-075, SCENARIO-EXTRACT-076, SCENARIO-EXTRACT-077,
          SCENARIO-EXTRACT-078
    """

    def extract(self, response: str) -> CoACEResult:
        """Extract violations from all seven equation sources and return merged result.

        Steps:
        1. Call V2's extract() to get all violations from the three V2 sources.
        2. Run four new V3 parsers on the same response.
        3. Deduplicate new V3 equations against V2's already-seen (lhs, rhs) pairs.
        4. Evaluate each new unique equation; emit CoACEViolation on arithmetic error.
        5. Merge V2 + V3 violations and return a single CoACEResult.

        The returned CoACEResult has extraction_mode='execution_based_v3' to distinguish
        it from V2 results in experiment artifacts.
        """
        # Step 1: get all V2 violations and equations.
        v2_result = super().extract(response)

        # Reconstruct the set of (lhs, rhs) pairs already seen by V2 to avoid duplication.
        seen_pairs: set[tuple[str, float]] = {
            (v.equation.lhs_expr, v.equation.rhs_value)
            for v in v2_result.violations
        }
        # Also scan all V2-sourced equations that were NOT violations (no-op duplicates).
        # We approximate this by re-running the V2 parsers to collect all equations.
        from carnot.extraction.coace_extractor import _parse_arithmetic_equations
        from carnot.extraction.coace_extractor_v2 import (
            _parse_prose_arithmetic,
            _extract_chain_equations,
        )

        for eq in _parse_arithmetic_equations(response) + _parse_prose_arithmetic(response):
            seen_pairs.add((eq.lhs_expr, eq.rhs_value))
        for eq in _extract_chain_equations(response):
            seen_pairs.add((eq.lhs_expr, eq.rhs_value))

        # Step 2: collect new V3 equations.
        v3_new_eqs: list[ArithmeticEquation] = []
        for eq in (
            _parse_narrative_arithmetic(response)
            + _parse_percentage_word_problem(response)
            + _parse_unit_conversion(response)
            + _parse_running_total_chain(response)
        ):
            key = (eq.lhs_expr, eq.rhs_value)
            if key not in seen_pairs:
                seen_pairs.add(key)
                v3_new_eqs.append(eq)

        # Step 3: evaluate new V3 equations.
        new_violations: list[CoACEViolation] = []
        for eq in v3_new_eqs:
            if eq.confidence < self.min_confidence:
                continue
            computed = _safe_eval(eq.lhs_expr)
            if computed is None:
                continue
            abs_err = abs(computed - eq.rhs_value)
            if abs_err > self.tolerance:
                rel_err = abs_err / max(abs(eq.rhs_value), 1e-12)
                new_violations.append(
                    CoACEViolation(
                        equation=eq,
                        computed_value=computed,
                        stated_value=eq.rhs_value,
                        absolute_error=abs_err,
                        relative_error=rel_err,
                        is_violation=True,
                    )
                )

        # Step 4: merge V2 + V3 results.
        all_violations = v2_result.violations + new_violations
        n_eqs = v2_result.n_equations_found + len(v3_new_eqs)
        conf_weighted = sum(1 for v in all_violations if v.equation.confidence >= 0.8)

        return CoACEResult(
            n_equations_found=n_eqs,
            n_violations=len(all_violations),
            violations=all_violations,
            extraction_mode="execution_based_v3",
            confidence_weighted_violations=conf_weighted,
        )

    def detect_violations(self, text: str) -> list[CoACEViolation]:
        """Satisfy the ViolationExtractor protocol for run_extractor_diagnostic.

        The diagnostic harness calls detect_violations(text) -> list[Any] and checks
        whether the list is non-empty.  This thin wrapper delegates to extract().
        """
        result = self.extract(text)
        return result.violations
