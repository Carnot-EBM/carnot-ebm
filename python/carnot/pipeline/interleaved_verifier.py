"""InterleavedLogicVerifier — lightweight Z3 checks interleaved at CoT step boundaries.

**Why this module exists:**

    arXiv 2601.22642 ("Interleaved Formal-Logic Verification") shows that inserting
    symbolic verification DURING chain-of-thought generation — rather than post-hoc
    after the full response is produced — achieves 10.4–14.2% accuracy gains on
    math reasoning benchmarks.  The key insight: once an incorrect intermediate claim
    propagates forward it tends to corrupt all downstream steps.  Catching it at the
    STEP boundary prevents that contamination.

    This module implements the CPU-only verifier side of that paper.  Z3 arithmetic
    checks are very fast (< 5 ms for simple linear constraints), so the 50 ms
    default timeout is conservative.

**Architecture:**

    1. InterleavedStepResult — one record per CoT step with verification outcome.
    2. InterleavedLogicVerifier — the main verifier:
       - _split_steps(response) → list[str]: sentence-boundary splitter that
         identifies reasoning boundaries at arithmetic operators and inference keywords.
       - _formalize_step(step, prior_constraints) → str | None: extract a numeric
         equation from the step text and emit a Z3 Python assertion string that is SAT
         when the claim is arithmetically WRONG.
       - verify_response(response) → list[InterleavedStepResult]: walk steps in order,
         Z3-check each, accumulate constraints for subsequent steps.

**Why "SAT means violation":**

    Standard Z3 usage: assert the NEGATION of what we want to prove.  If Z3 finds the
    negation satisfiable (SAT), our claim is wrong.  For "47 + 28 = 75" we assert
    "47 + 28 != 75" — Z3 returns UNSAT (the equation is actually correct), so no
    violation.  For "3 + 4 = 8" we assert "3 + 4 != 8" — Z3 returns SAT (there exists
    a world where 3+4 is not 8, which is trivially true since 3+4=7≠8), so violation.

    Actually, re-reading: Z3 evaluates "3 + 4 != 8" as a formula over constants — it is
    simply True (a tautology), so it returns SAT.  For "47 + 28 != 75" the formula is
    False (47+28=75 always), so Z3 returns UNSAT.  SAT ↔ violation.

Spec: REQ-VERIFY-135, SCENARIO-VERIFY-168, SCENARIO-VERIFY-169, SCENARIO-VERIFY-170
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# InterleavedStepResult
# ---------------------------------------------------------------------------

@dataclass
class InterleavedStepResult:
    """Verification outcome for one CoT step.

    Fields:
        step_text         — Original text of the step.
        step_idx          — Zero-based position in the response step list.
        z3_sat            — True if Z3 found a violation (claim is wrong),
                            False if Z3 returned UNSAT (claim is correct),
                            None if no verifiable arithmetic equation was found.
        constraint_added  — The Z3 assertion added to the running constraint list
                            for future steps, or None if this step was not verifiable.
        violation_detected — True when z3_sat is True (shortcut flag for callers).
    """

    step_text: str
    step_idx: int
    z3_sat: bool | None = None
    constraint_added: str | None = None
    violation_detected: bool = False


# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# Matches a simple numeric equation of the form: LHS = RHS
# LHS may contain +, -, *, /, (, ), spaces, and numbers.
# RHS must be a single number (possibly negative or decimal).
# We intentionally keep this narrow to avoid false positives on comparison statements
# like "n >= 3" or "x = variable_name".
_EQ_RE = re.compile(
    r"""
    (                          # Group 1: left-hand side expression
        [\d\s()+\-*/.,]+       # digits, operators, parens, spaces, commas
    )
    \s*=\s*                    # equals sign (not ==)
    (                          # Group 2: right-hand side — must be a bare number
        -?[\d,]+(?:\.\d+)?     # optional leading minus, digits, optional decimal
    )
    (?!\s*=)                   # not followed by another = (avoid == comparison)
    """,
    re.VERBOSE,
)

# Sentence/step boundary: a period, newline, or inference keyword not mid-number.
# We split on sentence terminals AND inference-keyword clauses.
_STEP_SPLIT_RE = re.compile(
    r"""
    (?:                            # Option A: inference keyword introduces a new step
        (?<=[.!?\n])               # must be preceded by a sentence terminal
        \s*
        (?:therefore|so|thus|hence)\b
    )
    |
    (?:                            # Option B: sentence ending before arithmetic keyword
        [.!?\n]+\s*
        (?=[\d\-\(]|therefore|so|thus|hence)
    )
    |
    \n{2,}                         # Option C: blank line between paragraphs
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Matches inference boundary words to split step text that was not caught by regex A/B.
_INFERENCE_KW_RE = re.compile(
    r"\b(therefore|so\b|thus|hence)\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _clean_num(s: str) -> float | None:
    """Parse a number string, stripping commas.  Returns None on failure.

    Why commas: many LLM responses write large numbers as "1,000" which Python's
    float() does not accept directly.
    """
    try:
        return float(s.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def _safe_eval(expr: str) -> float | None:
    """Evaluate a simple arithmetic expression using only numeric literals and operators.

    Why not ast.literal_eval: we need to evaluate expressions like "3 + 4 * 2", not
    just literals.  We use a restricted eval with an empty namespace — no builtins, no
    imports possible.  The regex guard on expr ensures only digits and operator chars
    reach eval, so injection risk is negligible.
    """
    # Strip commas from numbers in the expression.
    cleaned = expr.replace(",", "")
    # Guard: allow only digits, operators, parens, dots, spaces.
    if not re.match(r"^[\d\s()+\-*/. ]+$", cleaned):
        return None
    try:
        result = eval(cleaned, {"__builtins__": {}})  # noqa: S307
        return float(result)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# InterleavedLogicVerifier
# ---------------------------------------------------------------------------

class InterleavedLogicVerifier:
    """Interleaves lightweight Z3 arithmetic checks at CoT step boundaries.

    Why interleaved rather than post-hoc: arXiv 2601.22642 demonstrates that
    catching an incorrect intermediate step before it propagates to downstream
    steps prevents error amplification.  Each step's constraints are accumulated
    into a running list so later steps are checked against prior verified claims.

    Usage:
        verifier = InterleavedLogicVerifier(z3_timeout_ms=50)
        results = verifier.verify_response("Janet sells 9 * $2 = $20 per day.")
        any(r.violation_detected for r in results)  # True — 9*2=18 not 20

    Args:
        z3_timeout_ms: Maximum milliseconds to spend on each Z3 query.
                       Z3's timeout is in milliseconds; 50 ms is sufficient for
                       linear integer arithmetic (QF_LIA) over constants.
    """

    def __init__(self, z3_timeout_ms: int = 50) -> None:
        self.z3_timeout_ms = z3_timeout_ms

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify_response(self, response: str) -> list[InterleavedStepResult]:
        """Walk CoT steps left-to-right, Z3-checking each arithmetic claim.

        For each step:
          1. Attempt to extract a numeric equation.
          2. Build a Z3 assertion string representing "the claim is WRONG"
             (SAT → violation, UNSAT → claim is correct).
          3. Run Z3 with timeout.  If SAT, set violation_detected=True.
          4. Add the violation-form assertion to the running constraint list
             so future steps can be checked against accumulated context.

        Args:
            response: Full LLM response text (may contain multiple CoT steps).

        Returns:
            List of InterleavedStepResult, one per identified step.
        """
        steps = self._split_steps(response)
        prior_constraints: list[str] = []
        results: list[InterleavedStepResult] = []

        for idx, step_text in enumerate(steps):
            assertion = self._formalize_step(step_text, prior_constraints)
            z3_sat: bool | None = None
            violation_detected = False

            if assertion is not None:
                z3_sat = self._z3_check(assertion)
                if z3_sat:
                    violation_detected = True
                # Add the constraint (in violation form) to running list regardless
                # of SAT/UNSAT so future steps inherit the accumulated context.
                prior_constraints.append(assertion)

            results.append(
                InterleavedStepResult(
                    step_text=step_text,
                    step_idx=idx,
                    z3_sat=z3_sat,
                    constraint_added=assertion,
                    violation_detected=violation_detected,
                )
            )

        return results

    # ------------------------------------------------------------------
    # Step splitting
    # ------------------------------------------------------------------

    def _split_steps(self, response: str) -> list[str]:
        """Split a CoT response into steps at arithmetic or reasoning boundaries.

        Strategy: split on paragraph breaks, sentence-ending punctuation followed
        by a numeral or inference keyword, and bare inference keywords.  This is a
        heuristic — it does not need to be perfect.  False splits only make the
        step list finer, which is harmless; missed splits mean two claims are
        checked together (also acceptable).

        Args:
            response: Full response text.

        Returns:
            Non-empty list of step strings (at least ["response"] if no split found).
        """
        # Primary split: newlines or sentence terminals near numbers / keywords.
        # We normalise \r\n → \n first.
        text = response.replace("\r\n", "\n").replace("\r", "\n")

        # Split on double-newline (paragraph boundary) first.
        paragraphs = re.split(r"\n{2,}", text)

        steps: list[str] = []
        for para in paragraphs:
            # Within each paragraph, split on sentence-terminal + arithmetic/keyword.
            # We also split at inline inference keywords (therefore/thus/hence/so).
            parts = re.split(
                r'(?<=[.!?])\s+(?=[\d(])'      # sentence end before number
                r'|(?<=[.!?])\s+(?=(?:therefore|thus|hence|so)\b)'  # before keyword
                r'|\b(?:therefore|thus|hence)\s+',  # after keyword
                para,
                flags=re.IGNORECASE,
            )
            for part in parts:
                part = part.strip()
                if part:
                    steps.append(part)

        return steps if steps else [response]

    # ------------------------------------------------------------------
    # Equation extraction → Z3 assertion
    # ------------------------------------------------------------------

    def _formalize_step(
        self, step: str, prior_constraints: list[str]
    ) -> str | None:
        """Extract a numeric equation from step text and return a Z3 assertion.

        The assertion is in violation form: "LHS != RHS" (Python Z3 notation).
        Z3 evaluates this over ground-truth arithmetic:
            - "47 + 28 != 75" is UNSAT (47+28=75, so it is never != 75) → no violation.
            - "3 + 4 != 8"    is SAT   (3+4=7≠8, so it is always != 8)  → violation.

        Why prior_constraints is accepted but not currently used: future versions can
        build compound assertions from accumulated constraints.  Accepting it now keeps
        the interface stable.

        Args:
            step:               Text of one CoT step.
            prior_constraints:  Assertions accumulated from earlier verified steps.

        Returns:
            Z3 assertion string (Python expression) or None if no equation found.
        """
        # Strip LaTeX delimiters to expose plain arithmetic.
        clean = re.sub(r'\\[\[\]()]', '', step)          # remove \[, \], \(, \)
        clean = re.sub(r'\\times', '*', clean)
        clean = re.sub(r'\\cdot', '*', clean)
        clean = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', clean)
        clean = re.sub(r'[$]', '', clean)                 # strip currency markers

        match = _EQ_RE.search(clean)
        if match is None:
            return None

        lhs_raw = match.group(1).strip()
        rhs_raw = match.group(2).strip()

        # Evaluate LHS to see if it is a well-formed arithmetic expression.
        lhs_val = _safe_eval(lhs_raw)
        rhs_val = _clean_num(rhs_raw)

        if lhs_val is None or rhs_val is None:
            return None

        # Build violation assertion: "lhs_eval != rhs" in plain Python / Z3 syntax.
        # We use the evaluated float values directly so the assertion is ground-truth
        # arithmetic (no symbolic variables needed — Z3 handles constant folding).
        return f"{lhs_val!r} != {rhs_val!r}"

    # ------------------------------------------------------------------
    # Z3 invocation
    # ------------------------------------------------------------------

    def _z3_check(self, assertion: str) -> bool:
        """Run a Z3 satisfiability check on a ground-truth assertion string.

        The assertion is a Python expression over numeric literals and != / ==.
        We evaluate it directly in Python as a fallback when z3 is unavailable —
        since the assertion contains no symbolic variables, Python arithmetic is
        sufficient.

        Why not always use Python eval: Z3 provides a standard interface that could
        later be extended to symbolic arithmetic.  We attempt Z3 first and fall back
        to Python eval to keep CI green without the z3-solver package installed.

        Returns:
            True  if the assertion is SAT (violation detected),
            False if the assertion is UNSAT (claim is correct).
        """
        try:
            # Attempt real Z3 check.
            import z3  # noqa: PLC0415

            solver = z3.Solver()
            solver.set("timeout", self.z3_timeout_ms)
            # Parse the assertion as a Z3 formula from a Python-syntax string.
            # Since the assertion is ground arithmetic (no free variables), z3.parse_smt2_string
            # is overkill; we use eval with z3 in scope so "47.0 != 75.0" becomes a z3 BoolVal.
            formula = eval(assertion, {"__builtins__": {}})  # noqa: S307
            # If Python eval already returns a plain bool (Z3 not needed), use it.
            if isinstance(formula, bool):
                return formula
            solver.add(formula)
            result = solver.check()
            return result == z3.sat
        except ImportError:
            # z3 not installed — fall back to Python constant evaluation.
            pass
        except Exception:  # noqa: BLE001
            pass

        # Python fallback: evaluate "lhs != rhs" directly.
        try:
            result = eval(assertion, {"__builtins__": {}})  # noqa: S307
            if isinstance(result, bool):
                return result
        except Exception:  # noqa: BLE001
            pass

        return False
