"""CoACE Extractor — execution-based arithmetic constraint extraction from prose CoT.

**Why execution-based beats regex and Z3 (root cause from Exp 554):**

    VeriCoT checks logical CONSISTENCY via Z3 satisfiability — it can prove that
    a set of First-Order Logic premises are mutually inconsistent.  But Z3 has no
    way to know that '47 + 28 = 76' is arithmetically wrong unless the extractor
    also supplies '47 + 28 = 75'.  Z3 sees no contradiction in a single incorrect
    equation.

    VPRM checks FORMAT — it looks for natural-language patterns like 'A plus B gives C'.
    IT models write '47 + 28 = 76' (equation style, not prose style).  Zero patterns
    match, so zero violations are detected.

    CoACE (Caco, arXiv 2510.04081) cuts through both failure modes: extract the raw
    LHS expression (e.g., '47 + 28'), execute it as Python (eval('47+28') = 75),
    and compare to the stated RHS (76).  75 != 76 → violation.  No pattern library
    needed, no FOL extraction step, no Z3.

**Safety:**

    Python's eval() is powerful — it can run arbitrary code.  To prevent injection
    attacks, every expression goes through _safe_eval() which parses the expression
    to an AST and whitelist-checks every node before eval() is ever called.  Only
    numeric literals and the four arithmetic operators (+, -, *, /) are permitted.
    Any expression containing function calls, attribute access, or variable names
    returns None (blocked).

Spec: REQ-EXTRACT-033, REQ-EXTRACT-034,
      SCENARIO-EXTRACT-061, SCENARIO-EXTRACT-062, SCENARIO-EXTRACT-063, SCENARIO-EXTRACT-064
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ArithmeticEquation:
    """One arithmetic equation parsed from a prose CoT response.

    lhs_expr: the left-hand side as a Python-evaluable string (e.g., '47 + 28').
    rhs_value: the stated result as a float (e.g., 76.0).
    stated_result: the raw string of the stated result (e.g., '76').
    confidence: how certain we are this is a genuine equation claim.
                1.0 for '=', 0.8 for 'equals/gives/is', 0.5 for 'approximately'.
    """

    lhs_expr: str
    rhs_value: float
    stated_result: str
    confidence: float


@dataclass
class CoACEViolation:
    """One detected arithmetic violation: the LHS evaluates to a different value than stated.

    equation: the parsed equation that triggered this violation.
    computed_value: what eval(lhs_expr) actually returns.
    stated_value: the RHS value the model claimed.
    absolute_error: abs(computed_value - stated_value).
    relative_error: absolute_error / max(abs(stated_value), 1e-12) — avoids division by zero.
    is_violation: always True when this object exists; field kept for downstream compatibility.
    """

    equation: ArithmeticEquation
    computed_value: float
    stated_value: float
    absolute_error: float
    relative_error: float
    is_violation: bool = True


@dataclass
class CoACEResult:
    """Aggregated result of running CoACEExtractor on one response.

    n_equations_found: how many arithmetic equations were parsed from the text.
    n_violations: how many equations had computed != stated result.
    violations: list of CoACEViolation objects (one per wrong equation).
    extraction_mode: always 'execution_based' — records the method for provenance.
    confidence_weighted_violations: count of violations whose equation.confidence >= 0.8.
    """

    n_equations_found: int
    n_violations: int
    violations: list[CoACEViolation] = field(default_factory=list)
    extraction_mode: str = "execution_based"
    confidence_weighted_violations: int = 0


# ---------------------------------------------------------------------------
# _safe_eval — sandboxed arithmetic evaluator
# ---------------------------------------------------------------------------

def _build_allowed_nodes() -> tuple:
    """Build the whitelist of allowed AST node types at import time.

    ast.Num was deprecated in Python 3.8 and removed in Python 3.14.
    We check dynamically so the module works on Python 3.11-3.14+.
    """
    nodes = [
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
    ]
    # ast.Num existed through Python 3.13, removed in 3.14
    if hasattr(ast, "Num"):
        nodes.append(ast.Num)  # type: ignore[attr-defined]
    return tuple(nodes)


_ALLOWED_NODES = _build_allowed_nodes()


def _safe_eval(expr: str) -> Optional[float]:
    """Evaluate an arithmetic expression string, returning None if unsafe or invalid.

    Why this exists: Python's eval() runs arbitrary code.  We must guarantee that
    a malicious input like '__import__("os").system("rm -rf /")' cannot execute.
    This function parses the expression to an AST and walks every node against a
    strict whitelist before eval() is called.  If any non-arithmetic node is found,
    the function returns None without evaluating.

    Only allows: numeric literals, +, -, *, /, //, %, ** and unary +/-.
    Blocks: function calls, attribute access, names, subscripts, comprehensions.

    Returns the float result on success, None on any failure.
    """
    if not expr or not expr.strip():
        return None
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return None
        # ast.Constant must hold a numeric type, not a string or bool
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            return None

    try:
        result = eval(compile(tree, "<coace>", "eval"), {}, {})  # noqa: S307
        return float(result)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# _parse_arithmetic_equations — extract equations from prose text
# ---------------------------------------------------------------------------

# Patterns for the equality connective (in order of confidence).
# Each tuple: (regex_pattern, confidence_score)
_EQ_PATTERNS = [
    # Explicit '=' sign: '47 + 28 = 76'
    (
        r"([\d]+(?:\.\d+)?\s*(?:[+\-*/]\s*[\d]+(?:\.\d+)?)+)\s*=\s*([\d]+(?:\.\d+)?)",
        1.0,
    ),
    # 'equals', 'gives', 'is', 'results in': 'we add 47 and 28 to get 76' — partial
    # This simpler pattern catches 'number OP number equals number'
    (
        r"([\d]+(?:\.\d+)?\s*(?:[+\-*/]\s*[\d]+(?:\.\d+)?)+)\s+"
        r"(?:equals?|gives?|is|results?\s+in|to\s+get)\s+"
        r"([\d]+(?:\.\d+)?)",
        0.8,
    ),
    # 'approximately': 'we get approximately 76'
    (
        r"([\d]+(?:\.\d+)?\s*(?:[+\-*/]\s*[\d]+(?:\.\d+)?)+)\s+"
        r"(?:approximately|roughly|about)\s+"
        r"([\d]+(?:\.\d+)?)",
        0.5,
    ),
]

# Also match prose like 'add 47 and 28' then find a conclusion number.
# Simpler pattern: capture 'N op N [op N]* = M' with or without spaces.
_COMPACT_EQ = re.compile(
    r"(\d+(?:\.\d+)?)\s*([+\-*/])\s*(\d+(?:\.\d+)?)\s*=\s*(\d+(?:\.\d+)?)"
)


def _parse_arithmetic_equations(text: str) -> list[ArithmeticEquation]:
    """Extract arithmetic equations from free-form text.

    Searches for patterns of the form 'LHS = RHS' where LHS is an arithmetic
    expression (e.g., '47 + 28') and RHS is a single number.  Also recognises
    natural-language connectives like 'equals', 'gives', 'is'.

    Each match is returned as an ArithmeticEquation with:
    - lhs_expr: Python-evaluable string for the left side.
    - rhs_value: float of the stated result.
    - confidence: 1.0 for '=', 0.8 for prose connectives, 0.5 for approximations.

    Why we use multiple patterns: IT model outputs vary widely.  A model might
    write '47 + 28 = 76', '47 + 28 equals 76', or just '47 + 28 gives 76'.
    We want to catch all forms without a neural extraction step.
    """
    equations: list[ArithmeticEquation] = []
    seen_lhs: set[str] = set()

    for pattern_str, confidence in _EQ_PATTERNS:
        for m in re.finditer(pattern_str, text, re.IGNORECASE):
            lhs_raw = m.group(1).strip()
            rhs_raw = m.group(2).strip()
            # Normalise: collapse whitespace, make eval-friendly
            lhs_expr = re.sub(r"\s+", " ", lhs_raw)
            # Avoid duplicates from overlapping patterns
            if lhs_expr in seen_lhs:
                continue
            try:
                rhs_value = float(rhs_raw)
            except ValueError:
                continue
            seen_lhs.add(lhs_expr)
            equations.append(
                ArithmeticEquation(
                    lhs_expr=lhs_expr,
                    rhs_value=rhs_value,
                    stated_result=rhs_raw,
                    confidence=confidence,
                )
            )

    return equations


# ---------------------------------------------------------------------------
# CoACEExtractor — the public API
# ---------------------------------------------------------------------------


class CoACEExtractor:
    """Execution-based arithmetic constraint extractor for IT model CoT responses.

    **Why 'execution-based':**
        Instead of pattern-matching the RHS value (which can miss wrong answers that
        happen to match a pattern), we independently compute the LHS value via Python
        eval() and compare.  This catches any arithmetic error regardless of how the
        model phrased it, as long as the equation is syntactically recognisable.

    **Usage:**

        extractor = CoACEExtractor(tolerance=1e-6, min_confidence=0.5)
        result = extractor.extract("We add 47 and 28 to get 76, so the total is 76.")
        if result.n_violations > 0:
            print(f"Found {result.n_violations} arithmetic error(s)")

    Args:
        tolerance: Maximum absolute difference before declaring a violation.
                   Default 1e-6 handles floating-point noise (e.g., 1/3 ≈ 0.333…).
        min_confidence: Minimum equation confidence to attempt evaluation.
                        Default 0.5 includes all patterns.

    Spec: REQ-EXTRACT-033, REQ-EXTRACT-034
    """

    def __init__(self, tolerance: float = 1e-6, min_confidence: float = 0.5) -> None:
        self.tolerance = tolerance
        self.min_confidence = min_confidence

    def extract(self, response: str) -> CoACEResult:
        """Extract and execute all arithmetic equations in response.

        Parses equations, evaluates each LHS, compares to stated RHS, and returns
        a CoACEResult describing any discrepancies.

        Args:
            response: The full CoT response text from an LLM.

        Returns:
            CoACEResult with n_equations_found, n_violations, and violations list.
        """
        equations = _parse_arithmetic_equations(response)
        violations: list[CoACEViolation] = []

        for eq in equations:
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

        conf_weighted = sum(1 for v in violations if v.equation.confidence >= 0.8)

        return CoACEResult(
            n_equations_found=len(equations),
            n_violations=len(violations),
            violations=violations,
            extraction_mode="execution_based",
            confidence_weighted_violations=conf_weighted,
        )

    def to_constraint_terms(self, result: CoACEResult) -> list[dict]:
        """Convert CoACEResult violations to a lightweight constraint-term format.

        Each violation becomes a dict compatible with the VerifyRepairPipeline's
        ConstraintTerm interface (used for reporting, not JAX energy computation).
        The pipeline reads these dicts to decide whether repair is warranted.

        Returns a list of dicts with keys: name, lhs, computed, stated, abs_error.
        """
        terms = []
        for v in result.violations:
            terms.append(
                {
                    "name": f"arithmetic_violation_{v.equation.lhs_expr}",
                    "lhs": v.equation.lhs_expr,
                    "computed": v.computed_value,
                    "stated": v.stated_value,
                    "abs_error": v.absolute_error,
                    "confidence": v.equation.confidence,
                }
            )
        return terms
