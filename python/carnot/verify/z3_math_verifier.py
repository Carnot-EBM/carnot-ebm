"""Symbolic arithmetic verifier for structurally diverse verifier ensembles.

Spec: REQ-VERIFY-1107, SCENARIO-VERIFY-1107
"""

from __future__ import annotations

import ast
import operator
import re
from fractions import Fraction
from typing import Any

try:  # pragma: no cover - availability is environment dependent.
    import z3  # type: ignore[import]
except Exception:  # pragma: no cover - fallback is tested when z3 is absent.
    z3 = None  # type: ignore[assignment]


_ALLOWED_ARITH_CHARS = set("0123456789., \t+-*/()^%")
_BINARY_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}


class Z3MathVerifier:
    """Symbolic math verifier using Z3 SMT solver.

    Extracts concrete arithmetic equations such as ``47 + 28 = 75`` and
    numeric comparisons such as ``9.11 > 9.9`` from reasoning text, then checks
    whether each claim is true. Energy is the fraction of extracted claims that
    violate deterministic arithmetic. The kernel is numeric correctness via
    formal or exact arithmetic, not token fluency.
    """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(z3_available={self.z3_available})"

    @property
    def z3_available(self) -> bool:
        """Return whether the optional Z3 backend imported successfully."""
        return z3 is not None

    def score(self, text: str) -> float:
        """Return arithmetic violation energy in [0, 1]."""
        if not text or not text.strip():
            return 0.5

        equations = self._extract_equations(text)
        comparisons = self._extract_comparisons(text)
        if not equations and not comparisons:
            return 0.5

        violations = 0
        checked = 0
        for left, right in equations:
            try:
                checked += 1
                if not self._equation_holds(left, right):
                    violations += 1
            except Exception:
                return 0.5

        for left, op, right in comparisons:
            try:
                checked += 1
                if not self._comparison_holds(left, op, right):
                    violations += 1
            except Exception:
                return 0.5

        if checked == 0:
            return 0.5
        return float(max(0.0, min(1.0, violations / checked)))

    def _extract_equations(self, text: str) -> list[tuple[str, str]]:
        normalized = _normalize_math_text(text)
        equations: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for match in re.finditer("=", normalized):
            left = _take_arithmetic_left(normalized, match.start())
            right = _take_arithmetic_right(normalized, match.end())
            if not left or not right:
                continue
            if not (_has_binary_operator(left) or _has_binary_operator(right)):
                continue
            if not (_is_parseable_arithmetic(left) and _is_parseable_arithmetic(right)):
                continue
            pair = (left, right)
            if pair not in seen:
                seen.add(pair)
                equations.append(pair)
        return equations

    def _extract_comparisons(self, text: str) -> list[tuple[str, str, str]]:
        normalized = _normalize_math_text(text)
        comparisons: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()

        for match in re.finditer(r"(?<![<>=!])(?:<=|>=|<|>)(?![<>=])", normalized):
            op = match.group(0)
            left = _take_arithmetic_left(normalized, match.start())
            right = _take_arithmetic_right(normalized, match.end())
            if not left or not right:
                continue
            if not (_is_parseable_arithmetic(left) and _is_parseable_arithmetic(right)):
                continue
            claim = (left, op, right)
            if claim not in seen:
                seen.add(claim)
                comparisons.append(claim)

        number = r"[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
        word_ops = {
            "greater than": ">",
            "larger than": ">",
            "more than": ">",
            "less than": "<",
            "smaller than": "<",
            "fewer than": "<",
            "at least": ">=",
            "at most": "<=",
        }
        word_pattern = re.compile(
            rf"(?P<left>{number})\s+(?:is\s+)?(?P<op>{'|'.join(map(re.escape, word_ops))})\s+"
            rf"(?P<right>{number})",
            flags=re.IGNORECASE,
        )
        for match in word_pattern.finditer(normalized):
            left = match.group("left")
            op = word_ops[match.group("op").lower()]
            right = match.group("right")
            if not (_is_parseable_arithmetic(left) and _is_parseable_arithmetic(right)):
                continue
            claim = (left, op, right)
            if claim not in seen:
                seen.add(claim)
                comparisons.append(claim)

        return comparisons

    def _equation_holds(self, left: str, right: str) -> bool:
        if z3 is not None:
            try:
                left_expr = _to_z3(_parse_arithmetic(left))
                right_expr = _to_z3(_parse_arithmetic(right))
                solver = z3.Solver()
                solver.add(left_expr != right_expr)
                return solver.check() == z3.unsat
            except Exception:
                pass
        return _eval_arithmetic(left) == _eval_arithmetic(right)

    def _comparison_holds(self, left: str, op: str, right: str) -> bool:
        left_value = _eval_arithmetic(left)
        right_value = _eval_arithmetic(right)
        if op == ">":
            return left_value > right_value
        if op == "<":
            return left_value < right_value
        if op == ">=":
            return left_value >= right_value
        if op == "<=":
            return left_value <= right_value
        raise ValueError(f"unsupported comparison operator: {op}")


def _normalize_math_text(text: str) -> str:
    out = text
    out = re.sub(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}", r"(\1)/(\2)", out)
    replacements = {
        "\\times": "*",
        "\\cdot": "*",
        "\\div": "/",
        "\\(": " ",
        "\\)": " ",
        "\\[": " ",
        "\\]": " ",
        "$": "",
        "×": "*",
        "÷": "/",
        "−": "-",
        "–": "-",
        "—": "-",
        "^": "**",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out


def _take_arithmetic_left(text: str, eq_index: int) -> str:
    i = eq_index - 1
    while i >= 0 and text[i] in _ALLOWED_ARITH_CHARS:
        i -= 1
    expr = text[i + 1 : eq_index].strip()
    return _clean_expression_edge(expr)


def _take_arithmetic_right(text: str, start_index: int) -> str:
    i = start_index
    while i < len(text) and text[i] in _ALLOWED_ARITH_CHARS:
        i += 1
    expr = text[start_index:i].strip()
    return _clean_expression_edge(expr)


def _clean_expression_edge(expr: str) -> str:
    expr = expr.strip(" \t,;:")
    while expr.startswith("- ") and not _starts_with_signed_number(expr):
        expr = expr[2:].lstrip()
    return expr.strip(" \t,;:")


def _starts_with_signed_number(expr: str) -> bool:
    return bool(re.match(r"^[+-]\s*\d", expr))


def _has_binary_operator(expr: str) -> bool:
    cleaned = re.sub(r"^\s*[+-]\s*", "", expr)
    return any(op in cleaned for op in ("+", "-", "*", "/", "**"))


def _is_parseable_arithmetic(expr: str) -> bool:
    try:
        _parse_arithmetic(expr)
        return True
    except Exception:
        return False


def _parse_arithmetic(expr: str) -> ast.Expression:
    cleaned = _prepare_expression(expr)
    parsed = ast.parse(cleaned, mode="eval")
    _validate_ast(parsed)
    return parsed


def _prepare_expression(expr: str) -> str:
    cleaned = expr.replace(",", "")
    cleaned = re.sub(r"(?P<num>(?:\d+(?:\.\d*)?|\.\d+))\s*%", r"(\g<num>/100)", cleaned)
    cleaned = cleaned.replace("^", "**")
    return cleaned.strip()


def _validate_ast(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate_ast(node.body)
        return
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, _BINARY_OPS):
            raise ValueError("unsupported arithmetic operator")
        _validate_ast(node.left)
        _validate_ast(node.right)
        return
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError("unsupported unary operator")
        _validate_ast(node.operand)
        return
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return
    raise ValueError("unsupported arithmetic expression")


def _eval_arithmetic(expr: str) -> Fraction:
    return _eval_ast(_parse_arithmetic(expr).body)


def _eval_ast(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp):
        value = _eval_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Div) and right == 0:
            raise ZeroDivisionError("division by zero")
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right.numerator) > 8:
                raise ValueError("unsupported exponent")
            return left ** int(right)
        return _OPS[type(node.op)](left, right)
    raise ValueError("unsupported arithmetic expression")


def _to_z3(parsed: ast.Expression) -> Any:
    return _ast_to_z3(parsed.body)


def _ast_to_z3(node: ast.AST) -> Any:
    if z3 is None:
        raise RuntimeError("z3 unavailable")
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        value = Fraction(str(node.value))
        return z3.RealVal(f"{value.numerator}/{value.denominator}")
    if isinstance(node, ast.UnaryOp):
        value = _ast_to_z3(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _ast_to_z3(node.left)
        right = _ast_to_z3(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            raise ValueError("z3 exponent fallback")
    raise ValueError("unsupported arithmetic expression")
