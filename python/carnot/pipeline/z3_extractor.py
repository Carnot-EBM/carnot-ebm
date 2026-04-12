"""SMT-backed arithmetic extraction for reasoning traces.

Normalizes explicit equations, verbal arithmetic phrases, and approximate
numeric claims into Z3 constraints. Each extracted step is checked for
satisfiability and returned as a metadata-backed ``ConstraintResult`` so the
existing verification pipeline can consume it without structural changes.

Spec: REQ-VERIFY-001, REQ-VERIFY-009, SCENARIO-VERIFY-009
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import cast

from z3 import ArithRef, RatNumRef, RealVal, Solver, sat, simplify

from carnot.pipeline.extract import ConstraintResult

_NUMBER = r"-?\d+(?:\.\d+)?"
_APPROX = r"(?:about|around|approximately|approx(?:\.|imately)?|roughly)"
_EXPR = rf"{_NUMBER}(?:\s*(?:\*\*|[+\-*/])\s*{_NUMBER})+"
_EXPLICIT_PATTERN = re.compile(
    rf"(?P<prefix>{_APPROX}\s+)?"
    rf"(?<![\d.])(?P<expr>{_EXPR})\s*(?:=|is|are)\s*"
    rf"(?P<approx>{_APPROX}\s+)?(?P<claimed>{_NUMBER})"
    rf"(?!\d)"
    rf"(?!\.\d)"
    rf"(?!\s*(?:[+\-*/]\s*{_NUMBER}))",
    re.IGNORECASE,
)
_STEP_SPLIT_PATTERN = re.compile(r"\n+|(?<=[.?!])\s+")
_LATEX_FRACTION_PATTERN = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
_LATEX_TEXT_PATTERN = re.compile(r"\\text\{([^{}]+)\}")
_VERBAL_PATTERNS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(
            rf"half of (?P<operand>{_NUMBER})\s*(?:=|is|are)\s*"
            rf"(?P<approx>{_APPROX}\s+)?(?P<claimed>{_NUMBER})",
            re.IGNORECASE,
        ),
        "half_of",
        "operand / 2",
    ),
    (
        re.compile(
            rf"(?:double|twice) (?P<operand>{_NUMBER})\s*(?:=|is|are)\s*"
            rf"(?P<approx>{_APPROX}\s+)?(?P<claimed>{_NUMBER})",
            re.IGNORECASE,
        ),
        "double",
        "operand * 2",
    ),
    (
        re.compile(
            rf"(?:triple|three times) (?P<operand>{_NUMBER})\s*(?:=|is|are)\s*"
            rf"(?P<approx>{_APPROX}\s+)?(?P<claimed>{_NUMBER})",
            re.IGNORECASE,
        ),
        "triple",
        "operand * 3",
    ),
)


@dataclass(frozen=True)
class _StepConstraint:
    step_index: int
    step_text: str
    expression_text: str
    claimed_text: str
    approximate: bool
    operator: str


class Z3ArithmeticExtractor:
    """Verify arithmetic steps with Z3 rather than surface regex matches."""

    @property
    def supported_domains(self) -> list[str]:
        return ["arithmetic"]

    def extract(
        self, text: str, domain: str | None = None
    ) -> list[ConstraintResult]:
        if domain is not None and domain not in self.supported_domains:
            return []
        if not text.strip():
            return []

        normalized = self._normalize_text(text)
        if not normalized:
            return []

        results: list[ConstraintResult] = []
        seen: set[tuple[int, str, str, str]] = set()
        for step_index, step_text in enumerate(self._split_steps(normalized), start=1):
            for parsed in self._extract_step_constraints(step_text, step_index):
                key = (
                    parsed.step_index,
                    parsed.expression_text,
                    parsed.claimed_text,
                    parsed.operator,
                )
                if key in seen:
                    continue
                seen.add(key)
                result = self._solve_constraint(parsed)
                if result is not None:
                    results.append(result)
        return results

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\\$", "$").replace("\\%", "%")
        while True:
            updated = _LATEX_FRACTION_PATTERN.sub(r"\1 / \2", normalized)
            if updated == normalized:
                break
            normalized = updated
        normalized = _LATEX_TEXT_PATTERN.sub(r" \1 ", normalized)
        normalized = (
            normalized.replace("\\times", "*")
            .replace("×", "*")
            .replace("÷", "/")
            .replace("−", "-")
            .replace("$", "")
            .replace("\\", "")
        )
        normalized = re.sub(r"(?<=\d)\s*/\s*(?=[A-Za-z])", " ", normalized)
        normalized = re.sub(
            r"(?<=\d)\s+[A-Za-z]+(?:/[A-Za-z]+)*\s*(?=[=+\-*/])",
            " ",
            normalized,
        )
        normalized = re.sub(
            r"(?<=[+\-*/])\s*[A-Za-z]+(?:/[A-Za-z]+)*\s*(?=-?\d)",
            " ",
            normalized,
        )
        normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
        normalized = re.sub(r"[ \t]+", " ", normalized)
        normalized = re.sub(r"\n\s*", "\n", normalized)
        return normalized.strip()

    @staticmethod
    def _split_steps(text: str) -> list[str]:
        return [part.strip() for part in _STEP_SPLIT_PATTERN.split(text) if part.strip()]

    def _extract_step_constraints(
        self, step_text: str, step_index: int
    ) -> list[_StepConstraint]:
        parsed: list[_StepConstraint] = []
        for match in _EXPLICIT_PATTERN.finditer(step_text):
            parsed.append(
                _StepConstraint(
                    step_index=step_index,
                    step_text=step_text,
                    expression_text=match.group("expr").strip(),
                    claimed_text=match.group("claimed"),
                    approximate=bool(match.group("prefix") or match.group("approx")),
                    operator=self._infer_operator(match.group("expr")),
                )
            )

        for pattern, operator, template in _VERBAL_PATTERNS:
            match = pattern.search(step_text)
            if match is None:
                continue
            operand = match.group("operand")
            expression_text = template.replace("operand", operand)
            parsed.append(
                _StepConstraint(
                    step_index=step_index,
                    step_text=step_text,
                    expression_text=expression_text,
                    claimed_text=match.group("claimed"),
                    approximate=bool(match.group("approx")),
                    operator=operator,
                )
            )
        return parsed

    def _solve_constraint(self, parsed: _StepConstraint) -> ConstraintResult | None:
        expression = self._parse_expression(parsed.expression_text)
        if expression is None:
            return None

        claimed = self._parse_number(parsed.claimed_text)
        claimed_z3 = RealVal(str(claimed))
        solver = Solver()
        if parsed.approximate:
            tolerance = 5.0
            solver.add(expression >= RealVal(str(claimed - tolerance)))
            solver.add(expression <= RealVal(str(claimed + tolerance)))
        else:
            tolerance = 0.0
            solver.add(expression == claimed_z3)

        status = solver.check()
        correct = self._z3_value_to_number(expression)
        satisfied = status == sat

        description = f"{parsed.expression_text} = {self._format_number(claimed)}"
        if parsed.approximate:
            description = (
                f"{parsed.expression_text} ~= {self._format_number(claimed)}"
            )
        if not satisfied:
            description += f" (correct: {self._format_number(correct)})"

        return ConstraintResult(
            constraint_type="arithmetic",
            description=description,
            metadata={
                "operator": parsed.operator,
                "step_index": parsed.step_index,
                "step_text": parsed.step_text,
                "expression": parsed.expression_text,
                "claimed_result": claimed,
                "correct_result": correct,
                "approximate": parsed.approximate,
                "tolerance": tolerance,
                "satisfied": satisfied,
                "solver": "z3",
                "solver_status": str(status),
            },
        )

    def _parse_expression(self, expression_text: str) -> ArithRef | None:
        try:
            tree = ast.parse(expression_text, mode="eval")
        except SyntaxError:
            return None
        return self._ast_to_z3(tree.body)

    def _ast_to_z3(self, node: ast.AST) -> ArithRef | None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return None
            if isinstance(node.value, int | float):
                return RealVal(str(node.value))
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = self._ast_to_z3(node.operand)
            if operand is None:
                return None
            return -operand
        if isinstance(node, ast.BinOp):
            left = self._ast_to_z3(node.left)
            right = self._ast_to_z3(node.right)
            if left is None or right is None:
                return None
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        return None

    @staticmethod
    def _infer_operator(expression_text: str) -> str:
        for operator in ("**", "+", "-", "*", "/"):
            if operator in expression_text:
                return operator
        return "compound"

    @staticmethod
    def _parse_number(text: str) -> int | float:
        if "." in text:
            return float(text)
        return int(text)

    @staticmethod
    def _z3_value_to_number(expression: ArithRef) -> int | float:
        simplified = simplify(expression)
        if isinstance(simplified, RatNumRef):
            numerator = simplified.numerator_as_long()
            denominator = simplified.denominator_as_long()
            if denominator == 1:
                return numerator
            return numerator / denominator

        decimal = cast(str, simplified.as_decimal(12)).rstrip("?")
        return float(decimal)

    @staticmethod
    def _format_number(value: int | float) -> str:
        if isinstance(value, int):
            return str(value)
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
