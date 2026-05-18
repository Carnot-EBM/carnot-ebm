"""Eidoku CSP gate for hard arithmetic constraints.

The gate extracts numeric assignments from an LLM response, evaluates Python
expression constraints through an AST whitelist, and rejects outputs that fail
any hard structural constraint.

Spec: REQ-VERIFY-2354, SCENARIO-VERIFY-2354
"""

from __future__ import annotations

import ast
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


EXPERIMENT_ID = 2354
RUN_DATE = "20260518"
RANDOM_SEED = 42
VALIDATION_THRESHOLD = 0.75
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[3] / "results" / "experiment_2354_eidoku_csp.json"
)

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": "Terminal-prefix required.",
    "eidoku_gate_validated": "True if csp_gate_accuracy >= 0.75.",
    "csp_gate_accuracy": "Fraction of 50 examples correctly classified by the CSP gate.",
    "n_eval_examples": "Must be 50.",
    "random_seed": "Reproducibility. Must be 42.",
}

_ASSIGNMENT_RE = re.compile(
    r"(?<![A-Za-z0-9_])['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*"
    r"(?:=|:)\s*\$?([-+]?\d[\d,_]*(?:\.\d+)?)"
)


@dataclass(frozen=True)
class ArithmeticConstraintExample:
    """One deterministic arithmetic response and its expected CSP gate label."""

    response: str
    constraints: list[str]
    expected_gate_passed: bool


class EidokuCspGate:
    """Validate response assignments against hard arithmetic constraints."""

    def validate(self, response: str, constraints: list[str]) -> dict[str, Any]:
        """Return whether all constraints pass and which constraints violate.

        Args:
            response: Free-form response containing numeric assignments such as
                ``"x = 20; y = 22; total = 42"`` or a dict-like literal.
            constraints: Python expression strings, for example
                ``["x + y == total"]``.

        Returns:
            ``{"gate_passed": bool, "violations": list[str]}``. A constraint
            is a violation when it evaluates to ``False`` or cannot be safely
            evaluated.

        Spec: REQ-VERIFY-2354, SCENARIO-VERIFY-2354
        """

        variables = extract_numeric_assignments(response)
        violations: list[str] = []
        for constraint in constraints:
            result = safe_eval(constraint, variables)
            if result is not True:
                violations.append(constraint)
        return {"gate_passed": not violations, "violations": violations}


def extract_numeric_assignments(response: str) -> dict[str, float]:
    """Extract numeric variable assignments from dict literals and prose text."""

    assignments: dict[str, float] = {}
    _merge_literal_assignments(response, assignments)

    for match in _ASSIGNMENT_RE.finditer(response):
        parsed = _parse_number(match.group(2))
        if parsed is not None:
            assignments[match.group(1)] = parsed

    return assignments


def safe_eval(expression: str, variables: Mapping[str, float] | None = None) -> Any:
    """Evaluate a restricted arithmetic expression without Python ``eval``.

    Numeric literals are parsed through ``ast.literal_eval`` after the expression
    has been parsed into an AST. Names must be present in ``variables``. Calls,
    attributes, subscripts, comprehensions, imports, and all other syntax outside
    the arithmetic whitelist return ``None``.
    """

    if not expression.strip():
        return None
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree.body, dict(variables or {}))
    except (ArithmeticError, KeyError, SyntaxError, TypeError, ValueError, OverflowError):
        return None


def build_arithmetic_constraint_corpus(seed: int = RANDOM_SEED) -> list[ArithmeticConstraintExample]:
    """Build the deterministic 25-pass/25-fail arithmetic CSP corpus."""

    rng = random.Random(seed)
    examples: list[ArithmeticConstraintExample] = []
    constraints = ["x + y == total", "x * scale == scaled_x", "total - y == x"]

    for _ in range(25):
        x = rng.randint(-30, 70)
        y = rng.randint(-20, 60)
        scale = rng.randint(2, 5)
        total = x + y
        scaled_x = x * scale
        response = _format_response(x=x, y=y, total=total, scale=scale, scaled_x=scaled_x)
        examples.append(
            ArithmeticConstraintExample(
                response=response,
                constraints=list(constraints),
                expected_gate_passed=True,
            )
        )

    for _ in range(25):
        x = rng.randint(-30, 70)
        y = rng.randint(-20, 60)
        scale = rng.randint(2, 5)
        total = x + y + rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
        scaled_x = x * scale
        response = _format_response(x=x, y=y, total=total, scale=scale, scaled_x=scaled_x)
        examples.append(
            ArithmeticConstraintExample(
                response=response,
                constraints=list(constraints),
                expected_gate_passed=False,
            )
        )

    return examples


def evaluate_corpus(
    examples: list[ArithmeticConstraintExample] | None = None,
    *,
    gate: EidokuCspGate | None = None,
) -> dict[str, Any]:
    """Return classification accuracy for the arithmetic CSP corpus."""

    eval_examples = examples if examples is not None else build_arithmetic_constraint_corpus()
    evaluator = gate or EidokuCspGate()
    correct = 0
    for example in eval_examples:
        result = evaluator.validate(example.response, example.constraints)
        if result["gate_passed"] is example.expected_gate_passed:
            correct += 1

    n_examples = len(eval_examples)
    accuracy = correct / n_examples if n_examples else 0.0
    return {
        "correct_classifications": correct,
        "csp_gate_accuracy": accuracy,
        "eidoku_gate_validated": accuracy >= VALIDATION_THRESHOLD,
        "n_eval_examples": n_examples,
    }


def build_experiment_artifact(seed: int = RANDOM_SEED) -> dict[str, Any]:
    """Build the Exp 2354 terminal artifact payload."""

    corpus = build_arithmetic_constraint_corpus(seed=seed)
    metrics = evaluate_corpus(corpus)
    validated = bool(metrics["eidoku_gate_validated"])
    verdict_prefix = "complete" if validated else "failed"
    return {
        "experiment": EXPERIMENT_ID,
        "schema": "eidoku_csp_gate_v1",
        "run_date": RUN_DATE,
        "status": "complete" if validated else "failed",
        "spec_refs": ["REQ-VERIFY-2354", "SCENARIO-VERIFY-2354"],
        "honest_verdict": (
            f"{verdict_prefix}: Eidoku CSP gate accuracy "
            f"{metrics['csp_gate_accuracy']:.3f} over {metrics['n_eval_examples']} examples."
        ),
        "random_seed": seed,
        "field_principles": dict(FIELD_PRINCIPLES),
        **metrics,
    }


def write_experiment_artifact(
    path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Write the Exp 2354 JSON artifact and return the payload."""

    artifact = build_experiment_artifact(seed=seed)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _merge_literal_assignments(response: str, assignments: dict[str, float]) -> None:
    try:
        literal = ast.literal_eval(response)
    except (SyntaxError, ValueError):
        return
    if not isinstance(literal, Mapping):
        return
    for key, value in literal.items():
        if not isinstance(key, str):
            continue
        parsed = _coerce_numeric(value)
        if parsed is not None:
            assignments[key] = parsed


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, str):
        return _parse_number(value)
    return None


def _parse_number(text: str) -> float | None:
    try:
        value = ast.literal_eval(text.replace(",", ""))
    except (SyntaxError, ValueError):
        return None
    return _coerce_numeric(value)


def _eval_node(node: ast.AST, variables: dict[str, float]) -> Any:
    if isinstance(node, ast.Constant):
        value = ast.literal_eval(node)
        numeric = _coerce_numeric(value)
        if numeric is None:
            raise ValueError("non-numeric literal")
        return numeric

    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise KeyError(node.id)
        return _coerce_numeric(variables[node.id])

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _require_number(_eval_node(node.operand, variables))
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp):
        left = _require_number(_eval_node(node.left, variables))
        right = _require_number(_eval_node(node.right, variables))
        return _eval_binary(node.op, left, right)

    if isinstance(node, ast.Compare):
        return _eval_compare(node, variables)

    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        values = [_require_bool(_eval_node(value, variables)) for value in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)

    raise ValueError(f"unsupported AST node: {type(node).__name__}")


def _eval_binary(op: ast.operator, left: float, right: float) -> float:
    if isinstance(op, ast.Add):
        return _finite(left + right)
    if isinstance(op, ast.Sub):
        return _finite(left - right)
    if isinstance(op, ast.Mult):
        return _finite(left * right)
    if isinstance(op, ast.Div):
        if right == 0:
            raise ZeroDivisionError("division by zero")
        return _finite(left / right)
    if isinstance(op, ast.FloorDiv):
        if right == 0:
            raise ZeroDivisionError("division by zero")
        return _finite(left // right)
    if isinstance(op, ast.Mod):
        if right == 0:
            raise ZeroDivisionError("modulo by zero")
        return _finite(left % right)
    if isinstance(op, ast.Pow):
        if abs(right) > 12:
            raise OverflowError("power too large")
        return _finite(left**right)
    raise ValueError(f"unsupported binary operator: {type(op).__name__}")


def _eval_compare(node: ast.Compare, variables: dict[str, float]) -> bool:
    left = _require_number(_eval_node(node.left, variables))
    for op, comparator in zip(node.ops, node.comparators, strict=True):
        right = _require_number(_eval_node(comparator, variables))
        if not _compare(op, left, right):
            return False
        left = right
    return True


def _compare(op: ast.cmpop, left: float, right: float) -> bool:
    if isinstance(op, ast.Eq):
        return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(op, ast.NotEq):
        return not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right or math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right or math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)
    raise ValueError(f"unsupported comparison operator: {type(op).__name__}")


def _require_number(value: Any) -> float:
    numeric = _coerce_numeric(value)
    if numeric is None:
        raise TypeError("expected numeric value")
    return numeric


def _require_bool(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError("expected boolean value")
    return value


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise OverflowError("non-finite result")
    return float(value)


def _format_response(*, x: int, y: int, total: int, scale: int, scaled_x: int) -> str:
    return f"x = {x}; y = {y}; total = {total}; scale = {scale}; scaled_x = {scaled_x}"


if __name__ == "__main__":  # pragma: no cover - exercised by experiment command.
    write_experiment_artifact()
