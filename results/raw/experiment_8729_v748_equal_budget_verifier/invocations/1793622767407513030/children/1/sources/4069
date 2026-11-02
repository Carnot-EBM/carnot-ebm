"""Projection repair for simple arithmetic equality violations.

Spec: REQ-VERIFY-1147, SCENARIO-VERIFY-1147
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from ..verify.z3_math_verifier import (
    Z3MathVerifier,
    _eval_arithmetic,
    _has_binary_operator,
)

_TEXT_KEYS = ("equation", "constraint", "constraint_text", "source_text", "claim", "raw_claim")
_AFTER_NUMBER = r"(?=$|[^\d.]|\.(?!\d))"


@dataclass(frozen=True)
class _ProjectionPatch:
    """Concrete number replacement selected by the arithmetic projection."""

    incorrect: str
    correct: str
    side: Literal["left", "right", "number"]


class ArithmeticProjectionRepair:
    """Repair arithmetic equality violations by projecting onto the valid result.

    This repair path is deliberately narrower than prompt-based repair. It only
    handles equations where one side is an arithmetic expression and the other
    side is the model's claimed number, such as ``47 + 28 = 76``. The projection
    is the exact arithmetic value of the expression side.
    """

    def __init__(self) -> None:
        self._verifier = Z3MathVerifier()

    def repair(self, response: str, violation: dict) -> str:
        """Return ``response`` with a violated arithmetic answer patched.

        Parameters
        ----------
        response:
            Model response text containing the violated arithmetic claim.
        violation:
            Z3-style arithmetic violation metadata. The repair accepts equation
            text under common keys such as ``constraint`` or ``source_text`` and
            also accepts the prompt's compact ``{"lhs": correct, "rhs": wrong}``
            shape for answer-only responses.
        """

        patch = self._build_patch(response, violation)
        if patch is None:
            return response
        return _apply_patch(response, patch)

    def _build_patch(self, response: str, violation: dict) -> _ProjectionPatch | None:
        for left, right in self._candidate_equations(response, violation):
            patch = _projection_for_equation(left, right)
            if patch is not None:
                return patch
        return _patch_from_violation_numbers(violation)

    def _candidate_equations(self, response: str, violation: dict) -> list[tuple[str, str]]:
        equations: list[tuple[str, str]] = []
        for text in _candidate_texts(response, violation):
            for left, right in self._verifier._extract_equations(text):
                if not self._verifier._equation_holds(left, right):
                    equations.append((left, right))
        return equations


def _candidate_texts(response: str, violation: dict) -> list[str]:
    texts: list[str] = []
    for key in _TEXT_KEYS:
        value = violation.get(key)
        if isinstance(value, str) and "=" in value and value not in texts:
            texts.append(value)
    if response not in texts:
        texts.append(response)
    return texts


def _projection_for_equation(left: str, right: str) -> _ProjectionPatch | None:
    left_is_expr = _has_binary_operator(left)
    right_is_expr = _has_binary_operator(right)
    if left_is_expr and not right_is_expr:
        correct = _format_fraction(_eval_arithmetic(left))
        return _ProjectionPatch(incorrect=right, correct=correct, side="right")
    if right_is_expr and not left_is_expr:
        correct = _format_fraction(_eval_arithmetic(right))
        return _ProjectionPatch(incorrect=left, correct=correct, side="left")
    return None


def _patch_from_violation_numbers(violation: dict) -> _ProjectionPatch | None:
    correct = _coerce_violation_number(violation, ("lhs", "correct_result", "computed_result"))
    incorrect = _coerce_violation_number(violation, ("rhs", "claimed_result", "actual_result"))
    if correct is None or incorrect is None or correct == incorrect:
        return None
    return _ProjectionPatch(
        incorrect=_format_fraction(incorrect),
        correct=_format_fraction(correct),
        side="number",
    )


def _coerce_violation_number(violation: dict, keys: tuple[str, ...]) -> Fraction | None:
    for key in keys:
        value = violation.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return Fraction(str(value))
    return None


def _apply_patch(response: str, patch: _ProjectionPatch) -> str:
    if patch.side == "right":
        pattern = rf"(=\s*){re.escape(patch.incorrect)}{_AFTER_NUMBER}"
        fixed = re.sub(pattern, rf"\g<1>{patch.correct}", response, count=1)
        if fixed != response:
            return fixed
    if patch.side == "left":
        pattern = rf"(?<![\d.]){re.escape(patch.incorrect)}(\s*=)"
        fixed = re.sub(pattern, rf"{patch.correct}\g<1>", response, count=1)
        if fixed != response:
            return fixed
    return _replace_standalone_number(response, patch.incorrect, patch.correct)


def _replace_standalone_number(response: str, incorrect: str, correct: str) -> str:
    pattern = rf"(?<![\d.]){re.escape(incorrect)}{_AFTER_NUMBER}"
    return re.sub(pattern, correct, response, count=1)


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):.12g}"
