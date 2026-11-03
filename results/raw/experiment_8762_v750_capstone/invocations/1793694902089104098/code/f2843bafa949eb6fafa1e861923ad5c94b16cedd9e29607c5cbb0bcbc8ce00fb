"""Constraint-informed KAN verifier with fixed FourierCSP boundaries.

`CIKAN` is a small CPU verifier for Exp 1723.  It compiles FourierCSP
constraints into immutable boundary units, then trains only a residual
piecewise-linear KAN head.  The fixed boundary path contributes an additive
energy penalty whenever a logical constraint is violated, so training cannot
move the physical/logical constraint boundary itself.

Spec: REQ-KAN-1723, SCENARIO-KAN-1723.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


_TOKEN_RE = re.compile(r"\s*(AND|OR|XOR|NOT|[A-Za-z_][A-Za-z0-9_]*|\(|\))")


@dataclass(frozen=True)
class CIKANBoundary:
    """Immutable FourierCSP constraint boundary inside the CIKAN architecture.

    The boundary stores the original FourierCSP variables, expression, and
    polynomial text.  At inference it thresholds the selected input features and
    evaluates the Boolean expression.  A violation returns `1.0`; a satisfying
    assignment returns `0.0`.
    """

    name: str
    variables: tuple[str, ...]
    expression: str
    polynomial: str
    variable_indices: tuple[int, ...]
    penalty: float = 4.0
    threshold: float = 0.5

    def snapshot(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible snapshot for immutability checks."""

        return {
            "name": self.name,
            "variables": list(self.variables),
            "expression": self.expression,
            "polynomial": self.polynomial,
            "variable_indices": list(self.variable_indices),
            "penalty": float(self.penalty),
            "threshold": float(self.threshold),
        }

    def violation(self, x: Sequence[float]) -> float:
        """Return 1.0 when this fixed boundary is violated, else 0.0."""

        values = {
            variable: bool(float(x[index]) >= self.threshold)
            for variable, index in zip(self.variables, self.variable_indices, strict=True)
        }
        return 0.0 if _evaluate_boolean_expression(self.expression, values) else 1.0


class _BooleanExpressionParser:
    """Tiny recursive-descent parser for FourierCSP Boolean expressions."""

    def __init__(self, expression: str, values: Mapping[str, bool]) -> None:
        self.tokens = _tokenize(expression)
        self.values = values
        self.pos = 0

    def parse(self) -> bool:
        """Parse a full expression and reject trailing tokens."""

        result = self._parse_or()
        if self.pos != len(self.tokens):
            raise ValueError(f"unsupported token {self.tokens[self.pos]!r}")
        return result

    def _peek(self) -> str | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _take(self) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("unexpected end of expression")
        self.pos += 1
        return token

    def _parse_or(self) -> bool:
        result = self._parse_xor()
        while self._peek() == "OR":
            self._take()
            rhs = self._parse_xor()
            result = result or rhs
        return result

    def _parse_xor(self) -> bool:
        result = self._parse_and()
        while self._peek() == "XOR":
            self._take()
            rhs = self._parse_and()
            result = bool(result) ^ bool(rhs)
        return result

    def _parse_and(self) -> bool:
        result = self._parse_not()
        while self._peek() == "AND":
            self._take()
            rhs = self._parse_not()
            result = result and rhs
        return result

    def _parse_not(self) -> bool:
        if self._peek() == "NOT":
            self._take()
            return not self._parse_not()
        return self._parse_atom()

    def _parse_atom(self) -> bool:
        token = self._take()
        if token == "(":
            result = self._parse_or()
            if self._take() != ")":
                raise ValueError("missing closing parenthesis")
            return result
        if token in {")", "AND", "OR", "XOR", "NOT"}:
            raise ValueError(f"unsupported token {token!r}")
        if token not in self.values:
            raise ValueError(f"unknown FourierCSP variable {token!r}")
        return bool(self.values[token])


def _tokenize(expression: str) -> list[str]:
    """Tokenize a FourierCSP Boolean expression and reject unsupported syntax."""

    tokens: list[str] = []
    pos = 0
    while pos < len(expression):
        match = _TOKEN_RE.match(expression, pos)
        if not match:
            raise ValueError(f"unsupported token near {expression[pos:]!r}")
        token = match.group(1)
        pos = match.end()
        if token.upper() in {"AND", "OR", "XOR", "NOT"}:
            tokens.append(token.upper())
        else:
            tokens.append(token)
    if not tokens:
        raise ValueError("FourierCSP expression cannot be empty")
    return tokens


def _evaluate_boolean_expression(expression: str, values: Mapping[str, bool]) -> bool:
    """Evaluate a supported FourierCSP Boolean expression."""

    return _BooleanExpressionParser(expression, values).parse()


def _constraint_field(constraint: Any, field: str) -> Any:
    """Read one field from a FourierCSP dataclass or mapping."""

    if isinstance(constraint, Mapping):
        return constraint[field]
    return getattr(constraint, field)


class CIKAN:
    """Constraint-informed KAN with fixed FourierCSP architectural boundaries.

    The model energy is:

        E(x) = E_residual_splines(x) + sum_i penalty_i * violation_i(x)

    Only `E_residual_splines` is trained.  The `CIKANBoundary` objects are frozen
    dataclasses and the `fit()` loop never rewrites `self.boundaries`, preserving
    the FourierCSP constraint boundary as an architectural component.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        constraints: Sequence[Any] | None = None,
        *,
        boundary_penalty: float = 4.0,
        threshold: float = 0.5,
        n_knots: int = 5,
        learning_rate: float = 0.1,
        seed: int = 0,
    ) -> None:
        if not feature_names:
            raise ValueError("CIKAN requires at least one feature")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names must be unique")
        if boundary_penalty <= 0.0:
            raise ValueError("boundary_penalty must be positive")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if n_knots < 2:
            raise ValueError("n_knots must be at least 2")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")

        self.feature_names = tuple(str(name) for name in feature_names)
        self.boundary_penalty = float(boundary_penalty)
        self.threshold = float(threshold)
        self.n_knots = int(n_knots)
        self.learning_rate = float(learning_rate)
        self.knots = np.linspace(0.0, 1.0, self.n_knots, dtype=np.float64)

        rng = np.random.default_rng(seed)
        self.residual_control_points = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(len(self.feature_names), self.n_knots),
        ).astype(np.float64)
        self.bias = 0.0

        self.boundaries = tuple(
            self._compile_constraint(
                constraint=constraint,
                name=f"constraint_{idx}",
                boundary_penalty=self.boundary_penalty,
                threshold=self.threshold,
            )
            for idx, constraint in enumerate(constraints or ())
        )

    @classmethod
    def from_fouriercsp(
        cls,
        feature_names: Sequence[str],
        constraints: Sequence[Any],
        **kwargs: Any,
    ) -> CIKAN:
        """Build a CIKAN verifier from FourierCSP extractor outputs."""

        return cls(feature_names=feature_names, constraints=constraints, **kwargs)

    def _compile_constraint(
        self,
        constraint: Any,
        name: str,
        boundary_penalty: float,
        threshold: float,
    ) -> CIKANBoundary:
        variables = tuple(str(v) for v in _constraint_field(constraint, "variables"))
        expression = str(_constraint_field(constraint, "expression"))
        polynomial = str(_constraint_field(constraint, "polynomial"))

        feature_index = {name: idx for idx, name in enumerate(self.feature_names)}
        missing = [variable for variable in variables if variable not in feature_index]
        if missing:
            raise ValueError(f"unknown FourierCSP variable {missing[0]!r}")

        boundary = CIKANBoundary(
            name=name,
            variables=variables,
            expression=expression,
            polynomial=polynomial,
            variable_indices=tuple(feature_index[variable] for variable in variables),
            penalty=boundary_penalty,
            threshold=threshold,
        )

        # Validate the expression against a deterministic all-false assignment.
        boundary.violation(np.zeros(len(self.feature_names), dtype=np.float64))
        return boundary

    def boundary_snapshot(self) -> list[dict[str, Any]]:
        """Return snapshots of all fixed architectural boundaries."""

        return [boundary.snapshot() for boundary in self.boundaries]

    def boundary_violations(self, x: Sequence[float]) -> np.ndarray:
        """Return one violation flag per fixed boundary."""

        sample = self._coerce_sample(x)
        return np.array([boundary.violation(sample) for boundary in self.boundaries], dtype=np.float64)

    def boundary_energy(self, x: Sequence[float]) -> float:
        """Return the fixed penalty contributed by FourierCSP boundaries."""

        sample = self._coerce_sample(x)
        return float(sum(boundary.penalty * boundary.violation(sample) for boundary in self.boundaries))

    def residual_energy(self, x: Sequence[float]) -> float:
        """Return the trainable residual KAN spline energy for one sample."""

        sample = self._coerce_sample(x)
        total = float(self.bias)
        for feature_idx, value in enumerate(sample):
            basis = self._basis_values(float(value))
            total += float(np.dot(self.residual_control_points[feature_idx], basis))
        return total

    def energy(self, x: Sequence[float]) -> float:
        """Return scalar CIKAN energy; lower means more constraint-consistent."""

        return self.residual_energy(x) + self.boundary_energy(x)

    def energy_batch(self, xs: Sequence[Sequence[float]]) -> np.ndarray:
        """Return CIKAN energies for a batch of samples."""

        batch = self._coerce_batch(xs)
        return np.array([self.energy(row) for row in batch], dtype=np.float64)

    def predict_proba(self, xs: Sequence[Sequence[float]]) -> np.ndarray:
        """Return probability that each sample satisfies the constraints."""

        energies = self.energy_batch(xs)
        return 1.0 / (1.0 + np.exp(energies))

    def predict(self, xs: Sequence[Sequence[float]]) -> np.ndarray:
        """Return binary validity predictions from the CIKAN energy."""

        return (self.predict_proba(xs) >= 0.5).astype(np.float64)

    def fit(
        self,
        xs: Sequence[Sequence[float]],
        ys: Sequence[float],
        *,
        epochs: int = 50,
    ) -> list[float]:
        """Train only the residual KAN head with binary cross-entropy."""

        if epochs <= 0:
            raise ValueError("epochs must be positive")
        batch = self._coerce_batch(xs)
        labels = np.asarray(ys, dtype=np.float64)
        if labels.shape != (len(batch),):
            raise ValueError("ys must have one label per sample")

        history: list[float] = []
        for _ in range(epochs):
            probabilities = self.predict_proba(batch)
            history.append(_binary_cross_entropy(labels, probabilities))

            grad_ctrl = np.zeros_like(self.residual_control_points)
            grad_bias = 0.0
            for row, label, probability in zip(batch, labels, probabilities, strict=True):
                d_loss_d_energy = float(label - probability)
                grad_bias += d_loss_d_energy
                for feature_idx, value in enumerate(row):
                    grad_ctrl[feature_idx] += d_loss_d_energy * self._basis_values(float(value))

            scale = 1.0 / float(len(batch))
            self.bias -= self.learning_rate * grad_bias * scale
            self.residual_control_points -= self.learning_rate * grad_ctrl * scale
            np.clip(self.residual_control_points, -1.0, 1.0, out=self.residual_control_points)
            self.bias = float(np.clip(self.bias, -1.0, 1.0))

        return history

    def evaluate(self, xs: Sequence[Sequence[float]], ys: Sequence[float]) -> dict[str, float]:
        """Evaluate toy binary verification metrics."""

        batch = self._coerce_batch(xs)
        labels = np.asarray(ys, dtype=np.float64)
        if labels.shape != (len(batch),):
            raise ValueError("ys must have one label per sample")

        energies = self.energy_batch(batch)
        predictions = self.predict(batch)
        valid_energies = energies[labels >= 0.5]
        invalid_energies = energies[labels < 0.5]
        mean_valid = float(np.mean(valid_energies)) if len(valid_energies) else math.nan
        mean_invalid = float(np.mean(invalid_energies)) if len(invalid_energies) else math.nan

        return {
            "accuracy": float(np.mean(predictions == labels)),
            "mean_valid_energy": mean_valid,
            "mean_invalid_energy": mean_invalid,
            "energy_gap": float(mean_invalid - mean_valid),
        }

    def _basis_values(self, x: float) -> np.ndarray:
        """Return piecewise-linear hat-basis values for the residual KAN head."""

        x_clipped = float(np.clip(x, 0.0, 1.0))
        if x_clipped >= 1.0:
            basis = np.zeros(self.n_knots, dtype=np.float64)
            basis[-1] = 1.0
            return basis

        scaled = x_clipped * (self.n_knots - 1)
        left = int(math.floor(scaled))
        right = min(left + 1, self.n_knots - 1)
        frac = scaled - left
        basis = np.zeros(self.n_knots, dtype=np.float64)
        basis[left] = 1.0 - frac
        basis[right] += frac
        return basis

    def _coerce_sample(self, x: Sequence[float]) -> np.ndarray:
        sample = np.asarray(x, dtype=np.float64)
        if sample.shape != (len(self.feature_names),):
            raise ValueError(f"sample must have shape ({len(self.feature_names)},)")
        return sample

    def _coerce_batch(self, xs: Sequence[Sequence[float]]) -> np.ndarray:
        batch = np.asarray(xs, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != len(self.feature_names):
            raise ValueError(f"batch must have shape (n, {len(self.feature_names)})")
        return batch


def _binary_cross_entropy(labels: np.ndarray, probabilities: np.ndarray) -> float:
    probs = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    return float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))


__all__ = [
    "CIKAN",
    "CIKANBoundary",
]
