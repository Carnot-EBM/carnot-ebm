"""Exact-rational Kolmogorov-Arnold Network forward pass.

This module is intentionally separate from the JAX KAN implementation. The JAX
path is useful for training, gradients, and fast approximate inference; the
rational path below is a small deterministic reference used when formal
verification needs every arithmetic step to be replayable as exact field
operations.

Spec references: REQ-KAN-1602, SCENARIO-KAN-1602.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence

RationalInput = Fraction | int | str
Edge = tuple[int, int]


def _repo_root() -> Path:
    """Return the repository root so artifact paths are stable from any cwd."""

    return Path(__file__).resolve().parents[3]


DEFAULT_RESULT_PATH = _repo_root() / "results/experiment_1602_rkan.json"


def to_fraction(value: RationalInput) -> Fraction:
    """Convert an explicit rational encoding to `Fraction`.

    Floats are intentionally rejected. `Fraction(0.1)` is deterministic, but it
    captures the binary floating-point approximation of 0.1, not the rational
    value a proof author usually meant. Accepting only integers, strings, and
    existing `Fraction` values keeps the verification surface unambiguous.
    """

    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not an exact rational model input")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError("RKAN exact rational inputs must be Fraction, int, or str")


def serialize_fraction(value: Fraction) -> str:
    """Serialize a `Fraction` without losing exact numerator/denominator state."""

    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


@dataclass(frozen=True)
class RationalLinearSpline:
    """Piecewise-linear KAN spline evaluated entirely over `Fraction`.

    The spline stores control points over a rational domain, defaulting to
    `[-1, 1]` to mirror the simple KAN activation domain used elsewhere in
    Carnot. Evaluation clamps outside the domain and linearly interpolates
    between adjacent rational control points inside the domain.
    """

    control_points: Sequence[RationalInput]
    domain: tuple[RationalInput, RationalInput] = (Fraction(-1, 1), Fraction(1, 1))

    def __post_init__(self) -> None:
        points = tuple(to_fraction(point) for point in self.control_points)
        if len(points) < 2:
            raise ValueError("RationalLinearSpline requires at least two control points")

        lo = to_fraction(self.domain[0])
        hi = to_fraction(self.domain[1])
        if lo >= hi:
            raise ValueError("RationalLinearSpline domain must satisfy min < max")

        object.__setattr__(self, "control_points", points)
        object.__setattr__(self, "domain", (lo, hi))

    def evaluate(self, x: RationalInput) -> Fraction:
        """Evaluate the spline at `x` using only rational arithmetic."""

        xq = to_fraction(x)
        lo, hi = self.domain
        if xq <= lo:
            return self.control_points[0]
        if xq >= hi:
            return self.control_points[-1]

        segment_count = len(self.control_points) - 1
        scaled = (xq - lo) * segment_count / (hi - lo)
        left_index = scaled.numerator // scaled.denominator
        t = scaled - left_index
        left_value = self.control_points[left_index]
        right_value = self.control_points[left_index + 1]
        return left_value + t * (right_value - left_value)

    def as_serializable(self) -> dict[str, list[str]]:
        """Return a JSON-safe exact representation of the spline."""

        lo, hi = self.domain
        return {
            "domain": [serialize_fraction(lo), serialize_fraction(hi)],
            "control_points": [serialize_fraction(point) for point in self.control_points],
        }


class RationalKANEnergyFunction:
    """Small exact-rational KAN energy function for formal verification.

    Energy follows the same edge-plus-bias shape as `KANEnergyFunction`:

        E(x) = sum_ij f_ij(x_i * x_j) + sum_i g_i(x_i)

    The difference is that every multiplication, interpolation, and addition is
    a `Fraction` operation. This class is not a trainer; it is a deterministic
    reference evaluator for proof-oriented checks.
    """

    def __init__(
        self,
        input_dim: int,
        edge_control_points: Mapping[Edge, Sequence[RationalInput]] | None = None,
        bias_control_points: Sequence[Sequence[RationalInput]] | None = None,
        domain: tuple[RationalInput, RationalInput] = (Fraction(-1, 1), Fraction(1, 1)),
    ) -> None:
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")

        self.input_dim = input_dim
        self.edge_splines: dict[Edge, RationalLinearSpline] = {}
        for edge, points in (edge_control_points or {}).items():
            i, j = edge
            if i == j or not (0 <= i < input_dim and 0 <= j < input_dim):
                raise ValueError(f"edge index {edge!r} is invalid for input_dim={input_dim}")
            self.edge_splines[(i, j)] = RationalLinearSpline(points, domain=domain)

        if bias_control_points is None:
            bias_control_points = tuple((0, 0) for _ in range(input_dim))
        if len(bias_control_points) != input_dim:
            raise ValueError("bias_control_points length must equal input_dim")

        self.bias_splines = tuple(
            RationalLinearSpline(points, domain=domain) for points in bias_control_points
        )

    @property
    def n_params(self) -> int:
        """Count rational control points in edge and bias splines."""

        edge_params = sum(len(spline.control_points) for spline in self.edge_splines.values())
        bias_params = sum(len(spline.control_points) for spline in self.bias_splines)
        return edge_params + bias_params

    def forward(self, x: Sequence[RationalInput]) -> Fraction:
        """Evaluate exact RKAN energy for a single input vector."""

        xq = tuple(to_fraction(value) for value in x)
        if len(xq) != self.input_dim:
            raise ValueError(f"expected {self.input_dim} inputs, got {len(xq)}")

        total = Fraction(0, 1)
        for (i, j), spline in self.edge_splines.items():
            total += spline.evaluate(xq[i] * xq[j])
        for i, spline in enumerate(self.bias_splines):
            total += spline.evaluate(xq[i])
        return total

    def energy(self, x: Sequence[RationalInput]) -> Fraction:
        """Alias for `forward` matching the other model energy APIs."""

        return self.forward(x)

    def __call__(self, x: Sequence[RationalInput]) -> Fraction:
        """Call shorthand for `forward`."""

        return self.forward(x)

    def energy_batch(self, xs: Sequence[Sequence[RationalInput]]) -> tuple[Fraction, ...]:
        """Evaluate a batch as a tuple of exact rational energies."""

        return tuple(self.forward(x) for x in xs)

    def as_serializable(self) -> dict[str, object]:
        """Return JSON-safe exact parameters for audit artifacts."""

        return {
            "input_dim": self.input_dim,
            "edges": {
                f"{i},{j}": spline.as_serializable()
                for (i, j), spline in self.edge_splines.items()
            },
            "biases": [spline.as_serializable() for spline in self.bias_splines],
        }


def _reference_model() -> RationalKANEnergyFunction:
    """Build the deterministic Exp 1602 reference RKAN."""

    return RationalKANEnergyFunction(
        input_dim=3,
        edge_control_points={
            (0, 1): [0, 1, 2],
            (1, 2): ["1/3", "2/3", "1"],
        },
        bias_control_points=[
            [0, 0, 0],
            ["1/2", "1/2", "1/2"],
            [-1, 0, 1],
        ],
    )


def build_experiment_1602_artifact() -> dict[str, object]:
    """Build the stable Exp 1602 exact-rational RKAN artifact payload."""

    model = _reference_model()
    sample_inputs = (
        (Fraction(1, 2), Fraction(-1, 2), Fraction(1, 3)),
        (Fraction(-1, 1), Fraction(0, 1), Fraction(1, 1)),
    )
    sample_outputs = []
    repeated_forward_outputs_identical = True

    for vector in sample_inputs:
        first = model.forward(vector)
        second = model.forward(tuple(Fraction(v.numerator, v.denominator) for v in vector))
        repeated_forward_outputs_identical = repeated_forward_outputs_identical and first == second
        sample_outputs.append(
            {
                "input": [serialize_fraction(value) for value in vector],
                "energy": serialize_fraction(first),
                "type": "fractions.Fraction",
                "repeat_energy": serialize_fraction(second),
            }
        )

    return {
        "schema": "carnot.rkan_exact_fraction.v1",
        "status": "complete",
        "experiment": 1602,
        "experiment_id": 1602,
        "run_date": "20260509",
        "title": "Exact-rational KAN forward pass",
        "spec": ["REQ-KAN-1602", "SCENARIO-KAN-1602"],
        "module": "python/carnot/models/rkan.py",
        "artifact_path": "results/experiment_1602_rkan.json",
        "model": model.as_serializable(),
        "input_dim": model.input_dim,
        "edge_count": len(model.edge_splines),
        "bias_count": len(model.bias_splines),
        "n_params": model.n_params,
        "fraction_type": "fractions.Fraction",
        "exact_rational_forward_pass_ready": True,
        "float_operations_used": False,
        "repeated_forward_outputs_identical": repeated_forward_outputs_identical,
        "sample_outputs": sample_outputs,
        "reference_energy": sample_outputs[0]["energy"],
        "honest_verdict": "complete: exact_rational_kan_forward_pass_uses_fraction_arithmetic",
    }


def write_experiment_1602_artifact(
    output_path: str | Path = DEFAULT_RESULT_PATH,
) -> dict[str, object]:
    """Write `results/experiment_1602_rkan.json` and return the payload."""

    artifact = build_experiment_1602_artifact()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


__all__ = [
    "DEFAULT_RESULT_PATH",
    "RationalKANEnergyFunction",
    "RationalLinearSpline",
    "build_experiment_1602_artifact",
    "serialize_fraction",
    "to_fraction",
    "write_experiment_1602_artifact",
]
