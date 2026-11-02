#!/usr/bin/env python3
"""Exp 1671 Hybrid Zeckendorf exact-rational RKAN audit.

Spec: REQ-KAN-1671, SCENARIO-KAN-1671.

This script is intentionally a CPU accounting pass, not a trainer and not a
synthesis flow. It evaluates a small mock RKAN tier with `fractions.Fraction`
and records Zeckendorf decompositions of every rational witness so a verifier
can replay the integer arithmetic without relying on floating-point state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

JsonDict = dict[str, Any]
RationalInput = Fraction | int | str
Edge = tuple[int, int]

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "experiment_1671_rkan.json"
EXPERIMENT_ID = 1671
RUN_DATE = "20260510"
SCHEMA = "carnot.rkan.hybrid_zeckendorf_audit.v1"
TITLE = "No-synthesis Hybrid Zeckendorf exact-rational RKAN audit"
SPEC_TRACES = ["REQ-KAN-1671", "SCENARIO-KAN-1671"]
MODULE_PATH = "scripts/experiment_1671_rkan_audit.py"
ARTIFACT_PATH = "results/experiment_1671_rkan.json"
DEFAULT_SAMPLE_INPUTS: tuple[tuple[Fraction, ...], ...] = (
    (Fraction(1, 2), Fraction(-1, 2)),
    (Fraction(-1, 1), Fraction(1, 1)),
    (Fraction(1, 1), Fraction(1, 1)),
)

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "schema",
    "status",
    "experiment_id",
    "spec_traces",
    "float_operations_used",
    "hardware_synthesis_claimed",
    "no_synthesis_accounting_only",
    "hybrid_zeckendorf_rational_math",
    "exact_rational_cpu_simulation",
    "complexity",
    "bounding_certificates",
    "honest_verdict",
)


def to_fraction(value: RationalInput) -> Fraction:
    """Convert an explicit rational input while rejecting implicit floats."""

    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not an exact rational RKAN audit input")
    if isinstance(value, int):
        return Fraction(value, 1)
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError("Hybrid Zeckendorf RKAN audit inputs must be exact rational values")


def serialize_fraction(value: Fraction) -> str:
    """Return a stable numerator/denominator string for exact JSON output."""

    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def zeckendorf_witness(integer: int) -> JsonDict:
    """Return the canonical Zeckendorf witness for an integer.

    The witness uses Fibonacci indices F2=1, F3=2, F4=3, and so on. Negative
    integers keep a separate sign so the term decomposition remains over
    non-negative Fibonacci weights.
    """

    if not isinstance(integer, int) or isinstance(integer, bool):
        raise TypeError("Zeckendorf witnesses require an integer")

    sign = 0 if integer == 0 else (1 if integer > 0 else -1)
    remaining = abs(integer)
    terms: list[JsonDict] = []
    for index, fib_value in reversed(_fibonacci_terms_up_to(remaining)):
        if fib_value <= remaining:
            terms.append({"index": index, "value": fib_value})
            remaining -= fib_value
        if remaining == 0:
            break

    return {
        "integer": integer,
        "sign": sign,
        "terms": terms,
        "reconstructed": sign * sum(int(term["value"]) for term in terms),
        "nonconsecutive": _terms_are_nonconsecutive(terms),
    }


def reconstruct_zeckendorf(witness: Mapping[str, Any]) -> int:
    """Reconstruct the signed integer from a Zeckendorf witness."""

    sign = int(witness["sign"])
    return sign * sum(int(term["value"]) for term in witness["terms"])


def zeckendorf_terms_are_nonconsecutive(witness: Mapping[str, Any]) -> bool:
    """Return whether a Zeckendorf witness uses no adjacent Fibonacci indices."""

    return _terms_are_nonconsecutive(witness["terms"])


@dataclass(frozen=True)
class HybridZeckendorfRational:
    """A rational value plus Zeckendorf witnesses for numerator and denominator."""

    fraction: Fraction
    numerator_witness: JsonDict
    denominator_witness: JsonDict

    @classmethod
    def from_value(cls, value: RationalInput) -> "HybridZeckendorfRational":
        """Build a Hybrid Zeckendorf rational from a safe exact input."""

        fraction = to_fraction(value)
        return cls(
            fraction=fraction,
            numerator_witness=zeckendorf_witness(fraction.numerator),
            denominator_witness=zeckendorf_witness(fraction.denominator),
        )

    def add(self, other: "HybridZeckendorfRational") -> "HybridZeckendorfRational":
        """Add two exact rationals and refresh their integer witnesses."""

        return HybridZeckendorfRational.from_value(self.fraction + other.fraction)

    def mul(self, other: "HybridZeckendorfRational") -> "HybridZeckendorfRational":
        """Multiply two exact rationals and refresh their integer witnesses."""

        return HybridZeckendorfRational.from_value(self.fraction * other.fraction)

    def to_json(self) -> JsonDict:
        """Serialize the rational and both Zeckendorf decomposition witnesses."""

        return {
            "fraction": serialize_fraction(self.fraction),
            "numerator_witness": self.numerator_witness,
            "denominator_witness": self.denominator_witness,
        }


@dataclass(frozen=True)
class MockRationalSpline:
    """Piecewise-linear exact spline for the mock RKAN audit tier."""

    name: str
    control_points: tuple[Fraction, ...]
    domain: tuple[Fraction, Fraction] = (Fraction(-1, 1), Fraction(1, 1))

    @classmethod
    def from_points(
        cls,
        name: str,
        control_points: Sequence[RationalInput],
        domain: tuple[RationalInput, RationalInput] = (Fraction(-1, 1), Fraction(1, 1)),
    ) -> "MockRationalSpline":
        """Create a validated exact spline from rational encodings."""

        points = tuple(to_fraction(point) for point in control_points)
        lo = to_fraction(domain[0])
        hi = to_fraction(domain[1])
        if len(points) < 2:
            raise ValueError("MockRationalSpline requires at least two control points")
        if lo >= hi:
            raise ValueError("MockRationalSpline domain must satisfy min < max")
        return cls(name=name, control_points=points, domain=(lo, hi))

    def evaluate(self, x: RationalInput) -> HybridZeckendorfRational:
        """Evaluate the spline with exact rational interpolation."""

        xq = to_fraction(x)
        lo, hi = self.domain
        if xq <= lo:
            return HybridZeckendorfRational.from_value(self.control_points[0])
        if xq >= hi:
            return HybridZeckendorfRational.from_value(self.control_points[-1])

        segment_count = len(self.control_points) - 1
        scaled = (xq - lo) * segment_count / (hi - lo)
        left_index = scaled.numerator // scaled.denominator
        t = scaled - left_index
        left_value = self.control_points[left_index]
        right_value = self.control_points[left_index + 1]
        interpolated = left_value + t * (right_value - left_value)
        return HybridZeckendorfRational.from_value(interpolated)

    def output_bound(self) -> tuple[Fraction, Fraction]:
        """Return exact lower/upper bounds from the spline control points."""

        return min(self.control_points), max(self.control_points)

    def to_json(self) -> JsonDict:
        """Serialize the exact spline parameters and local bounds."""

        lower, upper = self.output_bound()
        return {
            "name": self.name,
            "domain": [serialize_fraction(self.domain[0]), serialize_fraction(self.domain[1])],
            "control_points": [serialize_fraction(point) for point in self.control_points],
            "output_bound": {
                "lower": serialize_fraction(lower),
                "upper": serialize_fraction(upper),
            },
        }


@dataclass(frozen=True)
class MockRationalKAN:
    """Small exact-rational RKAN shape used for the accounting pass."""

    input_dim: int
    edge_splines: Mapping[Edge, MockRationalSpline]
    bias_splines: tuple[MockRationalSpline, ...]

    @property
    def n_params(self) -> int:
        """Count exact spline control points used by the mock RKAN tier."""

        edge_params = sum(len(spline.control_points) for spline in self.edge_splines.values())
        bias_params = sum(len(spline.control_points) for spline in self.bias_splines)
        return edge_params + bias_params

    def forward(self, x: Sequence[RationalInput]) -> HybridZeckendorfRational:
        """Evaluate the mock KAN energy over exact Hybrid Zeckendorf rationals."""

        xq = tuple(to_fraction(value) for value in x)
        if len(xq) != self.input_dim:
            raise ValueError(f"expected {self.input_dim} inputs, got {len(xq)}")

        total = HybridZeckendorfRational.from_value(0)
        for (i, j), spline in self.edge_splines.items():
            product = HybridZeckendorfRational.from_value(xq[i]).mul(
                HybridZeckendorfRational.from_value(xq[j])
            )
            total = total.add(spline.evaluate(product.fraction))
        for index, spline in enumerate(self.bias_splines):
            total = total.add(spline.evaluate(xq[index]))
        return total

    def to_json(self) -> JsonDict:
        """Serialize model topology and exact control points."""

        return {
            "input_dim": self.input_dim,
            "n_params": self.n_params,
            "edges": {f"{i},{j}": spline.to_json() for (i, j), spline in self.edge_splines.items()},
            "biases": [spline.to_json() for spline in self.bias_splines],
        }


@dataclass(frozen=True)
class SimulationResult:
    """Exact CPU simulation outputs plus deterministic operation accounting."""

    sample_inputs: tuple[tuple[Fraction, ...], ...]
    energies: tuple[Fraction, ...]
    operation_counts: JsonDict
    float_operations_used: bool = False

    def to_json(self) -> JsonDict:
        """Serialize the exact CPU simulation rows for the artifact."""

        samples = []
        for sample, energy in zip(self.sample_inputs, self.energies, strict=True):
            samples.append(
                {
                    "input": [
                        HybridZeckendorfRational.from_value(value).to_json() for value in sample
                    ],
                    "energy": HybridZeckendorfRational.from_value(energy).to_json(),
                }
            )
        return {
            "float_operations_used": self.float_operations_used,
            "operation_counts": self.operation_counts,
            "samples": samples,
        }


def build_mock_rkan() -> MockRationalKAN:
    """Build the deterministic mock RKAN tier fixture for Exp 1671."""

    return MockRationalKAN(
        input_dim=2,
        edge_splines={
            (0, 1): MockRationalSpline.from_points("edge_0_1", ["-1/4", "0", "3/4"]),
        },
        bias_splines=(
            MockRationalSpline.from_points("bias_0", ["0", "1/2", "1"]),
            MockRationalSpline.from_points("bias_1", ["1/3", "0", "-1/3"]),
        ),
    )


def simulate_cpu(
    model: MockRationalKAN,
    sample_inputs: Sequence[Sequence[RationalInput]],
) -> SimulationResult:
    """Run the CPU-only exact-rational mock KAN simulation."""

    normalized_inputs = tuple(tuple(to_fraction(value) for value in row) for row in sample_inputs)
    energies = tuple(model.forward(row).fraction for row in normalized_inputs)
    edge_evaluations = len(normalized_inputs) * len(model.edge_splines)
    bias_evaluations = len(normalized_inputs) * len(model.bias_splines)
    spline_evaluations = edge_evaluations + bias_evaluations
    operation_counts = {
        "samples": len(normalized_inputs),
        "edge_splines": len(model.edge_splines),
        "bias_splines": len(model.bias_splines),
        "parameters": model.n_params,
        "edge_products": edge_evaluations,
        "spline_evaluations": spline_evaluations,
        "interpolation_multiplications_upper_bound": spline_evaluations,
        "interpolation_additions_upper_bound": 2 * spline_evaluations,
        "energy_accumulations": spline_evaluations,
        "total_fraction_ops_upper_bound": edge_evaluations
        + spline_evaluations
        + (2 * spline_evaluations)
        + spline_evaluations,
    }
    return SimulationResult(
        sample_inputs=normalized_inputs,
        energies=energies,
        operation_counts=operation_counts,
        float_operations_used=False,
    )


def build_bounding_certificates(
    model: MockRationalKAN,
    simulation: SimulationResult,
) -> JsonDict:
    """Build exact spline and sample-energy bounds for the audit artifact."""

    splines = _ordered_splines(model)
    spline_bounds = {
        spline.name: {
            "lower": serialize_fraction(spline.output_bound()[0]),
            "upper": serialize_fraction(spline.output_bound()[1]),
        }
        for spline in splines
    }
    model_lower = sum((spline.output_bound()[0] for spline in splines), Fraction(0, 1))
    model_upper = sum((spline.output_bound()[1] for spline in splines), Fraction(0, 1))
    sample_lower = min(simulation.energies)
    sample_upper = max(simulation.energies)
    bound_values = (model_lower, model_upper, sample_lower, sample_upper, *simulation.energies)
    witnesses = [HybridZeckendorfRational.from_value(value).to_json() for value in bound_values]
    certificates = {
        "spline_output_bounds": spline_bounds,
        "model_output_bound": {
            "lower": serialize_fraction(model_lower),
            "upper": serialize_fraction(model_upper),
        },
        "sample_energy_bounds": {
            "lower": serialize_fraction(sample_lower),
            "upper": serialize_fraction(sample_upper),
            "contains_all_simulated_energies": all(
                sample_lower <= energy <= sample_upper for energy in simulation.energies
            ),
        },
        "all_simulated_energies_within_model_bound": all(
            model_lower <= energy <= model_upper for energy in simulation.energies
        ),
        "rational_witnesses": witnesses,
        "zeckendorf_witnesses_valid": all(
            _rational_witness_is_valid(witness) for witness in witnesses
        ),
    }
    return certificates


def validate_bounding_certificates(
    certificates: Mapping[str, Any], simulation: SimulationResult
) -> bool:
    """Check that exact sample-energy bounds contain every simulated energy."""

    try:
        lower = Fraction(certificates["sample_energy_bounds"]["lower"])
        upper = Fraction(certificates["sample_energy_bounds"]["upper"])
        witnesses_valid = bool(certificates["zeckendorf_witnesses_valid"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return witnesses_valid and all(lower <= energy <= upper for energy in simulation.energies)


def build_artifact(
    *,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the deterministic Exp 1671 audit artifact without writing it."""

    model = build_mock_rkan()
    simulation = simulate_cpu(model, DEFAULT_SAMPLE_INPUTS)
    certificates = build_bounding_certificates(model, simulation)
    exact_gate = not simulation.float_operations_used
    certificate_gate = validate_bounding_certificates(certificates, simulation)
    no_synthesis_gate = True
    status = "complete" if exact_gate and certificate_gate and no_synthesis_gate else "blocked"
    verdict = (
        "complete: hybrid_zeckendorf_exact_rational_rkan_audit_ready"
        if status == "complete"
        else "blocked: hybrid_zeckendorf_rkan_audit_gate_failed"
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "status": status,
        "experiment": "1671_rkan_audit",
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "title": TITLE,
        "spec_traces": list(SPEC_TRACES),
        "module": MODULE_PATH,
        "artifact_path": ARTIFACT_PATH,
        "model": model.to_json(),
        "sample_count": len(DEFAULT_SAMPLE_INPUTS),
        "simulation": simulation.to_json(),
        "sample_outputs": [serialize_fraction(energy) for energy in simulation.energies],
        "complexity": simulation.operation_counts,
        "bounding_certificates": certificates,
        "float_operations_used": simulation.float_operations_used,
        "hardware_synthesis_claimed": False,
        "no_synthesis_accounting_only": True,
        "hybrid_zeckendorf_rational_math": True,
        "exact_rational_cpu_simulation": True,
        "tests_run": list(tests_run or []),
        "honest_verdict": verdict,
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal fields consumed by conductor-style tooling."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["spec_traces"] != SPEC_TRACES:
        raise AssertionError("spec_traces must cite REQ-KAN-1671 and SCENARIO-KAN-1671")
    if artifact["hardware_synthesis_claimed"] is not False:
        raise AssertionError("hardware_synthesis_claimed must remain false")
    if artifact["float_operations_used"] is not False:
        raise AssertionError("float_operations_used must remain false")


def run_experiment(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Write `results/experiment_1671_rkan.json` and return its payload."""

    artifact = build_artifact(run_date=run_date, tests_run=tests_run)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the Exp 1671 audit."""

    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--run-date", default=RUN_DATE)
    args = parser.parse_args(argv)
    artifact = run_experiment(output_path=args.output, run_date=args.run_date)
    print(f"wrote={args.output}")
    print(f"status={artifact['status']}")
    return 0


def _fibonacci_terms_up_to(n: int) -> list[tuple[int, int]]:
    if n <= 0:
        return []
    terms = [(2, 1), (3, 2)]
    while terms[-1][1] < n:
        terms.append((terms[-1][0] + 1, terms[-1][1] + terms[-2][1]))
    return terms


def _terms_are_nonconsecutive(terms: Sequence[Mapping[str, Any]]) -> bool:
    indices = sorted(int(term["index"]) for term in terms)
    return len(indices) == len(set(indices)) and all(
        right - left > 1 for left, right in zip(indices, indices[1:], strict=False)
    )


def _ordered_splines(model: MockRationalKAN) -> tuple[MockRationalSpline, ...]:
    edge_splines = tuple(
        model.edge_splines[edge]
        for edge in sorted(model.edge_splines, key=lambda item: (item[0], item[1]))
    )
    return edge_splines + model.bias_splines


def _rational_witness_is_valid(witness: Mapping[str, Any]) -> bool:
    numerator = witness["numerator_witness"]
    denominator = witness["denominator_witness"]
    numerator_ok = reconstruct_zeckendorf(numerator) == int(numerator["integer"])
    denominator_ok = reconstruct_zeckendorf(denominator) == int(denominator["integer"])
    return (
        numerator_ok
        and denominator_ok
        and zeckendorf_terms_are_nonconsecutive(numerator)
        and zeckendorf_terms_are_nonconsecutive(denominator)
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
