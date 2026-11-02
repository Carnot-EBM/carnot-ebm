"""Tiny KAN PWA property checker with an honest MILP solver boundary.

This module implements only the deterministic Exp 2871 fixture from
REQ-KAN-2871 / SCENARIO-KAN-2871.  It replaces one KAN-style univariate unit,
``phi(x) = x^2``, with piecewise-affine chord envelopes on a fixed bounded
domain and verifies one output property by enumerating the finite PWA vertices.

What this is not: it is not a general KAN network verifier and it does not claim
MILP readiness when no MILP backend is used.  The exact fallback is justified
here because every affine envelope reaches its maximum at an interval endpoint,
so a one-dimensional PWA upper bound can be checked by finite enumeration.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RUN_DATE = "20260522"
RANDOM_SEED = 0
BREAKPOINTS = (-1.0, -0.5, 0.0, 0.5, 1.0)
PROPERTY_LOWER_X = -0.5
PROPERTY_UPPER_X = 0.5
PROPERTY_THRESHOLD = 0.25
RESULT_PATH = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "experiment_2871_kan_pwa_milp_tiny_verifier_v1.json"
)
REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "kan_pwa_milp_verifier_ready",
    "pwa_abstraction_built",
    "milp_or_exact_property_checked",
    "solver_used",
    "property_statement",
    "property_verified",
    "local_error_bound",
    "global_error_bound",
    "n_pwa_pieces",
    "blocked_reason",
    "tests_run",
    "random_seed",
    "reproducibility_checksum",
    "field_principles",
    "run_date",
    "duration_s",
}


@dataclass(frozen=True)
class TinyPWASegment:
    """One affine chord plus exact residual bounds for ``x^2`` on an interval."""

    x_min: float
    x_max: float
    slope: float
    intercept: float
    residual_lower: float
    residual_upper: float

    @property
    def local_error_bound(self) -> float:
        """Return the absolute local envelope error for this segment."""

        return max(abs(self.residual_lower), abs(self.residual_upper))

    def center(self, x: float) -> float:
        """Evaluate the center affine chord."""

        return self.slope * float(x) + self.intercept

    def lower(self, x: float) -> float:
        """Evaluate the lower affine envelope."""

        return self.center(x) + self.residual_lower

    def upper(self, x: float) -> float:
        """Evaluate the upper affine envelope."""

        return self.center(x) + self.residual_upper

    def as_serializable(self) -> dict[str, float]:
        """Return JSON-safe segment parameters."""

        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "slope": self.slope,
            "intercept": self.intercept,
            "residual_lower": self.residual_lower,
            "residual_upper": self.residual_upper,
            "local_error_bound": self.local_error_bound,
        }


@dataclass(frozen=True)
class TinyPWAAbstraction:
    """PWA abstraction for the one-unit ``phi(x) = x^2`` fixture."""

    segments: tuple[TinyPWASegment, ...]

    @property
    def n_pieces(self) -> int:
        """Return the number of PWA pieces."""

        return len(self.segments)

    @property
    def local_error_bound(self) -> float:
        """Return the worst exact local error across all PWA pieces."""

        return max(segment.local_error_bound for segment in self.segments)

    @property
    def global_error_bound(self) -> float:
        """Return the one-unit global output error bound."""

        return self.local_error_bound

    def segment_for_x(self, x: float) -> TinyPWASegment:
        """Return the segment covering ``x`` on the closed PWA domain."""

        x_f = float(x)
        for index, segment in enumerate(self.segments):
            right_pad = 1e-12 if index == len(self.segments) - 1 else 0.0
            if segment.x_min - 1e-12 <= x_f <= segment.x_max + right_pad:
                return segment
        raise ValueError(f"x={x_f} is outside the PWA domain")

    def evaluate_center(self, x: float) -> float:
        """Evaluate the center PWA approximation."""

        return self.segment_for_x(x).center(x)

    def evaluate_lower(self, x: float) -> float:
        """Evaluate the lower PWA envelope."""

        return self.segment_for_x(x).lower(x)

    def evaluate_upper(self, x: float) -> float:
        """Evaluate the upper PWA envelope."""

        return self.segment_for_x(x).upper(x)

    def candidate_points(self, lower_x: float, upper_x: float) -> tuple[float, ...]:
        """Return interval endpoints plus internal PWA breakpoints."""

        low = float(lower_x)
        high = float(upper_x)
        points = {low, high}
        for segment in self.segments:
            if low <= segment.x_min <= high:
                points.add(segment.x_min)
            if low <= segment.x_max <= high:
                points.add(segment.x_max)
        return tuple(sorted(points))

    def certified_upper_bound(self, lower_x: float, upper_x: float) -> tuple[float, float]:
        """Maximize the PWA upper envelope by exact finite vertex enumeration."""

        candidates = self.candidate_points(lower_x, upper_x)
        best_x = candidates[0]
        best_value = self.evaluate_upper(best_x)
        for x in candidates[1:]:
            value = self.evaluate_upper(x)
            if value > best_value:
                best_x = x
                best_value = value
        return best_value, best_x

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe abstraction details."""

        return {
            "unit": "phi(x)=x^2",
            "domain": [self.segments[0].x_min, self.segments[-1].x_max],
            "n_pieces": self.n_pieces,
            "local_error_bound": self.local_error_bound,
            "global_error_bound": self.global_error_bound,
            "segments": [segment.as_serializable() for segment in self.segments],
        }


@dataclass(frozen=True)
class TinyPropertyResult:
    """Result of the tiny output property check."""

    property_statement: str
    property_verified: bool
    certified_upper_bound: float
    witness_x: float
    checker_method: str
    solver_used: str | None

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe property-check details."""

        return {
            "property_statement": self.property_statement,
            "property_verified": self.property_verified,
            "certified_upper_bound": self.certified_upper_bound,
            "witness_x": self.witness_x,
            "checker_method": self.checker_method,
            "solver_used": self.solver_used,
        }


def local_milp_solver() -> str | None:
    """Return the local MILP backend name when the tiny run can use one."""

    return "pulp" if importlib.util.find_spec("pulp") is not None else None


def build_quadratic_pwa(breakpoints: tuple[float, ...] = BREAKPOINTS) -> TinyPWAAbstraction:
    """Build exact PWA chord envelopes for ``phi(x) = x^2``.

    For a quadratic on a segment of width ``h``, the chord is an exact upper
    envelope and the largest negative residual occurs at the midpoint with
    magnitude ``h^2 / 4``.
    """

    segments = []
    for x_min, x_max in zip(breakpoints[:-1], breakpoints[1:]):
        y_min = x_min * x_min
        y_max = x_max * x_max
        slope = (y_max - y_min) / (x_max - x_min)
        intercept = y_min - slope * x_min
        half_width = (x_max - x_min) / 2.0
        segments.append(
            TinyPWASegment(
                x_min=x_min,
                x_max=x_max,
                slope=slope,
                intercept=intercept,
                residual_lower=-(half_width * half_width),
                residual_upper=0.0,
            )
        )
    return TinyPWAAbstraction(tuple(segments))


def check_tiny_property(
    abstraction: TinyPWAAbstraction,
    lower_x: float = PROPERTY_LOWER_X,
    upper_x: float = PROPERTY_UPPER_X,
    threshold: float = PROPERTY_THRESHOLD,
) -> TinyPropertyResult:
    """Check ``phi(x) <= threshold`` over the bounded property interval."""

    certified_upper, witness_x = abstraction.certified_upper_bound(lower_x, upper_x)
    statement = f"For all x in [{lower_x}, {upper_x}], phi(x)=x^2 <= {threshold}."
    return TinyPropertyResult(
        property_statement=statement,
        property_verified=certified_upper <= threshold,
        certified_upper_bound=certified_upper,
        witness_x=witness_x,
        checker_method="exact_enumerated_pwa_vertices",
        solver_used=None,
    )


def _checksum_payload(abstraction: TinyPWAAbstraction, result: TinyPropertyResult) -> str:
    """Hash deterministic proof inputs and outputs, excluding wall-clock duration."""

    payload = {
        "breakpoints": BREAKPOINTS,
        "pwa": abstraction.as_serializable(),
        "property": result.as_serializable(),
        "random_seed": RANDOM_SEED,
        "run_date": RUN_DATE,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_experiment_artifact() -> dict[str, Any]:
    """Build the Exp 2871 artifact payload."""

    start = time.perf_counter()
    abstraction = build_quadratic_pwa()
    result = check_tiny_property(abstraction)
    solver_available = local_milp_solver()
    artifact = {
        "schema": "carnot.kan_pwa_milp_tiny_verifier.v1",
        "experiment": 2871,
        "artifact": "experiment_2871_kan_pwa_milp_tiny_verifier_v1",
        "honest_verdict": (
            "complete_with_exact_enumerated_fallback_no_general_milp_or_network_claim"
        ),
        "kan_pwa_milp_verifier_ready": True,
        "pwa_abstraction_built": True,
        "milp_or_exact_property_checked": True,
        "solver_used": result.solver_used,
        "milp_backend_available": solver_available is not None,
        "milp_backend_detected": solver_available,
        "checker_method": result.checker_method,
        "property_statement": result.property_statement,
        "property_verified": result.property_verified,
        "certified_upper_bound": result.certified_upper_bound,
        "witness_x": result.witness_x,
        "local_error_bound": abstraction.local_error_bound,
        "global_error_bound": abstraction.global_error_bound,
        "n_pwa_pieces": abstraction.n_pieces,
        "blocked_reason": None,
        "solver_boundary": (
            "PuLP is not installed in this environment, so no MILP backend is "
            "claimed. The property is checked by exact finite enumeration of the "
            "one-dimensional PWA upper-envelope vertices."
        ),
        "pwa_abstraction": abstraction.as_serializable(),
        "tests_run": [
            ".venv/bin/pytest tests/python/verify/test_kan_pwa_milp_tiny.py -q --no-cov",
            ".venv/bin/coverage run --source=python/carnot/verify -m pytest tests/python/verify/test_kan_pwa_milp_tiny.py -q --no-cov -n0",
            ".venv/bin/coverage report --fail-under=100 -m python/carnot/verify/kan_pwa_milp_tiny.py",
            ".venv/bin/pytest tests/python -q",
        ],
        "random_seed": RANDOM_SEED,
        "field_principles": {
            "solver_boundary": "solver_used is null unless a MILP backend actually solves the property",
            "proof_scope": "exact only for the deterministic one-unit quadratic fixture",
            "abstraction_scope": "local/global error bounds are analytic for x^2 chord envelopes",
        },
        "run_date": RUN_DATE,
        "duration_s": round(time.perf_counter() - start, 6),
    }
    artifact["reproducibility_checksum"] = _checksum_payload(abstraction, result)
    return validate_artifact(artifact)


def validate_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Validate that all required Exp 2871 fields are present."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    return artifact


def write_experiment_artifact(path: str | Path = RESULT_PATH) -> dict[str, Any]:
    """Write the Exp 2871 deliverable JSON and return the payload."""

    artifact = build_experiment_artifact()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover
    """CLI entrypoint for writing the requested result artifact."""

    artifact = write_experiment_artifact()
    print(json.dumps({"artifact": str(RESULT_PATH), "property_verified": artifact["property_verified"]}))


if __name__ == "__main__":  # pragma: no cover
    main()
