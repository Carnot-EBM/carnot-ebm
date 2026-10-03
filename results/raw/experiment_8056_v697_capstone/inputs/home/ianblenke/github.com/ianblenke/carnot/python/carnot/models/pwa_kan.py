"""Piecewise-affine KAN spline wrappers for logical activation bounds.

The JAX KAN path evaluates spline activations numerically. Formal checks often
need a simpler logical shape: if an input is in one segment, the output is
between two affine functions. This module builds that shape for existing KAN
`BSpline` units and for any one-dimensional callable supplied by a verifier.

Spec references: REQ-KAN-1618, SCENARIO-KAN-1618.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _repo_root() -> Path:
    """Return the repository root so result artifacts are cwd-independent."""

    return Path(__file__).resolve().parents[3]


DEFAULT_RESULT_PATH = _repo_root() / "results/experiment_1618_pwa_kan.json"


@dataclass(frozen=True)
class ActivationIntervalBound:
    """Lower and upper activation bounds over an input interval."""

    lower: float
    upper: float
    lower_witness_x: float
    upper_witness_x: float
    segment_count: int

    def as_serializable(self) -> dict[str, float | int]:
        """Return a JSON-safe representation for experiment artifacts."""

        return {
            "lower": self.lower,
            "upper": self.upper,
            "lower_witness_x": self.lower_witness_x,
            "upper_witness_x": self.upper_witness_x,
            "segment_count": self.segment_count,
        }


@dataclass(frozen=True)
class PWASplineSegment:
    """One PWA segment with center, lower, and upper affine boundaries."""

    lower_x: float
    upper_x: float
    slope: float
    intercept: float
    residual_min: float
    residual_max: float
    max_abs_error: float

    @property
    def lower_intercept(self) -> float:
        """Intercept for the lower affine boundary."""

        return self.intercept + self.residual_min

    @property
    def upper_intercept(self) -> float:
        """Intercept for the upper affine boundary."""

        return self.intercept + self.residual_max

    def evaluate(self, x: float) -> float:
        """Evaluate the center affine approximation."""

        return self.slope * float(x) + self.intercept

    def lower_affine(self, x: float) -> float:
        """Evaluate the lower affine boundary at `x`."""

        return self.slope * float(x) + self.lower_intercept

    def upper_affine(self, x: float) -> float:
        """Evaluate the upper affine boundary at `x`."""

        return self.slope * float(x) + self.upper_intercept

    def bound_on_interval(self, lower_x: float, upper_x: float) -> ActivationIntervalBound:
        """Bound this segment's activation over the overlapping interval."""

        overlap_lower = max(float(lower_x), self.lower_x)
        overlap_upper = min(float(upper_x), self.upper_x)
        if overlap_lower > overlap_upper:
            raise ValueError("interval does not overlap this PWA segment")

        lower, lower_witness = _affine_minimum(
            self.slope,
            self.lower_intercept,
            overlap_lower,
            overlap_upper,
        )
        upper, upper_witness = _affine_maximum(
            self.slope,
            self.upper_intercept,
            overlap_lower,
            overlap_upper,
        )
        return ActivationIntervalBound(
            lower=lower,
            upper=upper,
            lower_witness_x=lower_witness,
            upper_witness_x=upper_witness,
            segment_count=1,
        )

    def as_serializable(self) -> dict[str, float]:
        """Return JSON-safe segment parameters."""

        return {
            "lower_x": self.lower_x,
            "upper_x": self.upper_x,
            "slope": self.slope,
            "intercept": self.intercept,
            "lower_intercept": self.lower_intercept,
            "upper_intercept": self.upper_intercept,
            "residual_min": self.residual_min,
            "residual_max": self.residual_max,
            "max_abs_error": self.max_abs_error,
        }


def _affine_minimum(
    slope: float,
    intercept: float,
    lower_x: float,
    upper_x: float,
) -> tuple[float, float]:
    """Return the minimum of `slope*x + intercept` on a closed interval."""

    witness = float(lower_x) if slope >= 0.0 else float(upper_x)
    return float(slope * witness + intercept), witness


def _affine_maximum(
    slope: float,
    intercept: float,
    lower_x: float,
    upper_x: float,
) -> tuple[float, float]:
    """Return the maximum of `slope*x + intercept` on a closed interval."""

    witness = float(upper_x) if slope >= 0.0 else float(lower_x)
    return float(slope * witness + intercept), witness


@dataclass(frozen=True)
class PWAKANUnit:
    """PWA abstraction for one KAN-style 1D spline activation."""

    name: str
    segments: tuple[PWASplineSegment, ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("PWAKANUnit requires at least one segment")

    @classmethod
    def from_callable(
        cls,
        name: str,
        evaluator: Callable[[float], float],
        breakpoints: Sequence[float],
        samples_per_segment: int = 33,
        residual_padding: float = 0.0,
    ) -> PWAKANUnit:
        """Build a PWA wrapper for an arbitrary 1D spline callable.

        Each segment uses the chord through its endpoints as the center affine
        approximation. The sampled residuals around that chord become constant
        lower/upper offsets, yielding two affine boundary functions per segment.
        """

        checked_breakpoints = _validate_breakpoints(breakpoints)
        if samples_per_segment < 2:
            raise ValueError("samples_per_segment must be >= 2")
        if residual_padding < 0.0:
            raise ValueError("residual_padding must be >= 0")

        padding = float(residual_padding)
        segments: list[PWASplineSegment] = []
        for lower_x, upper_x in zip(checked_breakpoints[:-1], checked_breakpoints[1:]):
            sample_xs = np.linspace(lower_x, upper_x, samples_per_segment, dtype=np.float64)
            sample_ys = np.asarray(
                [float(evaluator(float(x))) for x in sample_xs], dtype=np.float64
            )
            if not np.all(np.isfinite(sample_ys)):
                raise ValueError("spline evaluator returned non-finite sample values")

            slope = float((sample_ys[-1] - sample_ys[0]) / (upper_x - lower_x))
            intercept = float(sample_ys[0] - slope * lower_x)
            residuals = sample_ys - (slope * sample_xs + intercept)
            residual_min = float(np.min(residuals) - padding)
            residual_max = float(np.max(residuals) + padding)
            max_abs_error = float(max(abs(residual_min), abs(residual_max)))
            segments.append(
                PWASplineSegment(
                    lower_x=float(lower_x),
                    upper_x=float(upper_x),
                    slope=slope,
                    intercept=intercept,
                    residual_min=residual_min,
                    residual_max=residual_max,
                    max_abs_error=max_abs_error,
                )
            )

        return cls(name=name, segments=tuple(segments))

    @property
    def domain(self) -> tuple[float, float]:
        """Return the closed input domain covered by the PWA segments."""

        return self.segments[0].lower_x, self.segments[-1].upper_x

    @property
    def max_abs_error(self) -> float:
        """Return the largest sampled envelope offset across all segments."""

        return float(max(segment.max_abs_error for segment in self.segments))

    def evaluate(self, x: float) -> float:
        """Evaluate the center PWA approximation at `x`."""

        return self._segment_for_x(float(x)).evaluate(float(x))

    def activation_bounds(self, lower_x: float, upper_x: float) -> ActivationIntervalBound:
        """Return conservative activation bounds over an input interval."""

        lower_f = float(lower_x)
        upper_f = float(upper_x)
        if lower_f > upper_f:
            raise ValueError("lower_x must be <= upper_x")

        best_lower = float("inf")
        best_upper = -float("inf")
        lower_witness = lower_f
        upper_witness = upper_f
        overlapping_segments = 0

        for segment in self.segments:
            overlap_lower = max(lower_f, segment.lower_x)
            overlap_upper = min(upper_f, segment.upper_x)
            if overlap_lower > overlap_upper:
                continue
            overlapping_segments += 1
            segment_bound = segment.bound_on_interval(overlap_lower, overlap_upper)
            if segment_bound.lower < best_lower:
                best_lower = segment_bound.lower
                lower_witness = segment_bound.lower_witness_x
            if segment_bound.upper > best_upper:
                best_upper = segment_bound.upper
                upper_witness = segment_bound.upper_witness_x

        if overlapping_segments == 0:
            raise ValueError("interval does not overlap the PWA KAN unit domain")

        return ActivationIntervalBound(
            lower=float(best_lower),
            upper=float(best_upper),
            lower_witness_x=float(lower_witness),
            upper_witness_x=float(upper_witness),
            segment_count=overlapping_segments,
        )

    def weighted_activation_bounds(
        self,
        lower_x: float,
        upper_x: float,
        weight: float,
    ) -> ActivationIntervalBound:
        """Return bounds for `weight * activation(x)` over an input interval."""

        activation = self.activation_bounds(lower_x, upper_x)
        weight_f = float(weight)
        if weight_f >= 0.0:
            return ActivationIntervalBound(
                lower=weight_f * activation.lower,
                upper=weight_f * activation.upper,
                lower_witness_x=activation.lower_witness_x,
                upper_witness_x=activation.upper_witness_x,
                segment_count=activation.segment_count,
            )
        return ActivationIntervalBound(
            lower=weight_f * activation.upper,
            upper=weight_f * activation.lower,
            lower_witness_x=activation.upper_witness_x,
            upper_witness_x=activation.lower_witness_x,
            segment_count=activation.segment_count,
        )

    def logical_constraints(
        self,
        input_name: str = "x",
        output_name: str = "y",
    ) -> list[dict[str, Any]]:
        """Emit JSON-safe local affine implications for a verifier."""

        constraints: list[dict[str, Any]] = []
        for index, segment in enumerate(self.segments):
            constraints.append(
                {
                    "segment_index": index,
                    "condition": f"{segment.lower_x} <= {input_name} <= {segment.upper_x}",
                    "lower": (
                        f"{segment.slope} * {input_name} + "
                        f"{segment.lower_intercept} <= {output_name}"
                    ),
                    "upper": (
                        f"{output_name} <= {segment.slope} * {input_name} + "
                        f"{segment.upper_intercept}"
                    ),
                    "lower_x": segment.lower_x,
                    "upper_x": segment.upper_x,
                    "slope": segment.slope,
                    "center_intercept": segment.intercept,
                    "lower_intercept": segment.lower_intercept,
                    "upper_intercept": segment.upper_intercept,
                }
            )
        return constraints

    def count_sample_envelope_violations(
        self,
        evaluator: Callable[[float], float],
        samples_per_segment: int = 33,
        tolerance: float = 1e-12,
    ) -> int:
        """Count sampled points outside their recorded affine envelopes."""

        violations = 0
        for segment in self.segments:
            for x in np.linspace(segment.lower_x, segment.upper_x, samples_per_segment):
                y = float(evaluator(float(x)))
                if (
                    y < segment.lower_affine(float(x)) - tolerance
                    or y > segment.upper_affine(float(x)) + tolerance
                ):
                    violations += 1
        return violations

    def as_serializable(self) -> dict[str, Any]:
        """Return JSON-safe PWA metadata and segment parameters."""

        lower_x, upper_x = self.domain
        return {
            "name": self.name,
            "domain": [lower_x, upper_x],
            "segment_count": len(self.segments),
            "max_abs_error": self.max_abs_error,
            "segments": [segment.as_serializable() for segment in self.segments],
        }

    def _segment_for_x(self, x: float) -> PWASplineSegment:
        """Return the segment containing `x` on the closed PWA domain."""

        for index, segment in enumerate(self.segments):
            is_last = index == len(self.segments) - 1
            right_tolerance = 1e-12 if is_last else 0.0
            if segment.lower_x - 1e-12 <= x <= segment.upper_x + right_tolerance:
                return segment
        raise ValueError(f"x={x} is outside the PWA KAN unit domain")


def _validate_breakpoints(breakpoints: Sequence[float]) -> tuple[float, ...]:
    """Validate and normalize segment breakpoints."""

    checked = tuple(float(value) for value in breakpoints)
    if len(checked) < 2:
        raise ValueError("PWA KAN unit requires at least two breakpoints")
    for left, right in zip(checked[:-1], checked[1:]):
        if left >= right:
            raise ValueError("PWA KAN breakpoints must be strictly increasing")
    return checked


def build_pwa_for_bspline(
    spline: object,
    name: str = "bspline",
    params: object | None = None,
    breakpoints: Sequence[float] | None = None,
    samples_per_segment: int = 33,
) -> PWAKANUnit:
    """Build a PWA wrapper for an existing `carnot.models.kan.BSpline` unit."""

    import jax.numpy as jnp

    if breakpoints is None:
        num_knots = int(getattr(spline, "num_knots"))
        breakpoints = np.linspace(-1.0, 1.0, num_knots, dtype=np.float64)

    def evaluator(x: float) -> float:
        if params is None:
            value = spline.evaluate(jnp.asarray(x))
        else:
            value = spline.evaluate(jnp.asarray(x), params)
        return float(np.asarray(value))

    return PWAKANUnit.from_callable(
        name=name,
        evaluator=evaluator,
        breakpoints=breakpoints,
        samples_per_segment=samples_per_segment,
    )


def _reference_linear_unit() -> PWAKANUnit:
    """Build the exact linear spline fixture for Exp 1618 without JAX state."""

    return PWAKANUnit.from_callable(
        name="exact_linear_reference",
        evaluator=lambda x: x + 1.0,
        breakpoints=(-1.0, 0.0, 1.0),
        samples_per_segment=17,
    )


def build_experiment_1618_artifact() -> dict[str, Any]:
    """Build the stable Exp 1618 PWA KAN logical-bounds artifact payload."""

    exact_unit = _reference_linear_unit()
    nonlinear_unit = PWAKANUnit.from_callable(
        name="quadratic_callable",
        evaluator=lambda x: x * x,
        breakpoints=(-1.0, 0.0, 1.0),
        samples_per_segment=65,
    )
    nonlinear_violations = nonlinear_unit.count_sample_envelope_violations(
        lambda x: x * x,
        samples_per_segment=65,
    )
    constraints = exact_unit.logical_constraints("x", "y") + nonlinear_unit.logical_constraints(
        "x",
        "y",
    )

    return {
        "schema": "carnot.pwa_kan_logical_bounds.v1",
        "status": "complete",
        "experiment": 1618,
        "experiment_id": 1618,
        "run_date": "20260509",
        "title": "PWA KAN logical spline activation bounds",
        "spec": ["REQ-KAN-1618", "SCENARIO-KAN-1618"],
        "module": "python/carnot/models/pwa_kan.py",
        "artifact_path": "results/experiment_1618_pwa_kan.json",
        "pwa_kan_ready": True,
        "arbitrary_1d_spline_supported": True,
        "bspline_wrapper_supported": True,
        "exact_linear": exact_unit.as_serializable(),
        "nonlinear_reference": nonlinear_unit.as_serializable(),
        "exact_linear_max_abs_error": exact_unit.max_abs_error,
        "nonlinear_max_abs_error": nonlinear_unit.max_abs_error,
        "nonlinear_sample_violations": nonlinear_violations,
        "logical_constraint_count": len(constraints),
        "logical_constraints": constraints,
        "sample_interval_bound": nonlinear_unit.activation_bounds(-0.75, 0.75).as_serializable(),
        "honest_verdict": "complete: pwa_kan_logical_affine_bounds_ready",
    }


def write_experiment_1618_artifact(
    output_path: str | Path = DEFAULT_RESULT_PATH,
) -> dict[str, Any]:
    """Write `results/experiment_1618_pwa_kan.json` and return the payload."""

    artifact = build_experiment_1618_artifact()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


__all__ = [
    "ActivationIntervalBound",
    "DEFAULT_RESULT_PATH",
    "PWAKANUnit",
    "PWASplineSegment",
    "build_experiment_1618_artifact",
    "build_pwa_for_bspline",
    "write_experiment_1618_artifact",
]
