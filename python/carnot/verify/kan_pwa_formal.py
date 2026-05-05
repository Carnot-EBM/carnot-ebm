"""PWA abstraction and CPU-only energy-bound verification for small GS-KAN layers.

The verifier in this module is intentionally narrow. It handles the current
`GSKANEnergy` layer, whose shared spline functions are degree-1 B-splines
implemented by linear interpolation over fixed knots. For that model class, a
knot-aligned piecewise-affine (PWA) abstraction is exact up to floating-point
roundoff, so a simple input-box energy bound can be solved as independent
one-dimensional LPs rather than a full MILP.

This is still useful as formal-software scaffolding: it records the abstraction
error, adds that error to the certified upper bound, and refuses to mark a
property verified unless the certified upper bound is strictly below the tested
threshold. It makes no hardware-correctness claim.

Spec: REQ-VERIFY-1372, SCENARIO-VERIFY-1372
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PWASegment:
    """One affine piece `slope * x + intercept` over a closed input interval."""

    lower: float
    upper: float
    slope: float
    intercept: float
    max_abs_error: float

    def evaluate(self, x: float) -> float:
        """Evaluate this affine piece at `x` without re-checking interval membership."""
        return float(self.slope * x + self.intercept)


@dataclass(frozen=True)
class SplinePWAAbstraction:
    """PWA replacement for one shared GS-KAN parent spline."""

    group_index: int
    segments: tuple[PWASegment, ...]
    max_abs_error: float

    def evaluate(self, x: float) -> float:
        """Evaluate the PWA spline, selecting the segment that contains `x`.

        The input domain is closed, so a knot shared by two adjacent pieces is
        accepted by either piece. The last piece owns the right endpoint to keep
        `x=1` representable without a special sentinel interval.
        """
        x_f = float(x)
        for index, segment in enumerate(self.segments):
            is_last = index == len(self.segments) - 1
            if segment.lower - 1e-12 <= x_f <= segment.upper + (1e-12 if is_last else 0.0):
                return segment.evaluate(x_f)
        raise ValueError(f"x={x_f} is outside the PWA spline domain")

    def max_weighted_value(self, lower: float, upper: float, weight: float) -> tuple[float, float]:
        """Solve the one-variable LP for `weight * spline(x)` on `[lower, upper]`.

        Each overlapping segment is affine, so its maximum lies at one endpoint
        of the overlap. Checking those affine endpoint optima is equivalent to
        solving the per-segment LPs and taking the best feasible segment.
        """
        best_value = -float("inf")
        best_x = float(lower)
        for segment in self.segments:
            overlap_lower = max(float(lower), segment.lower)
            overlap_upper = min(float(upper), segment.upper)
            if overlap_lower > overlap_upper + 1e-12:
                continue

            weighted_slope = float(weight) * segment.slope
            candidate_x = overlap_upper if weighted_slope >= 0.0 else overlap_lower
            candidate_value = float(weight) * segment.evaluate(candidate_x)
            if candidate_value > best_value:
                best_value = candidate_value
                best_x = candidate_x

        if not np.isfinite(best_value):
            raise ValueError(
                f"input interval [{lower}, {upper}] does not overlap the spline domain"
            )
        return best_value, best_x


@dataclass(frozen=True)
class GSKANPWAAbstraction:
    """PWA abstraction for every shared parent spline in one `GSKANEnergy` layer."""

    n_vars: int
    n_groups: int
    n_knots: int
    pwa_segments_per_spline: int
    splines: tuple[SplinePWAAbstraction, ...]
    max_abs_error: float

    @property
    def spline_count(self) -> int:
        """Return the number of shared parent splines abstracted."""
        return len(self.splines)

    def evaluate_energy(self, x: Sequence[float], projection_weights: Sequence[float]) -> float:
        """Evaluate the abstracted one-layer GS-KAN energy at `x`."""
        if len(x) != self.n_vars:
            raise ValueError(f"x must have length {self.n_vars}, got {len(x)}")
        if len(projection_weights) != self.n_vars:
            raise ValueError(
                f"projection_weights must have length {self.n_vars}, got {len(projection_weights)}"
            )

        total = 0.0
        for var_index, x_i in enumerate(x):
            group_index = var_index % self.n_groups
            total += float(projection_weights[var_index]) * self.splines[group_index].evaluate(
                float(x_i)
            )
        return float(total)


@dataclass(frozen=True)
class ManualLPResult:
    """Certified upper bound from the manual LP fallback."""

    solver_name: str
    solver_status: str
    integer_constraints_needed: bool
    exact_pwa_upper_bound: float
    abstraction_error_budget: float
    certified_upper_bound: float
    maximizer: tuple[float, ...]


@dataclass(frozen=True)
class IntervalArithmeticBound:
    """Interval-arithmetic lower and upper energy bounds."""

    lower_bound: float
    upper_bound: float


@dataclass(frozen=True)
class EnergyBoundVerification:
    """Outcome of checking `energy(x) < threshold` over a bounded input box."""

    result: str
    formal_property_verified: bool
    threshold: float
    pwa_abstraction: GSKANPWAAbstraction
    lp_result: ManualLPResult
    interval_bound: IntervalArithmeticBound


def _eval_group(model: object, group_index: int, xs: np.ndarray) -> np.ndarray:
    """Evaluate a GS-KAN group spline through the model's implementation."""
    return np.asarray(model._eval_spline_group(group_index, xs), dtype=np.float64)


def _segment_breakpoints(model: object, pwa_segments_per_spline: int) -> np.ndarray:
    """Return breakpoints for the PWA abstraction.

    When the requested segment count matches the native GS-KAN knot interval
    count, the original knots are reused. That makes the abstraction exact for
    the current degree-1 spline implementation.
    """
    native_segment_count = int(model.n_knots) - 1
    if pwa_segments_per_spline == native_segment_count:
        return np.asarray(model._knots, dtype=np.float64)
    return np.linspace(-1.0, 1.0, pwa_segments_per_spline + 1, dtype=np.float64)


def build_gskan_pwa_abstraction(
    model: object,
    pwa_segments_per_spline: int | None = None,
    error_grid_points: int = 257,
) -> GSKANPWAAbstraction:
    """Build a PWA abstraction for every shared spline in a `GSKANEnergy` layer.

    The abstraction uses straight lines between selected breakpoints and then
    measures the largest absolute difference against the original spline on a
    dense grid inside each segment. The error is later propagated into the LP
    upper bound, so verification remains conservative even when callers choose
    fewer pieces than the native knot intervals.
    """
    required_attrs = ("n_vars", "n_groups", "n_knots", "_knots", "_eval_spline_group")
    missing = [name for name in required_attrs if not hasattr(model, name)]
    if missing:
        raise TypeError(f"model does not look like GSKANEnergy; missing {missing}")

    segment_count = int(pwa_segments_per_spline or (int(model.n_knots) - 1))
    if segment_count < 1:
        raise ValueError("pwa_segments_per_spline must be >= 1")
    if error_grid_points < 2:
        raise ValueError("error_grid_points must be >= 2")

    breakpoints = _segment_breakpoints(model, segment_count)
    splines: list[SplinePWAAbstraction] = []

    for group_index in range(int(model.n_groups)):
        segments: list[PWASegment] = []
        group_error = 0.0
        for lower, upper in zip(breakpoints[:-1], breakpoints[1:]):
            endpoints = np.asarray([lower, upper], dtype=np.float64)
            y0, y1 = _eval_group(model, group_index, endpoints)
            slope = float((y1 - y0) / (upper - lower))
            intercept = float(y0 - slope * lower)

            sample_xs = np.linspace(float(lower), float(upper), error_grid_points)
            original = _eval_group(model, group_index, sample_xs)
            approximate = slope * sample_xs + intercept
            max_abs_error = float(np.max(np.abs(original - approximate)))
            group_error = max(group_error, max_abs_error)

            segments.append(
                PWASegment(
                    lower=float(lower),
                    upper=float(upper),
                    slope=slope,
                    intercept=intercept,
                    max_abs_error=max_abs_error,
                )
            )

        splines.append(
            SplinePWAAbstraction(
                group_index=group_index,
                segments=tuple(segments),
                max_abs_error=float(group_error),
            )
        )

    return GSKANPWAAbstraction(
        n_vars=int(model.n_vars),
        n_groups=int(model.n_groups),
        n_knots=int(model.n_knots),
        pwa_segments_per_spline=segment_count,
        splines=tuple(splines),
        max_abs_error=float(max(spline.max_abs_error for spline in splines)),
    )


def _validate_input_bounds(
    input_bounds: Sequence[tuple[float, float]], n_vars: int
) -> tuple[tuple[float, float], ...]:
    """Validate an input box over the native GS-KAN `[-1, 1]` domain."""
    if len(input_bounds) != n_vars:
        raise ValueError(f"input_bounds must contain {n_vars} intervals, got {len(input_bounds)}")

    checked = []
    for index, (lower, upper) in enumerate(input_bounds):
        lower_f = float(lower)
        upper_f = float(upper)
        if lower_f > upper_f:
            raise ValueError(f"input_bounds[{index}] has lower > upper")
        if lower_f < -1.0 - 1e-12 or upper_f > 1.0 + 1e-12:
            raise ValueError(
                "GS-KAN PWA verification currently supports only bounds inside [-1, 1]"
            )
        checked.append((max(lower_f, -1.0), min(upper_f, 1.0)))
    return tuple(checked)


def maximize_energy_manual_lp(
    abstraction: GSKANPWAAbstraction,
    projection_weights: Sequence[float],
    input_bounds: Sequence[tuple[float, float]],
    include_abstraction_error: bool = True,
) -> ManualLPResult:
    """Maximize abstracted GS-KAN energy over an input box.

    A single GS-KAN layer is separable across input variables after the shared
    spline for each variable is selected. For each variable we solve the affine
    endpoint LP on every overlapping PWA segment, keep that variable's largest
    contribution, and sum the independent optima. No integer variables are
    needed for this one-layer input-box property.
    """
    if len(projection_weights) != abstraction.n_vars:
        raise ValueError(
            f"projection_weights must contain {abstraction.n_vars} values, "
            f"got {len(projection_weights)}"
        )
    checked_bounds = _validate_input_bounds(input_bounds, abstraction.n_vars)

    exact_upper = 0.0
    error_budget = 0.0
    maximizer: list[float] = []

    for var_index, (lower, upper) in enumerate(checked_bounds):
        group_index = var_index % abstraction.n_groups
        weight = float(projection_weights[var_index])
        spline = abstraction.splines[group_index]
        best_value, best_x = spline.max_weighted_value(lower, upper, weight)
        exact_upper += best_value
        maximizer.append(best_x)
        if include_abstraction_error:
            error_budget += abs(weight) * spline.max_abs_error

    certified_upper = exact_upper + error_budget
    return ManualLPResult(
        solver_name="manual_lp_fallback_per_variable_pwa_endpoint_solver",
        solver_status="optimal",
        integer_constraints_needed=False,
        exact_pwa_upper_bound=float(exact_upper),
        abstraction_error_budget=float(error_budget),
        certified_upper_bound=float(certified_upper),
        maximizer=tuple(float(v) for v in maximizer),
    )


def interval_arithmetic_energy_bound(
    model: object, input_bounds: Sequence[tuple[float, float]]
) -> IntervalArithmeticBound:
    """Compute a direct interval-arithmetic baseline for the GS-KAN energy.

    The baseline evaluates each true spline at the input interval endpoints and
    any native knots inside the interval. For the current PWA GS-KAN layer this
    is tight, but it is still useful as a simple baseline for future nonlinear
    spline variants.
    """
    checked_bounds = _validate_input_bounds(input_bounds, int(model.n_vars))
    knots = np.asarray(model._knots, dtype=np.float64)
    weights = np.asarray(model.proj_weights, dtype=np.float64)

    lower_total = 0.0
    upper_total = 0.0
    for var_index, (lower, upper) in enumerate(checked_bounds):
        group_index = var_index % int(model.n_groups)
        interior = knots[(knots > lower) & (knots < upper)]
        candidates = np.asarray([lower, *interior.tolist(), upper], dtype=np.float64)
        values = _eval_group(model, group_index, candidates)
        contributions = weights[var_index] * values
        lower_total += float(np.min(contributions))
        upper_total += float(np.max(contributions))

    return IntervalArithmeticBound(lower_bound=float(lower_total), upper_bound=float(upper_total))


def verify_energy_bound(
    model: object,
    input_bounds: Sequence[tuple[float, float]],
    threshold: float,
    pwa_segments_per_spline: int | None = None,
    error_grid_points: int = 257,
) -> EnergyBoundVerification:
    """Verify `GSKANEnergy.energy(x) < threshold` over a bounded input box."""
    abstraction = build_gskan_pwa_abstraction(
        model,
        pwa_segments_per_spline=pwa_segments_per_spline,
        error_grid_points=error_grid_points,
    )
    lp_result = maximize_energy_manual_lp(abstraction, model.proj_weights, input_bounds)
    interval_bound = interval_arithmetic_energy_bound(model, input_bounds)

    threshold_f = float(threshold)
    formal_property_verified = lp_result.certified_upper_bound < threshold_f
    if formal_property_verified:
        result = "verified"
    elif lp_result.exact_pwa_upper_bound >= threshold_f:
        result = "counterexample"
    else:
        result = "not_verified"

    return EnergyBoundVerification(
        result=result,
        formal_property_verified=bool(formal_property_verified),
        threshold=threshold_f,
        pwa_abstraction=abstraction,
        lp_result=lp_result,
        interval_bound=interval_bound,
    )


__all__ = [
    "EnergyBoundVerification",
    "GSKANPWAAbstraction",
    "IntervalArithmeticBound",
    "ManualLPResult",
    "PWASegment",
    "SplinePWAAbstraction",
    "build_gskan_pwa_abstraction",
    "interval_arithmetic_energy_bound",
    "maximize_energy_manual_lp",
    "verify_energy_bound",
]
