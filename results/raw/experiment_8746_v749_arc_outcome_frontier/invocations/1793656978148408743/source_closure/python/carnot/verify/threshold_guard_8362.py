"""REQ-VERIFY-8362: analytic spline bounds do not certify an undocumented sigmoid."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as F
from functools import lru_cache
import json
import math
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline, PPoly
from scipy.special import expit

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import spline_table_fidelity_8352 as old

Json = dict[str, Any]
Array: TypeAlias = NDArray[Any]
action = old.kernel.action
VERSION = "8362-rational-cells-v1"
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush measured counts so bounded numeric work remains visible."""
    print(f"[exp8362] phase={phase} completed={completed} pending={pending}", flush=True)


def validate(head: Json, x: Array | None = None) -> None:
    """Reject undefined policies before array indexing or direct evaluation."""
    c: Array = np.asarray(head["coefficients"], dtype=np.float64)
    if c.shape != (34,) or not np.isfinite(c).all() or np.max(np.abs(c)) > 4:
        raise ValueError("coefficients")
    if not math.isfinite(head["temperature"]) or head["temperature"] <= 0:
        raise ValueError("temperature")
    if x is not None and (
        x.shape != (5,) or not np.isfinite(x).all() or np.any((x[1:] < 0) | (x[1:] > 1))
    ):
        raise ValueError("features")


def upward(value: F) -> float:
    """Round exact rational bounds toward positive infinity, including conversion."""
    return float(np.nextafter(float(value), np.inf))


def rational_basis(i: int, degree: int, value: F, knots: list[F]) -> F:
    """Exact recursion proves extrema without using observed prediction errors."""
    if degree == 0:
        return F(int(knots[i] <= value < knots[i + 1]))
    left, right = knots[i + degree] - knots[i], knots[i + degree + 1] - knots[i + 1]
    return (
        (value - knots[i]) / left * rational_basis(i, degree - 1, value, knots) if left else F(0)
    ) + (
        (knots[i + degree + 1] - value) / right * rational_basis(i + 1, degree - 1, value, knots)
        if right
        else F(0)
    )


@lru_cache(maxsize=8)
def _certificate(signature: str) -> Json:
    """Cubic second derivatives are linear on knot pieces, so endpoints suffice."""
    head = json.loads(signature)
    c = head["coefficients"]
    knots = [F(v) for v in old.kernel.KNOTS]
    cells, independent = [], True
    for feature in range(4):
        coefficients = [F(v) for v in c[2 + 8 * feature : 10 + 8 * feature]]
        derivative, trimmed = coefficients, knots
        for degree in [3, 2]:
            derivative = [
                degree
                * (derivative[i + 1] - derivative[i])
                / (trimmed[i + degree + 1] - trimmed[i + 1])
                for i in range(len(derivative) - 1)
            ]
            trimmed = trimmed[1:-1]
        polynomial = PPoly.from_spline(
            BSpline(old.kernel.KNOTS, [float(v) for v in coefficients], 3)
        ).derivative(2)
        # Each basis stage has absolute error <= 12*previous + 64*u.
        # Widths exceed .19; exact basis magnitudes are <=1. Three stages
        # give 10048*u. Both endpoint construction and direct dots are covered.
        arithmetic = F(2 * 10048 + 64, 2**53) * sum(abs(v) for v in coefficients)
        feature_cells = []
        for index in range(64):
            lo, hi = F(index, 64), F(index + 1, 64)
            split = sorted({lo, hi, *(v for v in knots if lo < v < hi)})
            extrema: list[F] = [
                abs(derivative[-1])
                if v == 1
                else abs(
                    sum((derivative[i] * rational_basis(i, 1, v, trimmed) for i in range(6)), F(0))
                )
                for v in split
            ]
            maximum = max(extrema)
            scipy_max = max(abs(float(polynomial(float(v)))) for v in split)
            independent &= math.isclose(float(maximum), scipy_max, rel_tol=1e-12, abs_tol=1e-12)
            interpolation = (hi - lo) ** 2 * maximum / 8
            total = interpolation + F(1, 4096) + arithmetic
            feature_cells.append(
                dict(
                    cell=index,
                    split_points=[float(v) for v in split],
                    second_derivative_supremum=upward(maximum),
                    second_derivative_exact=str(maximum),
                    scipy_supremum=scipy_max,
                    interpolation_term=upward(interpolation),
                    quantization_term=1 / 4096,
                    arithmetic_term=upward(arithmetic),
                    logit_error_bound=upward(total),
                    exact_error_bound=str(total),
                )
            )
        cells.append(feature_cells)
    return dict(
        version=VERSION,
        head_sha256=canonical_hash(head),
        cells=cells,
        spline_proof_passed=True,
        independent_extrema_passed=bool(independent),
        numerical_policy_proof_passed=False,
        unresolved_assumptions=[
            "No authenticated all-input rounding/monotonicity contract for the unchanged scipy.special.expit and its platform exp implementation."
        ],
        derivation="Exact binary-rational derivative recurrence; split at every knot; linear second derivative attains absolute extrema at piece endpoints; whole-cell h^2*M/8 plus int16 half-step and conservative basic-operation error. C2 continuity permits the whole-cell remainder across knots.",
    )


def certificate(head: Json, table: Array) -> Json:
    """Bind coefficients and exact table bytes; caller mutations cannot alter the cached proof."""
    validate(head)
    result = deepcopy(
        _certificate(
            json.dumps(
                dict(coefficients=head["coefficients"], temperature=head["temperature"]),
                sort_keys=True,
            )
        )
    )
    result["table_sha256"] = canonical_hash(table.tolist())
    return result


def region(lo: float, hi: float) -> str | None:
    """Strict confident thresholds preserve escalation at exact ties."""
    if hi < 0.25:
        return "accept"
    if lo > 0.75:
        return "reject"
    if lo >= 0.25 and hi <= 0.75:
        return "escalate"
    return None


def guard(head: Json, x: Array, table: Array, supplied: Json | None) -> Json:
    """An empirical interval is useful evidence but cannot grant a certified fast result."""
    validate(head, x)
    with np.errstate(over="ignore", invalid="ignore"):
        p = float(old.direct(head, x[None])[0])
    row: Json = dict(
        x=x.tolist(),
        direct_probability=p,
        direct_action=action(p),
        guarded_action=action(p),
        fast_path=False,
        empirical_fast_path=False,
        table_probability=None,
        table_action=None,
        probability_interval=None,
        scaled_logit_interval=None,
        empirical_action=None,
        interval_certified=False,
    )
    reason = ""
    if supplied is None:
        reason = "missing_bounds"
    elif supplied.get("version") != VERSION:
        reason = "stale_version"
    elif table.shape != (4, 65) or table.dtype != np.dtype("<i2"):
        reason = "table_format"
    elif np.any((table == -32768) | (table == 32767)):
        reason = "saturated_table"
    elif supplied.get("table_sha256") != canonical_hash(table.tolist()) or not np.array_equal(
        table, old.table(head, 65, "int16")[0]
    ):
        reason = "swapped_table"
    elif supplied != dict(
        _certificate(
            json.dumps(
                dict(coefficients=head["coefficients"], temperature=head["temperature"]),
                sort_keys=True,
            )
        ),
        table_sha256=canonical_hash(table.tolist()),
    ):
        reason = "underbound_or_changed_certificate"
    if reason:
        row["fallback_reason"] = reason
        return row
    positions = x[1:] * 64
    indices = np.minimum(np.floor(positions).astype(int), 63)
    weights = positions - indices
    local = (
        table[np.arange(4), indices] * (1 - weights) + table[np.arange(4), indices + 1] * weights
    ) / 2048
    c = head["coefficients"]
    with np.errstate(over="ignore", invalid="ignore"):
        z = float((c[0] * x[0] + c[1] + local.sum()) / head["temperature"])
    if not math.isfinite(z) or abs(z) > 32 or p in [0.0, 1.0]:
        row["fallback_reason"] = "saturation_or_overflow"
        return row
    assert supplied is not None
    # Global arithmetic is unchanged. Allow different local sum order and
    # rounding in multiply, adds, division and table interpolation explicitly.
    magnitude = abs(c[0] * x[0]) + abs(c[1]) + float(np.abs(local).sum()) + 1
    radius = (
        sum(supplied["cells"][j][int(i)]["logit_error_bound"] for j, i in enumerate(indices))
        + 64 * np.finfo(float).eps * magnitude
    ) / head["temperature"]
    lower, upper = np.nextafter(z - radius, -np.inf), np.nextafter(z + radius, np.inf)
    # This widening is an empirical libm allowance, not a universal proof.
    lo = max(0.0, float(np.nextafter(expit(lower) - 1e-12, -np.inf)))
    hi = min(1.0, float(np.nextafter(expit(upper) + 1e-12, np.inf)))
    table_p = float(old.lookup(head, x[None], table, "int16", "linear")[0])
    candidate = region(lo, hi)
    row.update(
        table_probability=table_p,
        table_action=action(table_p),
        probability_interval=[lo, hi],
        scaled_logit_interval=[float(lower), float(upper)],
        empirical_action=candidate,
        empirical_fast_path=candidate is not None,
        fallback_reason="unproven_numerical_policy"
        if candidate is not None
        else "threshold_intersection",
    )
    return row


def panel(head: Json) -> tuple[Array, list[str]]:
    """Keep every frozen vector and add adjacent floats without opening labels."""
    x, kinds, _ = old.panel(head)
    rows = x.tolist()
    for feature in range(4):
        for knot in np.unique(old.kernel.KNOTS):
            for direction in [-np.inf, np.inf]:
                value = float(np.nextafter(knot, direction))
                if 0 <= value <= 1:
                    row = [0.0, 0.5, 0.5, 0.5, 0.5]
                    row[feature + 1] = value
                    rows.append(row)
                    kinds.append("nextafter_knot")
    for row in x[-6:][1::3]:
        for direction in [-np.inf, np.inf]:
            changed = row.copy()
            changed[0] = np.nextafter(changed[0], direction)
            rows.append(changed.tolist())
            kinds.append("nextafter_threshold")
    return np.asarray(rows, dtype=np.float64), kinds


def evaluate(head: Json, table: Array, x: Array, kinds: list[str]) -> list[Json]:
    """Keep every vector, even a boundary fallback, so aggregate zeros can be replayed."""
    cert = certificate(head, table)
    rows = []
    reference = old.reference(head, x)
    for index, (vector, kind) in enumerate(zip(x, kinds, strict=True)):
        row = guard(head, vector, table, cert)
        row.update(
            vector_id=index,
            kind=kind,
            reference_probability=float(reference[index]),
            reference_action=action(float(reference[index])),
        )
        rows.append(row)
        if (index + 1) % 512 == 0:
            progress("vectors", index + 1, len(x) - index - 1)
    progress("vectors", len(x), 0)
    return rows
