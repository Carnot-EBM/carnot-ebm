"""REQ-VERIFY-8437: faster arithmetic preserves the rest of the measured service.

These counterfactuals assume identical work and service semantics. Host repeats
provide descriptive uncertainty, not independent deployment evidence.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

Json = dict[str, Any]
PRIMARY = dict(workload="natural", batch=1, predictions_per_update=8)
ACCELERATIONS = {"1": 1, "10": 10, "100": 100, "infinity": None}
SEED = 7268437


def speed(total: float, kernel: float, acceleration: int | None) -> float | str:
    """Reject invalid partitions rather than turning absent work into a speed gain."""
    if not all(math.isfinite(v) for v in (total, kernel)) or total <= 0 or not 0 <= kernel <= total:
        raise ValueError("invalid_T_K")
    if acceleration is not None and acceleration < 1:
        raise ValueError("invalid_acceleration")
    if kernel == total:
        return "infinite" if acceleration is None else float(acceleration)
    return total / (total - kernel + (0 if acceleration is None else kernel / acceleration))


def bounds(row: Json) -> Json:
    """Conversion is a separate optimistic sensitivity; persistence is never a kernel."""
    total, kernel, conversion = (
        float(row[n]) for n in ("complete_wall_s", "kernel_s", "conversion_s")
    )
    if not math.isfinite(conversion) or conversion < 0:
        raise ValueError("invalid_conversion")
    return dict(
        repeat_id=row.get("repeat_id"),
        cell=row["cell"],
        repeat=row.get("repeat"),
        role="primary" if row["cell"] == PRIMARY else "secondary",
        T_s=total,
        K_s=kernel,
        Kplus_s=kernel + conversion,
        fraction_accelerable=kernel / total if total > 0 else None,
        fraction_accelerable_optimistic=(kernel + conversion) / total if total > 0 else None,
        lower_bound_nonkernel_removal=max(0, 0.9 - kernel / total) if total > 0 else None,
        arithmetic={label: speed(total, kernel, a) for label, a in ACCELERATIONS.items()},
        optimistic_conversion={
            label: speed(total, kernel + conversion, a) for label, a in ACCELERATIONS.items()
        },
    )


def analyze(rows: list[Json]) -> Json:
    """Pool matched repeat totals in each draw; averaging speed ratios changes the estimand."""
    panel = [bounds(row) for row in rows]
    primary = [row for row in panel if row["role"] == "primary"]
    result: Json = dict(
        per_repeat_bounds=panel,
        primary_cell=PRIMARY,
        descriptive_interval=None,
        observed_primary_range=None,
        fraction_accelerable=None,
        lower_bound_nonkernel_removal=None,
        arithmetic_10x_feasible_under_fixed_service=None,
        every_primary_repeat_rules_out_arithmetic_10x=None,
    )
    if not primary:
        return result
    t = np.array([row["T_s"] for row in primary])
    kernel = np.array([row["K_s"] for row in primary])
    plus = np.array([row["Kplus_s"] for row in primary])
    indices = np.random.default_rng(SEED).integers(0, len(primary), (10000, len(primary)))
    sums_t = t[indices].sum(axis=1)
    interval: Json = {}
    observed: Json = {}
    for key, values in [("arithmetic", kernel), ("optimistic_conversion", plus)]:
        sums_k = values[indices].sum(axis=1)
        interval[key], observed[key] = {}, {}
        for label, a in ACCELERATIONS.items():
            denominator = sums_t - sums_k + (0 if a is None else sums_k / a)
            ratios = np.divide(
                sums_t, denominator, out=np.full(10000, np.inf), where=denominator != 0
            )
            # Rank endpoints keep an explicit infinite bound valid without NaN interpolation.
            ordered = np.sort(ratios)
            lo, hi = (
                np.quantile(ratios, [0.025, 0.975])
                if np.isfinite(ratios).all()
                else (ordered[249], ordered[9749])
            )

            def encode(v: Any) -> float | str:
                """JSON cannot carry numerical infinity as a portable bound."""
                return "infinite" if math.isinf(float(v)) else float(v)

            interval[key][label] = dict(
                lower_95=encode(lo),
                upper_95=encode(hi),
                pooled_point=speed(float(t.sum()), float(values.sum()), a),
            )
            finite = [row[key][label] for row in primary if row[key][label] != "infinite"]
            observed[key][label] = dict(
                min=min(finite) if finite else "infinite",
                max="infinite" if len(finite) != len(primary) else max(finite),
            )
    fractions = [row["fraction_accelerable"] for row in primary]
    removal = [row["lower_bound_nonkernel_removal"] for row in primary]
    result.update(
        descriptive_interval=dict(
            **interval,
            confidence=0.95,
            resamples=10000,
            seed=SEED,
            method="paired whole repeat pooled ratio; interpolated empirical quantiles (rank endpoints for infinite bounds)",
            independent_deployment_samples=False,
            repeats=len(primary),
        ),
        observed_primary_range=observed,
        fraction_accelerable=dict(
            range=[min(fractions), max(fractions)], pooled=float(kernel.sum() / t.sum())
        ),
        lower_bound_nonkernel_removal=dict(
            range=[min(removal), max(removal)], pooled=max(0, 0.9 - float(kernel.sum() / t.sum()))
        ),
        arithmetic_10x_feasible_under_fixed_service=bool(kernel.sum() / t.sum() >= 0.9),
        every_primary_repeat_rules_out_arithmetic_10x=all(f < 0.9 for f in fractions),
    )
    return result
