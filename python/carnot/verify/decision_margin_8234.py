"""REQ-VERIFY-8234: public decision margins change loss weights, never permissions.

Labels fit small heads only within their original roles. The native probability
sets each weight before fitting, so calibration cannot redefine the objective.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable
from typing import Any, cast
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

from carnot.verify import evidence_energy_8154 as base
from carnot.verify import restricted_action_rule_8207 as rule

Json = dict[str, Any]
BASE_OBJECTIVE = base.objective
COMPARATORS = [
    "additive_uniform",
    "additive_margin",
    "logistic_uniform",
    "logistic_margin",
    "energy_uniform",
    "energy_global",
]


def margin_weight(p0: float | None, permission: str) -> float | None:
    """Near action ties receive more loss weight without consulting correctness."""
    if permission not in {"accept", "reject", "escalate"}:
        raise ValueError("permission")
    if p0 is None:
        return None
    if not math.isfinite(p0) or not 0 <= p0 <= 1:
        raise ValueError("probability")
    losses = [1 - p0, 0.5] + ([5 * p0] if permission == "accept" else [])
    ordered = sorted(losses)
    return 1 + 4 * math.exp(-(ordered[1] - ordered[0]) / 0.05)


def weights(rows: list[Json]) -> list[Json]:
    """Retain missing inputs as slots rather than creating replacement examples."""
    return [
        dict(
            unit_id=r["unit_id"],
            weight=margin_weight(r["p0"], r["baseline_action"]) if r["x"] is not None else None,
        )
        for r in rows
    ]


def validate_roles(roles: Json) -> None:
    """Disjoint source roles prevent targets from entering another fitting stage."""
    items = [
        r for key in ["head_fit", "temperature_fit", "calibration", "reserved"] for r in roles[key]
    ]
    if (
        len({r["unit_id"] for r in items}) != len(items)
        or len({r["source_cluster_id"] for r in items}) != len(items)
        or len(roles["reserved"]) != 128
    ):
        raise ValueError("roles")


def objective(
    theta: base.Array, phi: base.Array, y: base.Array, ridge: float, w: base.Array
) -> tuple[float, base.Array]:
    """Normalize weighted loss while keeping the original ridge strength intact."""
    if w.shape != y.shape or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("weights")
    if np.all(w == w[0]):
        return cast(tuple[float, base.Array], BASE_OBJECTIVE(theta, phi, y, ridge))
    penalty = theta.copy()
    penalty[0] = 0
    z, total = phi @ theta, float(w.sum())
    return (
        float(w @ (np.logaddexp(0, z) - y * z) / total + ridge * (penalty @ penalty)),
        np.asarray(phi.T @ (w * (expit(z) - y)) / total + 2 * ridge * penalty),
    )


def train(rows: list[Json], roles: Json) -> Json:
    """Pair every basis with both weight rules under the unchanged solver budget."""
    validate_roles(roles)
    lookup = {r["unit_id"]: r for r in rows}
    permitted = {
        r["unit_id"] for k in ["head_fit", "temperature_fit", "calibration"] for r in roles[k]
    }
    if len(lookup) != len(rows) or not lookup or set(lookup) != permitted:
        raise ValueError("fit_operands")
    selected = {
        k: [
            lookup[s["unit_id"]]
            for s in roles[k]
            if lookup[s["unit_id"]]["x"] is not None
            and lookup[s["unit_id"]]["p0"] is not None
            and lookup[s["unit_id"]]["y"] in (0, 1)
        ]
        for k in ["head_fit", "temperature_fit"]
    }
    fit, temp = selected["head_fit"], selected["temperature_fit"]
    if len(fit) < 16 or not temp:
        raise ValueError("fit_operands")
    x, y = np.asarray([r["x"] for r in fit]), np.asarray([r["y"] for r in fit])
    if x.shape[1:] != (16,) or not np.isfinite(x).all():
        raise ValueError("fit_operands")
    ids = [r["source_cluster_id"] for r in fit]
    order = sorted(range(len(ids)), key=lambda i: hashlib.sha256(ids[i].encode()).hexdigest())
    folds: Any = np.empty(len(fit), dtype=int)
    for rank, i in enumerate(order):
        folds[i] = rank % 4
    deadline = time.monotonic() + 600
    heads: list[Json] = []
    records: list[Json] = []
    for basis in rule.ARMS:
        for weighting in ["uniform", "margin"]:
            arm = basis + "_" + weighting
            base.progress("before_benchmark_" + arm, len(heads), 6 - len(heads))
            w = (
                np.ones(len(fit))
                if weighting == "uniform"
                else np.asarray(
                    [margin_weight(r["p0"], r["baseline_action"]) for r in fit], dtype=float
                )
            )
            losses = {}
            for ridge in base.RIDGES:
                held = []
                for fold in range(4):
                    mask = folds != fold
                    g = base.geometry(
                        x[mask], [s for s, keep in zip(ids, mask, strict=True) if keep]
                    )
                    with patch.object(
                        base,
                        "objective",
                        lambda t, p, labels, reg, cal=False: objective(t, p, labels, reg, w[mask]),
                    ):
                        solved = base.solve(
                            rule.design(basis, x[mask], g), y[mask], ridge, deadline
                        )
                    if not solved["converged"]:
                        raise ValueError("fit_nonconvergence")
                    z = rule.design(basis, x[~mask], g) @ np.asarray(solved["weights"])
                    loss = float(w[~mask] @ (np.logaddexp(0, z) - y[~mask] * z) / w[~mask].sum())
                    held.append(loss)
                    records.append(
                        dict(
                            arm=arm,
                            ridge=ridge,
                            fold=fold,
                            log_loss=loss,
                            geometry=g,
                            solve_receipt=solved,
                        )
                    )
                    base.progress("fold_completed", len(records), 120 - len(records))
                losses[ridge] = float(np.mean(held))
            g, ridge = base.geometry(x, ids), base.choose_ridge(losses)
            with patch.object(
                base,
                "objective",
                lambda t, p, labels, reg, cal=False: objective(t, p, labels, reg, w),
            ):
                solved = base.solve(rule.design(basis, x, g), y, ridge, deadline)
            if not solved["converged"]:
                raise ValueError("fit_nonconvergence")
            z = rule.design(basis, np.asarray([r["x"] for r in temp]), g) @ np.asarray(
                solved["weights"]
            )
            ty = np.asarray([r["y"] for r in temp])
            loss_fn: Callable[[float], float] = lambda t: float(
                np.mean(np.logaddexp(0, z / t) - ty * z / t)
            )
            base.progress("before_benchmark_temperature_" + arm)
            tuned = minimize_scalar(
                loss_fn, bounds=(0.25, 4), method="bounded", options=dict(xatol=1e-12)
            )
            if not tuned.success:
                raise ValueError("temperature_nonconvergence")
            temperature = min([0.25, float(tuned.x), 4.0], key=lambda t: (loss_fn(t), t))
            heads.append(
                dict(
                    arm=arm,
                    basis=basis,
                    weighting=weighting,
                    geometry=g,
                    weights=solved["weights"],
                    ridge=ridge,
                    temperature=temperature,
                    fit_ids=[r["unit_id"] for r in fit],
                    solve_receipt=solved,
                    temperature_ids=[r["unit_id"] for r in temp],
                )
            )
            base.progress("after_benchmark_" + arm, len(heads), 6 - len(heads))
    return dict(heads=heads, fit_fold_rows=records)


def select_comparator(scored: list[Json], roles: Json) -> Json:
    """Only calibration labels select a control; missing decisions keep their cost."""
    ids = {r["unit_id"] for r in roles["calibration"]}
    if any(r["role"] != "calibration" or r["unit_id"] not in ids for r in scored):
        raise ValueError("calibration")
    summaries: list[Json] = []
    for arm in COMPARATORS:
        rs = [r for r in scored if r["arm"] == arm]
        if len(rs) != len(ids) or {r["unit_id"] for r in rs} != ids:
            raise ValueError("calibration")
        costs = math.fsum(base.loss(r["action"], r["y"]) for r in rs)
        complete = [r for r in rs if r["p"] is not None and r["y"] in (0, 1)]
        brier = math.fsum((r["p"] - r["y"]) ** 2 for r in complete)
        summaries.append(
            dict(
                arm=arm,
                all_slot_cost=dict(numerator=costs, denominator=len(rs)),
                unweighted_brier=dict(numerator=brier, denominator=len(complete)),
            )
        )
    winner = min(
        summaries,
        key=lambda s: (
            s["all_slot_cost"]["numerator"] / s["all_slot_cost"]["denominator"],
            s["unweighted_brier"]["numerator"] / s["unweighted_brier"]["denominator"]
            if s["unweighted_brier"]["denominator"]
            else math.inf,
            COMPARATORS.index(s["arm"]),
        ),
    )
    return dict(winner, candidates=summaries, calibration_ids=sorted(ids))


def null_gate(gains: list[float], protocol: Json) -> bool:
    """Equal cost controls must not satisfy the registered benefit thresholds."""
    draws = (
        np.random.default_rng(protocol["seed"])
        .choice(gains, (protocol["draws"], len(gains)), replace=True)
        .mean(axis=1)
    )
    return bool(
        len(gains) == protocol["intended"]
        and sum(g > 0 for g in gains) >= protocol["improved_sources_min"]
        and np.quantile(draws, protocol["alpha"]) > protocol["lower_cost_gain_gt"]
    )
