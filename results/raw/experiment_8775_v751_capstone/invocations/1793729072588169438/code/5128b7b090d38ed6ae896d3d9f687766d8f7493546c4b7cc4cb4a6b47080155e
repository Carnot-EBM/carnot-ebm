"""REQ-VERIFY-8154: fit small policies from the same twelve source signals.

Public fit geometry cannot depend on outcomes. Energies and logistic controls
share weights because changing probability notation adds no truth evidence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline  # type: ignore[import-untyped]
from scipy.optimize import minimize  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]
from scipy.spatial.distance import pdist  # type: ignore[import-untyped]

from carnot.experiment_8021_v695_typed_decision_test import action, loss
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
Array = NDArray[np.float64]
FITTED_ARMS = [
    "scalar_holistic",
    "scalar_span",
    "mean_probability",
    "linear12",
    "additive_cubic",
    "radial16",
]
ARMS = [*FITTED_ARMS, "equivalent_logistic", "always_escalate"]
RIDGES = [0.0001, 0.001, 0.01, 0.1, 1.0]
CONFIG = dict(
    ridges=RIDGES,
    folds=4,
    max_iterations=256,
    total_seconds=600,
    degree=3,
    quantiles=[0.25, 0.5, 0.75],
    endpoint_multiplicity=4,
    padding=1e-8,
    centers=16,
    calibration_ridge=0.0001,
    calibration_initial=[0.0, 1.0],
    costs=dict(false_accept=5, false_reject=1, escalate=0.5, correct=0),
    tie_rule="escalate",
    probability_clip=[1e-6, 1 - 1e-6],
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real completed counts make a slow fit distinguishable from a stalled child."""
    print(f"[exp8154] phase={phase} completed={completed} pending={pending}", flush=True)


def geometry(x: Array, source_ids: list[str]) -> Json:
    """Freeze knots and centers from public fit rows before consulting targets."""
    mean, scale = x.mean(0), x.std(0)
    scale[scale == 0] = 1
    z = (x - mean) / scale
    knots = [
        [
            *[float(z[:, j].min() - 1e-8)] * 4,
            *np.quantile(z[:, j], [0.25, 0.5, 0.75]).tolist(),
            *[float(z[:, j].max() + 1e-8)] * 4,
        ]
        for j in range(12)
    ]
    keys = [hashlib.sha256(s.encode()).hexdigest() for s in source_ids]
    selected = [min(range(len(x)), key=lambda i: keys[i])]
    while len(selected) < min(16, len(x)):
        remaining = [i for i in range(len(x)) if i not in selected]
        selected.append(
            min(
                remaining,
                key=lambda i: (-float(np.min(np.sum((z[selected] - z[i]) ** 2, axis=1))), keys[i]),
            )
        )
    distances = pdist(z)
    positive = distances[distances > 0]
    return dict(
        mean=mean.tolist(),
        scale=scale.tolist(),
        knots=knots,
        centers=z[selected].tolist(),
        center_source_ids=[source_ids[i] for i in selected],
        width=float(np.median(positive)) if len(positive) else 1.0,
        fit_source_ids=source_ids,
    )


def design(arm: str, x: Array, g: Json) -> Array:
    """Use sealed transforms for every query; scalar controls restrict information."""
    z = (x - np.asarray(g["mean"])) / np.asarray(g["scale"])
    if arm in ("scalar_holistic", "scalar_span"):
        columns = z[:, [0 if arm == "scalar_holistic" else 9]]
    elif arm == "mean_probability":
        p = np.clip((expit(x[:, 0]) + expit(x[:, 9])) / 2, 1e-6, 1 - 1e-6)
        columns = np.log(p / (1 - p))[:, None]
    elif arm == "linear12":
        columns = z
    elif arm == "additive_cubic":
        columns = np.column_stack(
            [
                BSpline.design_matrix(np.clip(z[:, j], k[3], k[-4]), k, 3).toarray()
                for j, k in enumerate(g["knots"])
            ]
        )
    elif arm == "radial16":
        squared = np.sum((z[:, None, :] - np.asarray(g["centers"])[None, :, :]) ** 2, axis=2)
        columns = np.exp(-squared / (2 * g["width"] ** 2))
    else:
        raise ValueError("arm")
    return np.column_stack((np.ones(len(x)), columns))


def objective(
    theta: Array, phi: Array, y: Array, ridge: float, penalize_intercept: bool = False
) -> tuple[float, Array]:
    """Leave base intercepts free so ridge cannot impose a target prevalence."""
    penalty = theta.copy()
    if not penalize_intercept:
        penalty[0] = 0
    z = phi @ theta
    return float(np.mean(np.logaddexp(0, z) - y * z) + ridge * (penalty @ penalty)), np.asarray(
        phi.T @ (expit(z) - y) / len(y) + 2 * ridge * penalty
    )


def solve(
    phi: Array,
    y: Array,
    ridge: float,
    deadline: float,
    *,
    initial: Array | None = None,
    calibration: bool = False,
) -> Json:
    """Reject degenerate targets and retain convergence receipts for each solve."""
    if len(set(y.tolist())) != 2 or not np.isin(y, [0, 1]).all() or not np.isfinite(phi).all():
        raise ValueError("fit_operands")
    began = time.monotonic()
    theta = np.zeros(phi.shape[1]) if initial is None else initial.copy()
    iterations = 0
    heartbeat = began

    def boundary(value: Array) -> None:
        """Enforce the shared fitting deadline at each actual solver iteration."""
        nonlocal iterations, heartbeat
        iterations += 1
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError("fit_budget_600s")
        if now - heartbeat >= 60:
            progress("fit_iterations", iterations, 256 - iterations)
            heartbeat = now

    boundary(theta)
    result = minimize(
        lambda t: objective(t, phi, y, ridge, calibration),
        theta,
        jac=True,
        method="L-BFGS-B",
        callback=boundary,
        options=dict(maxiter=256, ftol=1e-12, gtol=1e-8, maxls=40),
    )
    return dict(
        weights=result.x.tolist(),
        initial_loss=objective(theta, phi, y, ridge, calibration)[0],
        final_loss=float(result.fun),
        converged=bool(result.success),
        iterations=int(result.nit),
        message=str(result.message),
        duration_s=time.monotonic() - began,
        ridge=ridge,
        gradient_norm=float(np.max(np.abs(result.jac))),
    )


def choose_ridge(losses: dict[float, float]) -> float:
    """A tied loss chooses the larger declared ridge rather than extra capacity."""
    return min(losses, key=lambda ridge: (losses[ridge], -ridge))


def train(rows: list[Json], raw: Path) -> Json:
    """Select ridge on fit folds, seal base weights, then use tune targets only."""
    began = time.monotonic()
    deadline = began + 600
    fit, tune = (
        [r for r in rows if r["role"] == role and r["status"] == "completed" and r["y"] in (0, 1)]
        for role in ("fit", "tune")
    )
    ids = [r["source_cluster_id"] for r in fit]
    tids = [r["source_cluster_id"] for r in tune]
    if set(ids) & set(tids) or len(set(ids)) != len(ids) or len(set(tids)) != len(tids):
        raise ValueError("source_leak")
    x, tx = (np.asarray([r["x"] for r in role], dtype=float) for role in (fit, tune))
    y, ty = (np.asarray([r["y"] for r in role], dtype=float) for role in (fit, tune))
    order = sorted(range(len(fit)), key=lambda i: hashlib.sha256(ids[i].encode()).hexdigest())
    folds = np.empty(len(fit), dtype=int)
    for rank, i in enumerate(order):
        folds[i] = rank % 4
    g = geometry(x, ids)
    heads, fold_rows, failures = [], [], []
    for arm in FITTED_ARMS:
        progress("before_benchmark_" + arm, len(heads), len(FITTED_ARMS) - len(heads))
        losses: dict[float, float] = {}
        for ridge in RIDGES:
            scores = []
            for fold in range(4):
                mask = folds != fold
                fg = geometry(x[mask], [s for s, keep in zip(ids, mask, strict=True) if keep])
                record = dict(
                    arm=arm,
                    ridge=ridge,
                    fold=fold,
                    geometry=fg,
                    fit_source_ids=fg["fit_source_ids"],
                    held_source_ids=[s for s, keep in zip(ids, ~mask, strict=True) if keep],
                )
                try:
                    h = solve(design(arm, x[mask], fg), y[mask], ridge, deadline)
                    if not h["converged"]:
                        raise ValueError("fit_nonconvergence")
                    z = design(arm, x[~mask], fg) @ np.asarray(h["weights"])
                    score = float(np.mean(np.logaddexp(0, z) - y[~mask] * z))
                    scores.append(score)
                    record.update(h, log_loss=score, passed=True)
                except (ValueError, TimeoutError) as error:
                    record.update(passed=False, error=str(error))
                    failures.append(dict(record))
                fold_rows.append(record)
                progress("fold_completed_" + arm, len(fold_rows), 120 - len(fold_rows))
            if len(scores) == 4:
                losses[ridge] = float(np.mean(scores))
        try:
            if len(losses) != len(RIDGES):
                raise ValueError("incomplete_source_folds")
            ridge = choose_ridge(losses)
            h = solve(design(arm, x, g), y, ridge, deadline)
            if not h["converged"]:
                raise ValueError("full_fit_nonconvergence")
            heads.append(
                dict(h, arm=arm, geometry=g, fit_fold_losses={str(k): v for k, v in losses.items()})
            )
        except (ValueError, TimeoutError) as error:
            failures.append(dict(arm=arm, phase="full_fit", error=str(error)))
        progress("after_benchmark_" + arm, len(heads), len(FITTED_ARMS) - len(heads))
    atomic_json(raw / "base_heads.json", dict(heads=heads))
    progress("base_heads_sealed_before_tune_calibration", len(heads), 0)
    for h in heads:
        progress("before_benchmark_calibration_" + h["arm"])
        try:
            z = design(h["arm"], tx, h["geometry"]) @ np.asarray(h["weights"])
            cal = solve(
                np.column_stack((np.ones(len(tx)), z)),
                ty,
                0.0001,
                deadline,
                initial=np.array([0.0, 1.0]),
                calibration=True,
            )
            if not cal["converged"]:
                raise ValueError("calibration_nonconvergence")
            h.update(calibration=cal["weights"], calibration_receipt=cal, tune_source_ids=tids)
        except (ValueError, TimeoutError) as error:
            failures.append(dict(arm=h["arm"], phase="calibration", error=str(error)))
        progress("after_benchmark_calibration_" + h["arm"])
    result = dict(
        heads=heads,
        fit_fold_rows=fold_rows,
        failures=failures,
        decision_rule=CONFIG["costs"],
        tie_rule="escalate",
        training_budget=dict(
            CONFIG,
            actual_duration_s=time.monotonic() - began,
            solves=len(fold_rows) + len(heads) * 2,
            fit_iterations=sum(r.get("iterations", 0) for r in fold_rows)
            + sum(
                h["iterations"] + h.get("calibration_receipt", {}).get("iterations", 0)
                for h in heads
            ),
        ),
    )
    atomic_json(raw / "frozen_heads.json", result)
    progress("all_heads_sealed_before_reserved_capture", len(heads), 0)
    return result


def evaluate(rows: list[Json], heads: list[Json]) -> Json:
    """Recompute costs per original source; logistic identity is a control only."""
    records, parity = [], []
    for row in rows:
        eligible = row["status"] == "completed" and row["y"] in (0, 1)
        ps: Json = {}
        if eligible:
            x = np.asarray([row["x"]], dtype=float)
            for h in heads:
                z = float((design(h["arm"], x, h["geometry"]) @ np.asarray(h["weights"]))[0])
                b, a = h["calibration"]
                z = b + a * z
                w = np.exp(np.array([0.0, z]) - max(0.0, z))
                p, lp = float(w[1] / w.sum()), float(expit(z))
                decision = action(p)
                parity.append(
                    dict(
                        unit_id=row["unit_id"],
                        arm=h["arm"],
                        E0=0.0,
                        E1=-z,
                        z=z,
                        energy_probability=p,
                        logistic_probability=lp,
                        energy_action=decision,
                        logistic_action=action(lp),
                        passed=abs(p - lp) <= 1e-10 and decision == action(lp),
                    )
                )
                ps[h["arm"]] = p
                if h["arm"] == "radial16":
                    ps["equivalent_logistic"] = lp
        ps["always_escalate"] = None
        for arm in ARMS:
            p = ps.get(arm)
            d = action(p)
            cost = loss(d, row["y"]) if eligible else None
            cp = np.clip(p, 1e-6, 1 - 1e-6) if p is not None else None
            records.append(
                dict(
                    unit_id=row["unit_id"],
                    source_cluster_id=row["source_cluster_id"],
                    role=row["role"],
                    arm=arm,
                    condition="complete_original_source",
                    metric="typed_decision_cost",
                    numerator=cost,
                    denominator=int(eligible),
                    status="completed" if eligible else "excluded",
                    exclusion_reason=None
                    if eligible
                    else row["exclusion_reason"] or "unknown_target",
                    p=p,
                    y=row["y"],
                    action=d,
                    brier=float((p - row["y"]) ** 2) if eligible and p is not None else None,
                    log_loss=float(-row["y"] * np.log(cp) - (1 - row["y"]) * np.log1p(-cp))
                    if eligible and cp is not None
                    else None,
                    false_accept=int(eligible and d == "accept" and row["y"] == 1),
                    coverage=int(eligible and d != "escalate"),
                )
            )
    summaries = []
    for role in ("fit", "tune"):
        for arm in ARMS:
            subset = [
                r for r in records if r["role"] == role and r["arm"] == arm and r["denominator"]
            ]
            summaries.append(
                dict(
                    role=role,
                    arm=arm,
                    n=len(subset),
                    typed_cost=float(np.mean([r["numerator"] for r in subset])) if subset else None,
                    **{
                        k: float(np.mean([r[k] for r in subset if r[k] is not None]))
                        if any(r[k] is not None for r in subset)
                        else None
                        for k in ("brier", "log_loss", "coverage", "false_accept")
                    },
                )
            )
    return dict(
        rows=records,
        tuning_rows=[r for r in records if r["role"] == "tune"],
        energy_logistic_parity_rows=parity,
        development_metrics=summaries,
        reduction_checksum=canonical_hash(records),
    )
