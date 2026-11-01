"""REQ-VERIFY-8208: fit matched heads while targets stay in their frozen roles.

The qualified optimizer still owns numerical fitting. Recording its callbacks
adds evidence about real parameter changes without changing its objective.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import time
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]
from scipy.special import expit  # type: ignore[import-untyped]

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import evidence_energy_8154 as base
from carnot.verify import restricted_action_rule_8207 as rule
from carnot.verify import selective_rule_8194 as selective

Json = dict[str, Any]
MINIMIZE = base.minimize
CONFIG = dict(
    base.CONFIG, seed=7098208, arms=rule.ARMS, coefficients=17, temperature_bounds=[0.25, 4]
)


def solve(phi: base.Array, y: base.Array, ridge: float, deadline: float) -> Json:
    """Observe initial, iteration and final losses from the unchanged solver."""
    trajectory = [base.objective(np.zeros(phi.shape[1]), phi, y, ridge)[0]]

    def observed(*args: Any, **kwargs: Any) -> Any:
        """Keep the original deadline callback active before saving each loss."""
        boundary = kwargs["callback"]

        def callback(theta: base.Array) -> None:
            boundary(theta)
            trajectory.append(base.objective(theta, phi, y, ridge)[0])

        kwargs["callback"] = callback
        return MINIMIZE(*args, **kwargs)

    with patch.object(base, "minimize", observed):
        result = base.solve(phi, y, ridge, deadline)
    if not result["converged"]:
        raise ValueError("fit_nonconvergence")
    trajectory.append(result["final_loss"])
    return dict(
        result,
        loss_trajectory=trajectory,
        parameter_hashes=dict(
            before=canonical_hash([0.0] * phi.shape[1]), after=canonical_hash(result["weights"])
        ),
    )


def train(rows: list[Json], roles: Json) -> Json:
    """Select ridge on fit folds, then temperature on a different fixed role."""
    allowed = {
        "unit_id",
        "source_cluster_id",
        "source_id",
        "x",
        "historical_x",
        "y",
        "role",
        "status",
        "exclusion_reason",
        "slot",
    }
    ids = [r["unit_id"] for r in rows]
    sources = [r["source_cluster_id"] for r in rows]
    registered = [r for role in ("head_fit", "temperature_fit", "calibration") for r in roles[role]]
    if (
        not rows
        or len(set(ids)) != len(ids)
        or len(set(sources)) != len(sources)
        or {r["unit_id"] for r in registered} != set(ids)
        or len(registered) != len(rows)
        or {r["source_cluster_id"] for r in roles["reserved"]} & set(sources)
        or any(
            set(r) - allowed
            or r["role"] not in ("fit", "tune")
            or (r["x"] is not None and (len(r["x"]) != 16 or not np.isfinite(r["x"]).all()))
            for r in rows
        )
    ):
        raise ValueError("source_or_feature_schema")
    selected = {
        role: [
            r
            for r in rows
            if r["unit_id"] in {s["unit_id"] for s in roles[role]}
            and r["x"] is not None
            and r["y"] in (0, 1)
        ]
        for role in ("head_fit", "temperature_fit")
    }
    fit, temp = selected["head_fit"], selected["temperature_fit"]
    if not fit or not temp:
        raise ValueError("fit_operands")
    x, y = (
        np.asarray([r["x"] for r in fit], dtype=float),
        np.asarray([r["y"] for r in fit], dtype=float),
    )
    ids = [r["source_cluster_id"] for r in fit]
    order = sorted(range(len(fit)), key=lambda i: hashlib.sha256(ids[i].encode()).hexdigest())
    folds = np.empty(len(fit), dtype=int)
    for rank, i in enumerate(order):
        folds[i] = rank % CONFIG["folds"]
    deadline = time.monotonic() + CONFIG["total_seconds"]
    heads, records = [], []
    for arm in rule.ARMS:
        base.progress("before_benchmark_8208_" + arm, len(heads), 3 - len(heads))
        losses = {}
        for ridge in base.RIDGES:
            held_losses = []
            for fold in range(CONFIG["folds"]):
                mask = folds != fold
                g = base.geometry(x[mask], [s for s, keep in zip(ids, mask, strict=True) if keep])
                fitted = solve(rule.design(arm, x[mask], g), y[mask], ridge, deadline)
                z = rule.design(arm, x[~mask], g) @ np.asarray(fitted["weights"])
                loss = float(np.mean(np.logaddexp(0, z) - y[~mask] * z))
                held_losses.append(loss)
                records.append(
                    dict(
                        fitted,
                        arm=arm,
                        fold=fold,
                        geometry=g,
                        held_source_ids=[s for s, keep in zip(ids, ~mask, strict=True) if keep],
                        log_loss=loss,
                    )
                )
                base.progress("fold_completed_8208", len(records), 60 - len(records))
            losses[ridge] = float(np.mean(held_losses))
        g = base.geometry(x, ids)
        ridge = base.choose_ridge(losses)
        fitted = solve(rule.design(arm, x, g), y, ridge, deadline)
        base.progress("after_benchmark_8208_" + arm, len(heads) + 1, 2 - len(heads))
        base.progress("before_benchmark_temperature_8208_" + arm)
        z = rule.design(arm, np.asarray([r["x"] for r in temp]), g) @ np.asarray(fitted["weights"])
        ty = np.asarray([r["y"] for r in temp])
        objective = lambda t: float(np.mean(np.logaddexp(0, z / t) - ty * z / t))
        tuning = minimize_scalar(
            objective, bounds=(0.25, 4), method="bounded", options=dict(xatol=1e-12)
        )
        if not tuning.success:
            raise ValueError("temperature_nonconvergence")
        temperature = min([0.25, float(tuning.x), 4.0], key=lambda t: (objective(t), t))
        heads.append(
            dict(
                arm=arm,
                geometry=g,
                weights=fitted["weights"],
                temperature=temperature,
                ridge=ridge,
                fit_ids=[r["unit_id"] for r in fit],
                temperature_ids=[r["unit_id"] for r in temp],
                solve_receipt=fitted,
                parameter_hashes=fitted["parameter_hashes"],
                fit_fold_losses={str(k): v for k, v in losses.items()},
                temperature_receipt=dict(success=True, n=len(temp), nll=objective(temperature)),
            )
        )
        base.progress("after_benchmark_temperature_8208_" + arm, 1, 0)
    return dict(heads=heads, fit_fold_rows=records)


def reduce(data: Json) -> Json:
    """Recompute source costs from frozen heads and independently check notation."""
    rows, roles, heads, baseline = (data[k] for k in ("rows", "roles", "heads", "baseline"))
    role_lookup = {
        r["unit_id"]: role
        for role in ("head_fit", "temperature_fit", "calibration")
        for r in roles[role]
    }
    scored, parity, shifts = [], [], []
    for row in rows:
        query = {k: row[k] for k in ("unit_id", "source_cluster_id", "x", "historical_x")}
        predictions = []
        for head in heads:
            prediction = rule.predict(head, query, baseline)
            predictions.append(prediction)
            p = prediction["p"]
            z = (
                math.fsum(
                    float(a) * float(b)
                    for a, b in zip(
                        rule.design(head["arm"], np.asarray([row["x"]]), head["geometry"])[0],
                        head["weights"],
                        strict=True,
                    )
                )
                if p is not None
                else None
            )
            logistic = float(expit(z / head["temperature"])) if z is not None else None
            passed = (p is None and logistic is None) or (
                abs(p - logistic) <= 1e-10
                and rule.action(logistic, prediction["baseline_action"]) == prediction["action"]
            )
            parity.append(
                dict(
                    unit_id=row["unit_id"],
                    arm=head["arm"],
                    maximum_absolute_error=abs(p - logistic) if p is not None else 0.0,
                    passed=passed,
                )
            )
            for shift in (-100.0, 0.0, 100.0):
                shifted = (
                    rule.probability(shift, shift - z, head["temperature"])
                    if z is not None
                    else None
                )
                shifts.append(
                    dict(
                        unit_id=row["unit_id"],
                        arm=head["arm"],
                        shift=shift,
                        passed=(shifted is None and p is None)
                        or (
                            abs(shifted - p) <= 1e-10
                            and rule.action(shifted, prediction["baseline_action"])
                            == prediction["action"]
                        ),
                    )
                )
            if head["arm"] == "energy":
                predictions.append(
                    dict(
                        prediction,
                        arm="equivalent_logistic_identity",
                        p=logistic,
                        action=rule.action(logistic, prediction["baseline_action"]),
                    )
                )
        fp = None
        if row["historical_x"] is not None:
            phi = base.design("radial16", np.asarray([row["historical_x"]]), baseline["geometry"])[
                0
            ]
            offset, slope = baseline["calibration"]
            fp = float(
                expit(
                    offset
                    + slope
                    * math.fsum(
                        float(a) * float(b) for a, b in zip(phi, baseline["weights"], strict=True)
                    )
                )
            )
        original = dict(
            unit_id=row["unit_id"],
            source_cluster_id=row["source_cluster_id"],
            baseline_action=base.action(fp),
        )
        predictions.extend(
            [
                dict(original, arm="original_frozen_v707_radial", p=fp, action=base.action(fp)),
                dict(original, arm="always_escalate", p=None, action="escalate"),
            ]
        )
        if data.get("original_selective_heads"):
            local = next(
                h for h in data["original_selective_heads"] if h["arm"] == "local_evidence_radial16"
            )
            lp = (
                selective.probability(local, row["x"], scalar=True)
                if row["x"] is not None
                else None
            )
            labels = selective.prediction_set(lp, local["quantiles"])
            predictions.append(
                dict(
                    original,
                    arm="original_local_set",
                    p=lp,
                    action="accept" if labels == [0] else "reject" if labels == [1] else "escalate",
                )
            )
        for prediction in predictions:
            p, target = prediction["p"], row["y"]
            scored.append(
                dict(
                    prediction,
                    role=role_lookup[row["unit_id"]],
                    condition=role_lookup[row["unit_id"]],
                    y=target,
                    metric="restricted_typed_cost",
                    numerator=base.loss(prediction["action"], target) if target in (0, 1) else None,
                    denominator=int(target in (0, 1)),
                    status="completed" if row["x"] is not None else "excluded",
                    exclusion_reason=None
                    if row["x"] is not None
                    else row.get("exclusion_reason") or "missing_features",
                    brier=(p - target) ** 2 if p is not None and target in (0, 1) else None,
                )
            )
    if not all(r["passed"] for r in [*parity, *shifts]):
        raise ValueError("probability_decision_parity")
    return dict(
        rows=scored,
        equivalent_logistic_parity=dict(passed=True, rows=parity),
        common_shift_invariance=dict(passed=True, rows=shifts),
        selected_simple_control=rule.select_simple(heads, rows, roles, baseline),
    )


def diagnostics(rows: list[Json], roles: Json, baseline: Json) -> Json:
    """Separate diagnostic heads cannot alter the natural comparator choice."""
    evidence = {}
    for index, condition in enumerate(("constant_features", "shuffled_targets")):
        base.progress("before_benchmark_8208_" + condition, index, 2 - index)
        changed = deepcopy(rows)
        available = [
            r
            for r in changed
            if r["x"] is not None
            and r["y"] in (0, 1)
            and r["unit_id"] in {s["unit_id"] for s in roles["head_fit"]}
        ]
        if condition == "constant_features":
            for row in changed:
                if row["x"] is not None:
                    row["x"] = [0.0] * 16
        else:
            targets = np.random.default_rng(CONFIG["seed"]).permutation([r["y"] for r in available])
            for row, target in zip(available, targets, strict=True):
                row["y"] = int(target)
        evidence[condition] = dict(
            rows=changed,
            roles=roles,
            baseline=baseline,
            **train(changed, roles),
            diagnostic_only=True,
        )
        base.progress("after_benchmark_8208_" + condition, index + 1, 1 - index)
    return evidence
