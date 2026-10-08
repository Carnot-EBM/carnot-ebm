"""REQ-VERIFY-8207: costs and a baseline permission mask govern small heads.

The mask applies before choosing an action for every input. It supplies no
correctness labels; reused targets can fit probabilities only in their own role.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

from carnot.verify import evidence_energy_8154 as base
from carnot.verify import selective_rule_8194 as old

Json = dict[str, Any]
ARMS = ["energy", "additive", "logistic"]


def probability(good: float, bad: float, temperature: float) -> float:
    """Subtract the energies before exponentiating so extreme values stay finite."""
    return old.logit_probability([0, good - bad], temperature)


def action(p: float | None, baseline_action: str) -> str:
    """A permission restricts available actions without asserting semantic truth."""
    if baseline_action not in {"accept", "reject", "escalate"}:
        raise ValueError("baseline_action")
    if p is None:
        return "escalate"
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("probability")
    costs = {"reject": 1 - p, "escalate": 0.5}
    if baseline_action == "accept":
        costs["accept"] = 5 * p
    winners = [a for a, cost in costs.items() if cost == min(costs.values())]
    return winners[0] if len(winners) == 1 else "escalate"


def design(arm: str, x: base.Array, geometry: Json) -> base.Array:
    """All heads see sixteen signals and have seventeen fitted coefficients."""
    if arm == "energy":
        return base.design("radial16", x, geometry)
    z = (x - np.asarray(geometry["mean"])) / np.asarray(geometry["scale"])
    if arm not in {"additive", "logistic"}:
        raise ValueError("arm")
    return np.column_stack((np.ones(len(x)), np.tanh(z) if arm == "additive" else z))


def train(rows: list[Json], roles: Json) -> list[Json]:
    """Only fit labels set weights; a separate fixed role sets scalar temperature."""
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
    x, y = np.asarray([r["x"] for r in fit]), np.asarray([r["y"] for r in fit])
    geometry = base.geometry(x, [r["source_cluster_id"] for r in fit])
    heads = []
    for arm in ARMS:
        old.progress("before_benchmark_8207_" + arm, len(heads), 3 - len(heads))
        solved = base.solve(design(arm, x, geometry), y, 0.01, time.monotonic() + 600)
        if not solved["converged"]:
            raise ValueError("fit_nonconvergence")
        z = design(arm, np.asarray([r["x"] for r in temp]), geometry) @ np.asarray(
            solved["weights"]
        )
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
                geometry=geometry,
                weights=solved["weights"],
                temperature=temperature,
                fit_ids=[r["unit_id"] for r in fit],
                temperature_ids=[r["unit_id"] for r in temp],
                solve_receipt=solved,
            )
        )
        old.progress("after_benchmark_8207_" + arm, len(heads), 3 - len(heads))
    return heads


def predict(head: Json, query: Json, baseline: Json) -> Json:
    """Compute the frozen baseline on the same input without opening its target."""
    if set(query) - {"unit_id", "source_cluster_id", "x", "historical_x"}:
        raise ValueError("evaluator_label")
    fp = None
    if query["historical_x"] is not None:
        phi = base.design("radial16", np.asarray([query["historical_x"]]), baseline["geometry"])[0]
        offset, slope = baseline["calibration"]
        fp = probability(
            0,
            -(
                offset
                + slope
                * math.fsum(
                    float(a) * float(b) for a, b in zip(phi, baseline["weights"], strict=True)
                )
            ),
            1,
        )
    baseline_action = base.action(fp)
    p = None
    if query["x"] is not None and fp is not None:
        phi = design(head["arm"], np.asarray([query["x"]]), head["geometry"])[0]
        z = math.fsum(float(a) * float(b) for a, b in zip(phi, head["weights"], strict=True))
        p = probability(0, -z, head["temperature"])
    return dict(
        unit_id=query["unit_id"],
        source_cluster_id=query["source_cluster_id"],
        arm=head["arm"],
        p=p,
        baseline_action=baseline_action,
        allowed_actions=[
            "reject",
            "escalate",
            *(["accept"] if baseline_action == "accept" else []),
        ],
        action=action(p, baseline_action),
    )


def select_simple(heads: list[Json], rows: list[Json], roles: Json, baseline: Json) -> Json:
    """Tune costs choose the comparator; reserved targets never enter this function."""
    ids = {r["unit_id"] for r in roles["calibration"]}
    costs = {}
    for head in heads:
        if head["arm"] != "energy":
            costs[head["arm"]] = float(
                np.mean(
                    [
                        base.loss(
                            predict(
                                head,
                                {
                                    k: r[k]
                                    for k in ("unit_id", "source_cluster_id", "x", "historical_x")
                                },
                                baseline,
                            )["action"],
                            r["y"],
                        )
                        for r in rows
                        if r["unit_id"] in ids
                    ]
                )
            )
    return dict(
        selected=min(costs, key=lambda a: (costs[a], a)), tune_costs=costs, tune_ids=sorted(ids)
    )
