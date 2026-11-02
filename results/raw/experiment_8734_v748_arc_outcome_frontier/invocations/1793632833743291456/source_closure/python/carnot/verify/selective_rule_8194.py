"""REQ-VERIFY-8194: prediction sets keep calibration separate from evaluation.

These finite-sample equations need class-conditional exchangeability for a
coverage guarantee. Reused exposed evidence does not establish that condition.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import math
from pathlib import Path
import time
from typing import Any
from unittest.mock import patch

import numpy as np
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]

from carnot.reporting.current_work_receipt import atomic_json
from carnot.verify import evidence_energy_8154 as base
from carnot.verify import sentence_decision_audit_8185 as audit

Json = dict[str, Any]
ARMS = [
    "local_set",
    "local_point",
    "radial_set",
    "radial_point",
    "frozen_v707_radial",
    "equivalent_logistic_set",
    "always_escalate",
]
SEED = 7088194


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush measured counts so actual fitting and child waits remain visible."""
    print(f"[exp8194] phase={phase} completed={completed} pending={pending}", flush=True)


def quantile(scores: list[float]) -> Json:
    """Use an exact order statistic; JSON null plus a flag represents infinity."""
    rank = math.ceil((len(scores) + 1) * 0.95)
    return dict(
        n=len(scores),
        rank=rank,
        threshold=sorted(scores)[rank - 1] if rank <= len(scores) else None,
        infinity=rank > len(scores),
    )


def prediction_set(p: float | None, qs: list[Json]) -> list[int]:
    """Retain threshold ties; missing evidence keeps both labels and escalates."""
    if p is None:
        return [0, 1]
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("probability")
    return [
        y for y, score in enumerate([p, 1 - p]) if qs[y]["infinity"] or score <= qs[y]["threshold"]
    ]


def logit_probability(logits: list[float], temperature: float) -> float:
    """Only the logit difference matters, so a common shift cannot add evidence."""
    if not 0.25 <= temperature <= 4:
        raise ValueError("temperature")
    z = (logits[1] - logits[0]) / temperature
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def freeze_roles(rows: list[Json], reserved: list[Json]) -> Json:
    """Assign roles from identity alone before reserved targets may be opened."""
    if any(r["role"] not in ("fit", "tune") for r in rows):
        raise ValueError("role")
    allowed = {
        "unit_id",
        "source_cluster_id",
        "role",
        "x",
        "y",
        "status",
        "exclusion_reason",
        "historical_paired_control",
        "slot",
        "source_id",
    }
    if any(set(r) - allowed for r in rows) or any(
        "y" in r or "evaluator_label" in r for r in reserved
    ):
        raise ValueError("evaluator_label")
    all_rows = [*rows, *reserved]
    if (
        len({r["source_cluster_id"] for r in all_rows}) != 320
        or len({r["unit_id"] for r in all_rows}) != 320
    ):
        raise ValueError("source_identity")
    fit = sorted(
        [r for r in rows if r["role"] == "fit"],
        key=lambda r: hashlib.sha256((r["source_id"] + "v708-fit-role").encode()).hexdigest(),
    )
    tune = sorted([r for r in rows if r["role"] == "tune"], key=lambda r: r["source_cluster_id"])
    if len(fit) != 128 or len(tune) != 64 or len(reserved) != 128:
        raise ValueError("role_count")
    return {
        role: [dict(unit_id=r["unit_id"], source_cluster_id=r["source_cluster_id"]) for r in rs]
        for role, rs in [
            ("head_fit", fit[:96]),
            ("temperature_fit", fit[96:]),
            ("calibration", tune),
            ("reserved", sorted(reserved, key=lambda r: r["source_cluster_id"])),
        ]
    }


def class_support(rows: list[Json], roles: Json) -> Json:
    """Missing source evidence remains a slot and never becomes a training row."""
    lookup = {r["source_cluster_id"]: r for r in rows}
    result = {}
    for role in ("head_fit", "temperature_fit", "calibration"):
        available = [
            lookup[r["source_cluster_id"]]
            for r in roles[role]
            if lookup[r["source_cluster_id"]]["x"] is not None
            and lookup[r["source_cluster_id"]]["y"] in (0, 1)
        ]
        result[role] = dict(
            intended=len(roles[role]),
            completed=len(available),
            classes={str(y): sum(r["y"] == y for r in available) for y in (0, 1)},
        )
    return result


def basis(head: Json, x: list[float]) -> base.Array:
    """The feature control uses its original twelve inputs and its own geometry."""
    return base.design(
        "radial16", np.asarray([x[: head["dimensions"]]], dtype=float), head["geometry"]
    )[0]


def probability(head: Json, x: list[float], *, scalar: bool = False) -> float:
    """Scalar summation independently checks vector prediction arithmetic."""
    phi = basis(head, x)
    z = (
        math.fsum(float(a) * float(b) for a, b in zip(phi, head["weights"], strict=True))
        if scalar
        else float(phi @ np.asarray(head["weights"]))
    )
    if scalar:
        return logit_probability([0, z], head["temperature"])
    scaled = z / head["temperature"]
    gibbs = np.exp(np.asarray([0.0, scaled]) - max(0.0, scaled))
    return float(gibbs[1] / gibbs.sum())


def train(rows: list[Json], roles: Json, raw: Path) -> Json:
    """Select weights on head folds, temperature on its role and thresholds last.

    Both heads use identical available head-fit sources. Calibration labels
    affect only order statistics; they cannot fit weights or temperatures.
    """
    lookup = {r["source_cluster_id"]: r for r in rows}
    selected = {
        role: [
            lookup[r["source_cluster_id"]]
            for r in roles[role]
            if lookup[r["source_cluster_id"]]["x"] is not None
            and lookup[r["source_cluster_id"]]["y"] in (0, 1)
        ]
        for role in ("head_fit", "temperature_fit", "calibration")
    }
    fit, temp, cal = (selected[k] for k in ("head_fit", "temperature_fit", "calibration"))
    ids = [r["source_cluster_id"] for r in fit]
    order = sorted(range(len(fit)), key=lambda i: hashlib.sha256(ids[i].encode()).hexdigest())
    folds = np.empty(len(fit), dtype=int)
    for rank, i in enumerate(order):
        folds[i] = rank % 4
    heads, records = [], []
    deadline = time.monotonic() + 600
    for arm, dimensions in [("local_evidence_radial16", 16), ("radial16", 12)]:
        progress("before_benchmark_head_" + arm, len(heads), 2 - len(heads))
        x = np.asarray([r["x"][:dimensions] for r in fit], dtype=float)
        y = np.asarray([r["y"] for r in fit], dtype=float)
        losses = {}
        for ridge in base.RIDGES:
            scores = []
            for fold in range(4):
                mask = folds != fold
                g = base.geometry(x[mask], [s for s, keep in zip(ids, mask, strict=True) if keep])
                solved = base.solve(base.design("radial16", x[mask], g), y[mask], ridge, deadline)
                if not solved["converged"]:
                    raise ValueError("fit_nonconvergence")
                z = base.design("radial16", x[~mask], g) @ np.asarray(solved["weights"])
                loss = float(np.mean(np.logaddexp(0, z) - y[~mask] * z))
                scores.append(loss)
                records.append(
                    dict(
                        solved,
                        arm=arm,
                        ridge=ridge,
                        fold=fold,
                        held_source_ids=[s for s, keep in zip(ids, ~mask, strict=True) if keep],
                        geometry=g,
                        log_loss=loss,
                    )
                )
                progress("head_fold_completed", len(records), 40 - len(records))
            losses[ridge] = float(np.mean(scores))
        g = base.geometry(x, ids)
        h = base.solve(base.design("radial16", x, g), y, base.choose_ridge(losses), deadline)
        if not h["converged"]:
            raise ValueError("full_fit_nonconvergence")
        h.update(
            arm=arm,
            dimensions=dimensions,
            geometry=g,
            head_fit_source_ids=ids,
            temperature_source_ids=[r["source_cluster_id"] for r in temp],
            calibration_source_ids=[r["source_cluster_id"] for r in cal],
            fit_fold_losses={str(k): v for k, v in losses.items()},
        )
        progress("after_benchmark_head_" + arm, len(heads) + 1, 1 - len(heads))
        progress("before_benchmark_temperature_" + arm)
        z = np.asarray([float(basis(h, r["x"]) @ np.asarray(h["weights"])) for r in temp])
        ty = np.asarray([r["y"] for r in temp])
        fitted = minimize_scalar(
            lambda t: float(np.mean(np.logaddexp(0, z / t) - ty * z / t)),
            bounds=(0.25, 4),
            method="bounded",
            options=dict(xatol=1e-12),
        )
        if not fitted.success:
            raise ValueError("temperature_nonconvergence")
        candidates = [0.25, float(fitted.x), 4.0]
        h["temperature"] = min(
            candidates, key=lambda t: (float(np.mean(np.logaddexp(0, z / t) - ty * z / t)), t)
        )
        h["temperature_receipt"] = dict(
            success=True,
            scalar_parameter_count=1,
            bounds=[0.25, 4],
            n=len(temp),
            nll=float(np.mean(np.logaddexp(0, z / h["temperature"]) - ty * z / h["temperature"])),
        )
        progress("after_benchmark_temperature_" + arm, 1)
        h["quantiles"] = [
            quantile(
                [
                    probability(h, r["x"]) if label == 0 else 1 - probability(h, r["x"])
                    for r in cal
                    if r["y"] == label
                ]
            )
            for label in (0, 1)
        ]
        heads.append(h)
    result = dict(heads=heads, fit_fold_rows=records)
    atomic_json(raw / "frozen_heads.json", result)
    return result


def predict(
    rows: list[Json], heads: list[Json], frozen: Json, *, scalar: bool = False
) -> list[Json]:
    """Seal all seven decisions without accepting evaluator labels as inputs."""
    records = []
    lookup = {h["arm"]: h for h in heads}
    for r in rows:
        if set(r) - {
            "unit_id",
            "source_cluster_id",
            "x",
            "historical_x",
            "status",
            "exclusion_reason",
        }:
            raise ValueError("evaluator_label")
        ps = {
            arm: probability(h, r["x"], scalar=scalar) if r["x"] is not None else None
            for arm, h in lookup.items()
        }
        lp = (
            probability(lookup["local_evidence_radial16"], r["x"], scalar=True)
            if r["x"] is not None
            else None
        )
        fp = None
        if r["historical_x"] is not None:
            phi = base.design("radial16", np.asarray([r["historical_x"]]), frozen["geometry"])[0]
            b, a = frozen["calibration"]
            z = b + a * math.fsum(
                float(x) * float(w) for x, w in zip(phi, frozen["weights"], strict=True)
            )
            fp = logit_probability([0, z], 1)
        for arm in ARMS:
            head = lookup[
                "local_evidence_radial16"
                if arm.startswith("local") or arm == "equivalent_logistic_set"
                else "radial16"
            ]
            p = (
                None
                if arm == "always_escalate"
                else fp
                if arm == "frozen_v707_radial"
                else lp
                if arm == "equivalent_logistic_set"
                else ps[head["arm"]]
            )
            labels = prediction_set(p, head["quantiles"]) if arm.endswith("set") else None
            action = (
                ("accept" if labels == [0] else "reject" if labels == [1] else "escalate")
                if labels is not None
                else base.action(p)
            )
            records.append(
                dict(
                    unit_id=r["unit_id"],
                    source_cluster_id=r["source_cluster_id"],
                    arm=arm,
                    condition="all_original_reserved_slots",
                    p=p,
                    prediction_set=labels,
                    action=action,
                    status="completed" if p is not None else "excluded",
                    exclusion_reason=None if p is not None else "missing_evidence",
                )
            )
    return records


def statistics(rows: list[Json]) -> Json:
    """Separate class coverage, action coverage and errors among accepted sources."""
    groups: Json = {}
    for r in rows:
        group = groups.setdefault(r["source_cluster_id"], {})
        if r["arm"] in group:
            raise ValueError("duplicate_source_arm")
        group[r["arm"]] = r
    paired = [g for g in groups.values() if set(g) == set(ARMS)]
    complete = [
        g
        for g in paired
        if all(
            g[a]["p"] is not None and g[a]["y"] in (0, 1)
            for a in ["local_set", "radial_set", "frozen_v707_radial"]
        )
    ]
    classes = Counter(g["local_set"]["y"] for g in complete)
    metrics = []
    for arm in ARMS:
        subset = [g[arm] for g in paired]
        accepts = [r for r in subset if r["action"] == "accept"]
        observed = [r for r in subset if r["p"] is not None]
        coverage = []
        for label in (0, 1):
            cr = [r for r in observed if r["y"] == label and r["prediction_set"] is not None]
            covered = sum(label in r["prediction_set"] for r in cr)
            coverage.append(
                dict(
                    label=label,
                    numerator=covered,
                    denominator=len(cr),
                    coverage=covered / len(cr) if cr else None,
                )
            )
        metrics.append(
            dict(
                arm=arm,
                count=len(subset),
                typed_cost=sum(r["numerator"] for r in subset) / len(subset) if subset else None,
                singleton_coverage=sum(r["action"] != "escalate" for r in subset) / len(subset)
                if subset
                else None,
                singleton_count=sum(r["action"] != "escalate" for r in subset),
                error_among_accepts=dict(
                    numerator=sum(r["y"] == 1 for r in accepts),
                    denominator=len(accepts),
                    rate=sum(r["y"] == 1 for r in accepts) / len(accepts) if accepts else None,
                ),
                class_coverage=coverage,
                brier=sum(r["brier"] for r in observed) / len(observed) if observed else None,
                brier_count=len(observed),
            )
        )
    gains = [g["frozen_v707_radial"]["numerator"] - g["local_set"]["numerator"] for g in paired]
    with patch.dict(audit.CONFIG, seed=SEED):
        interval = audit.interval(gains)
    improved = sum(v > 0 for v in gains)
    extra = sum(
        g["local_set"]["false_accept"] == 1 and g["frozen_v707_radial"]["false_accept"] == 0
        for g in paired
    )
    delta = (
        sum(g["local_set"]["brier"] - g["frozen_v707_radial"]["brier"] for g in complete)
        / len(complete)
        if complete
        else None
    )
    support = len(complete) >= 96 and all(classes[y] >= 12 for y in (0, 1))
    passed = bool(
        support
        and interval["valid_draws"] >= 9500
        and interval["lower_one_sided_975"] > 0.02
        and improved >= 5
        and extra == 0
        and delta is not None
        and delta <= 0.01
    )
    contrasts = []
    for treatment, control in [
        ("local_set", "local_point"),
        ("radial_set", "radial_point"),
        ("local_set", "radial_set"),
        ("local_point", "radial_point"),
        ("equivalent_logistic_set", "local_set"),
        ("local_set", "always_escalate"),
    ]:
        contrasts.append(
            dict(
                treatment=treatment,
                comparator=control,
                primary=False,
                mean_cost_gain=sum(
                    g[control]["numerator"] - g[treatment]["numerator"] for g in paired
                )
                / len(paired)
                if paired
                else None,
            )
        )
    return dict(
        eligible_count=len(complete),
        class_support_reserved={str(y): classes[y] for y in (0, 1)},
        support_sufficient=support,
        all_slot_metrics=metrics,
        secondary_contrasts=contrasts,
        H1=dict(
            treatment="local_set",
            comparator="frozen_v707_radial",
            alpha=0.025,
            all_slot_count=len(paired),
            interval=interval,
            improved_sources=improved,
            extra_false_accepts=extra,
            brier_increase=delta,
            brier_pair_count=len(complete),
            support_sufficient=support,
            passed=passed,
        ),
        h1_development_signal_score=int(passed),
    )
