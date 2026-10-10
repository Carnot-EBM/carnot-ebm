"""REQ-VERIFY-8350: source-level costs preserve missing evidence and fixed arms."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.verify import sentence_spline_fit_8334 as fitted
from carnot.verify.reserved_prediction_seal_8335 import ARMS

Json = dict[str, Any]
CONFIG = dict(
    seed=7178311,
    draws=10000,
    alpha=0.025,
    intended=128,
    minimum_sources=80,
    minimum_per_class=8,
    mean_gain_min=0.02,
    lower_bound_exclusive=0,
    brier_degradation_max=0.01,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts expose real work without adding a synthetic time floor."""
    print(f"[exp8350] phase={phase} completed={completed} pending={pending}", flush=True)


def cost(action: str, y: int | None) -> float | None:
    """Unknown truth cannot establish correctness, but escalation has fixed cost."""
    if action == "escalate":
        return 0.5
    return float((action == "reject") != bool(y)) if y in (0, 1) else None


def interval(bounds: list[tuple[float, float]]) -> Json:
    """Paired endpoints share each source draw so unknown labels stay bounded."""
    result: Json = dict(
        requested_draws=10000,
        valid_draws=0,
        random_seed=7178311,
        alpha=0.025,
        cluster_unit="original_source",
        mean_gain=None,
        mean_gain_lower=None,
        mean_gain_upper=None,
        lower_one_sided_975=None,
        scope="descriptive_exposed_development_not_confirmatory",
    )
    if bounds:
        progress("before_benchmark_cluster_bootstrap", 0, 10000)
        values = np.asarray(bounds)
        draws = np.random.default_rng(7178311).integers(0, len(bounds), (10000, len(bounds)))
        means = values[draws].mean(axis=1)
        known = bool(np.all(values[:, 0] == values[:, 1]))
        result.update(
            valid_draws=10000,
            source_count=len(bounds),
            mean_gain=float(values[:, 0].mean()) if known else None,
            mean_gain_lower=float(values[:, 0].mean()),
            mean_gain_upper=float(values[:, 1].mean()),
            lower_one_sided_975=float(np.quantile(means[:, 0], 0.025)),
            upper_descriptive_975=float(np.quantile(means[:, 1], 0.975)),
            bootstrap_checksum=canonical_hash(means.tolist()),
        )
        progress("after_benchmark_cluster_bootstrap", 10000, 0)
    return result


def diagnostic(predictions: list[Json]) -> list[Json]:
    """First96 action margins contain no utility and cannot stand in for full H1."""
    rows = []
    for slot in range(1, 97):
        selected = [p for p in predictions if p["slot"] == slot]
        rows.append(
            dict(
                slot=slot,
                predictor_only=True,
                action_counts=dict(Counter(p["action"] for p in selected)),
                margins=[
                    dict(
                        arm=p["arm"],
                        p=p["p"],
                        action=p["action"],
                        boundary_margin=min(abs(p["p"] - 0.25), abs(p["p"] - 0.75))
                        if p["p"] is not None
                        else None,
                    )
                    for p in selected
                ],
            )
        )
    return rows


def reduce(predictions: list[Json], targets: list[Json], comparator: str, optimizer: Json) -> Json:
    """Join original units once and compute costs rather than importing any mean."""
    labels = {t["unit_id"]: t for t in targets}
    groups: dict[str, Json] = {}
    for p in predictions:
        group = groups.setdefault(p["unit_id"], {})
        if p["arm"] in group or p["action"] != fitted.action(p["p"]):
            raise ValueError("duplicate_arm_or_action")
        group[p["arm"]] = p
    if (
        len(groups) != 128
        or len(labels) != 128
        or len(targets) != 128
        or set(groups) != set(labels)
        or comparator not in fitted.ARMS[1:]
        or any(set(g) != set(ARMS) for g in groups.values())
    ):
        raise ValueError("original128_join")
    paired, rows, complete = [], [], []
    seen = set()
    for unit, group in groups.items():
        t, c, s = labels[unit], group[comparator], group["spline34"]
        if (
            t["y"] not in (0, 1, None)
            or t["source_cluster_id"] in seen
            or any(
                p["source_cluster_id"] != t["source_cluster_id"] or p["slot"] != t["slot"]
                for p in group.values()
            )
        ):
            raise ValueError("source_target_identity")
        seen.add(t["source_cluster_id"])
        known = t["y"] in (0, 1)
        gains = [
            float(cost(c["action"], y)) - float(cost(s["action"], y))
            for y in ([t["y"]] if known else [0, 1])
        ]
        qualified = known and all(p["p"] is not None for p in group.values())
        pair = dict(
            slot=t["slot"],
            unit_id=unit,
            source_cluster_id=t["source_cluster_id"],
            y=t["y"],
            qualified=qualified,
            comparator_id=comparator,
            comparator_cost=cost(c["action"], t["y"]),
            spline_cost=cost(s["action"], t["y"]),
            gain_lower=min(gains),
            gain_upper=max(gains),
            changed_decision=c["action"] != s["action"],
        )
        paired.append(pair)
        if qualified:
            complete.append(pair)
        for arm, p in group.items():
            costs = [float(cost(p["action"], y)) for y in ([t["y"]] if known else [0, 1])]
            clipped = min(1 - 1e-12, max(1e-12, p["p"])) if p["p"] is not None else None
            rows.append(
                dict(
                    p,
                    y=t["y"],
                    qualified=qualified,
                    cost=cost(p["action"], t["y"]),
                    cost_lower=min(costs),
                    cost_upper=max(costs),
                    brier=(p["p"] - t["y"]) ** 2 if qualified else None,
                    log_loss=-t["y"] * math.log(clipped) - (1 - t["y"]) * math.log1p(-clipped)
                    if qualified
                    else None,
                )
            )
    return finish(rows, paired, complete, comparator, optimizer)


def finish(
    rows: list[Json], paired: list[Json], complete: list[Json], comparator: str, optimizer: Json
) -> Json:
    """Readiness and scientific support are different from a favorable result."""

    def summary(selected: list[Json]) -> Json:
        known_costs = [r["cost"] for r in selected if r["cost"] is not None]
        qualified = [r for r in selected if r["qualified"]]
        decisions = [r for r in qualified if r["action"] != "escalate"]
        return dict(
            intended_count=len(selected),
            cost_known_count=len(known_costs),
            cost_mean=sum(known_costs) / len(selected)
            if len(known_costs) == len(selected) and selected
            else None,
            cost_mean_lower=sum(r["cost_lower"] for r in selected) / len(selected)
            if selected
            else None,
            cost_mean_upper=sum(r["cost_upper"] for r in selected) / len(selected)
            if selected
            else None,
            qualified_count=len(qualified),
            probability_denominator=len(qualified),
            brier=sum(r["brier"] for r in qualified) / len(qualified) if qualified else None,
            log_loss=sum(r["log_loss"] for r in qualified) / len(qualified) if qualified else None,
            decision_count=len(decisions),
            achieved_coverage=len(decisions) / len(qualified) if qualified else None,
            risk_at_achieved_coverage=sum(r["cost"] for r in decisions) / len(decisions)
            if decisions
            else None,
            action_counts=dict(Counter(r["action"] for r in selected)),
        )

    arms = [
        dict(
            arm=arm,
            all_intended=summary([r for r in rows if r["arm"] == arm]),
            complete_case=summary([r for r in rows if r["arm"] == arm and r["qualified"]]),
        )
        for arm in ARMS
    ]
    lookup = {a["arm"]: a for a in arms}
    boot = dict(
        all_intended=interval([(r["gain_lower"], r["gain_upper"]) for r in paired]),
        complete_case=interval([(r["gain_lower"], r["gain_upper"]) for r in complete]),
    )
    counts = Counter(r["y"] for r in complete)
    control_rows = [
        dict(action=a, y=y, observed=cost(a, y), expected=expected)
        for a, y, expected in [
            ("accept", 0, 0),
            ("accept", 1, 1),
            ("reject", 0, 1),
            ("reject", 1, 0),
            ("escalate", 0, 0.5),
            ("escalate", 1, 0.5),
        ]
    ]
    control = dict(
        passed=all(r["observed"] == r["expected"] for r in control_rows),
        verdict_class="circular_positive",
        rows=control_rows,
    )
    headroom = sum(r["comparator_cost"] for r in complete) / len(complete) if complete else None
    support = len(complete) >= 80 and min(counts[0], counts[1]) >= 8
    informative = bool(support and control["passed"] and headroom is not None and headroom > 0)
    s, c = lookup["spline34"]["complete_case"], lookup[comparator]["complete_case"]
    degradation = s["brier"] - c["brier"] if complete else None
    b = boot["all_intended"]
    signal = int(
        informative
        and b["mean_gain"] is not None
        and b["mean_gain"] >= 0.02
        and b["lower_one_sided_975"] > 0
        and degradation <= 0.01
    )
    scope = (
        "development_signal"
        if signal
        else "null_insufficient_support"
        if not informative
        else "null_optimization_limited"
        if not optimizer.get("passed") or not optimizer.get("geometry_qualified")
        else "null_informative"
    )
    return dict(
        rows=rows,
        paired_cost_rows=paired,
        bootstrap_summary=boot,
        missing_bounds=[r for r in paired if r["y"] is None],
        arm_results=arms,
        qualified_count=len(complete),
        class_support={str(y): counts[y] for y in (0, 1)},
        support_sufficient=support,
        typed_action_control=control,
        permissible_action_oracle=dict(
            headroom=headroom,
            verdict_class="circular_positive",
            qualified_denominator=len(complete),
            minimum_cost=0,
        ),
        changed_decision_count=sum(r["changed_decision"] for r in paired),
        changed_decisions_by_arm={
            arm: sum(
                a["action"] != b["action"]
                for a, b in zip(
                    [r for r in rows if r["arm"] == arm],
                    [r for r in rows if r["arm"] == "spline34"],
                    strict=True,
                )
            )
            for arm in ARMS
        },
        brier_degradation=degradation,
        procedure_null_informative=informative,
        representation_null_informative=bool(
            informative and optimizer.get("passed") and optimizer.get("geometry_qualified")
        ),
        optimizer_qualification=optimizer,
        science_disposition=scope,
        h1_development_signal_score=signal,
        equal_input_scope="Frozen spline34 versus RBF34 training procedures; geometry needs optimizer qualification.",
        sigmoid34_parity=all(
            a["p"] == b["p"] and a["action"] == b["action"]
            for a, b in zip(
                [r for r in rows if r["arm"] == "spline34"],
                [r for r in rows if r["arm"] == "sigmoid34"],
                strict=True,
            )
        ),
    )
