"""REQ-VERIFY-7997: targets enter only after public predictions are sealed.

Primitive decisions keep failed and unknown slots. Independent source groups
determine uncertainty; seeds do not increase the effective sample size.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from carnot.verify import typed_development_7997 as m
from carnot.verify.qwen_energy_calibration_7972 import holm

Json = dict[str, Any]


def targets_by_id(rows: list[Json]) -> Json:
    """Missing target fields are contract errors rather than negative labels."""
    result = {}
    for r in rows:
        if (
            set(r) != {"family_id", "y"}
            or r["family_id"] in result
            or (r["y"] is not None and (type(r["y"]) is not int or r["y"] not in (0, 1)))
        ):
            raise ValueError("target_contract")
        result[r["family_id"]] = r["y"]
    return result


def calibrate(rows: list[Json], targets: list[Json]) -> tuple[Json, list[Json]]:
    """One temperature per arm uses primary-seed Brier and a fixed tie order."""
    labels = targets_by_id(targets)
    policies, scores = {}, []
    for arm in m.ARMS:
        choices = []
        for order, t in enumerate(m.CONFIG["temperatures"]):
            eligible = [
                r
                for r in rows
                if r["arm"] == arm
                and r["primary"]
                and r["temperature"] == t
                and r["probability"] is not None
                and labels.get(r["family_id"]) is not None
            ]
            if not eligible:
                raise ValueError("calibration_support")
            numerator = sum((r["probability"] - labels[r["family_id"]]) ** 2 for r in eligible)
            choices.append((numerator / len(eligible), order, t))
            scores.append(
                dict(
                    arm=arm,
                    temperature=t,
                    numerator=numerator,
                    denominator=len(eligible),
                    eligibility=True,
                    failure_status=False,
                    censor_status=False,
                )
            )
        loss, _, t = min(choices)
        policies[arm] = dict(
            temperature=t,
            calibration_brier=loss,
            accept_below=0.05,
            reject_above=0.75,
            ties="escalate",
            coefficient_refit_steps=0,
        )
    return policies, scores


def uncertainty(diff: NDArray[np.float64]) -> Json:
    """Paired bootstrap and sign tests use the same independently grouped deltas."""
    if not len(diff):
        return dict(gain=0.0, interval=[None, None], raw_p=1.0, independent=0)
    rng = np.random.default_rng(m.CONFIG["random_seed"])
    boot, extreme = [], 0
    for begin in range(0, 10000, 1000):
        boot.extend(diff[rng.integers(0, len(diff), (1000, len(diff)))].mean(axis=1).tolist())
        signs = rng.choice([-1, 1], (1000, len(diff)))
        extreme += int(np.sum((diff * signs).mean(axis=1) >= diff.mean()))
        print(f"[exp7997] paired_draws={begin + 1000}/10000 groups={len(diff)}", flush=True)
    return dict(
        gain=float(diff.mean()),
        interval=np.quantile(boot, [0.025, 0.975]).tolist(),
        raw_p=(1 + extreme) / 10001,
        independent=len(diff),
    )


def benefit(summary: Json, comparisons: Json, support: Json) -> bool:
    """Every registered gate is necessary, including Brier noninferiority."""
    return bool(
        support["passed"]
        and summary["spline"]["automation"] >= 0.5
        and all(
            comparisons[a]["gain"] >= 0.02
            and comparisons[a]["interval"][0] > 0
            and comparisons[a]["adjusted_p"] < 0.05
            and comparisons[a]["brier_degradation_interval"][1] <= 0.01
            and summary["spline"]["false_accepts"] <= summary[a]["false_accepts"]
            for a in ("logistic", "mlp")
        )
    )


def evaluate(predictions: list[Json], targets: list[Json]) -> Json:
    """Reduce primary-seed costs while retaining all seed rows for inspection."""
    labels = targets_by_id(targets)
    if set(labels) != {r["family_id"] for r in predictions}:
        raise ValueError("target_roster")
    rows, summary, bounds = [], {}, {}
    costs: Json = {a: defaultdict(list) for a in m.ARMS}
    briers: Json = {a: defaultdict(list) for a in m.ARMS}
    for r in predictions:
        y, p = labels[r["family_id"]], r["probability"]
        decision = m.action(p)
        cost = (
            None
            if y is None
            else (0.25 if decision == "escalate" else (5 * y if decision == "accept" else 1 - y))
        )
        brier = None if y is None or p is None else (p - y) ** 2
        row = dict(
            r,
            y=y,
            decision=decision,
            actual_cost=cost,
            brier=brier,
            numerator=cost,
            denominator=int(y is not None),
            eligibility=p is not None and y is not None,
            cost_lower=0.0 if y is None else cost,
            cost_upper=5.0 if y is None else cost,
        )
        rows.append(row)
        if r["primary"] and cost is not None:
            costs[r["arm"]][r["source_cluster_id"]].append(cost)
            if brier is not None:
                briers[r["arm"]][r["source_cluster_id"]].append(brier)
    primary = [r for r in rows if r["primary"]]
    for arm in m.ARMS:
        selected = [r for r in primary if r["arm"] == arm]
        eligible = [r for r in selected if r["eligibility"]]
        n = len(selected)
        known = [r for r in selected if r["y"] is not None]
        ys, ps = [r["y"] for r in eligible], [r["probability"] for r in eligible]
        reliability = []
        for i in range(10):
            members = [r for r in eligible if min(9, int(r["probability"] * 10)) == i]
            reliability.append(
                dict(
                    bin=i,
                    denominator=len(members),
                    mean_probability=float(np.mean([r["probability"] for r in members]))
                    if members
                    else None,
                    observed_frequency=float(np.mean([r["y"] for r in members]))
                    if members
                    else None,
                )
            )
        summary[arm] = dict(
            automation=sum(r["decision"] != "escalate" for r in selected) / n if n else 0.0,
            false_accepts=sum(r["decision"] == "accept" and r["y"] == 1 for r in known),
            intended_denominator=n,
            probability_denominator=len(eligible),
            known_cost=float(np.mean([r["actual_cost"] for r in known])) if known else None,
            brier=float(np.mean([r["brier"] for r in eligible])) if eligible else None,
            auroc=float(roc_auc_score(ys, ps)) if len(set(ys)) == 2 else None,
            reliability=reliability,
            descriptive=True,
        )
        bounds[arm] = dict(
            lower=sum(r["cost_lower"] for r in selected) / n if n else 0.0,
            upper=sum(r["cost_upper"] for r in selected) / n if n else 5.0,
            unknown_labels=n - len(known),
            intended=n,
            unknown_slot_bounds=[0, 5],
        )
    units = [r for r in primary if r["arm"] == "spline"]
    usable = [r for r in units if r["eligibility"]]
    class_counts = {
        str(y): len({r["source_cluster_id"] for r in usable if r["y"] == y}) for y in (0, 1)
    }
    independent = len({r["source_cluster_id"] for r in usable})
    support = dict(
        independent=independent,
        class_counts=class_counts,
        minimum=192,
        per_class=20,
        passed=independent >= 192 and min(class_counts.values()) >= 20,
    )
    comparisons = {}
    for arm in ("logistic", "mlp", "scalar", "raw_q"):
        ids = sorted(set(costs[arm]) & set(costs["spline"]))
        delta = np.array([np.mean(costs[arm][i]) - np.mean(costs["spline"][i]) for i in ids])
        comparison = uncertainty(delta)
        bids = sorted(set(briers[arm]) & set(briers["spline"]))
        degradation = np.array(
            [np.mean(briers["spline"][i]) - np.mean(briers[arm][i]) for i in bids]
        )
        bd = uncertainty(degradation)
        comparison.update(
            brier_degradation=bd["gain"],
            brier_degradation_interval=bd["interval"],
            secondary=arm not in ("logistic", "mlp"),
        )
        comparisons[arm] = comparison
    adjusted = holm({a: comparisons[a]["raw_p"] for a in ("logistic", "mlp")})
    for arm, p in adjusted.items():
        comparisons[arm]["adjusted_p"] = p
    identity_error = max(
        (
            abs(r["probability"] - r["identity_probability"])
            for r in rows
            if r["arm"] == "spline" and r["probability"] is not None
        ),
        default=0.0,
    )
    headroom = min(
        sum(r["actual_cost"] for r in primary if r["arm"] == a and r["actual_cost"] is not None)
        for a in ("logistic", "mlp")
    )
    independent_totals = {}
    for arm in m.ARMS:
        chosen = [r for r in rows if r["arm"] == arm and r["primary"]]
        totals = dict(
            denominator=len(chosen),
            numerator=sum(r["actual_cost"] for r in chosen if r["actual_cost"] is not None),
            known_denominator=sum(r["y"] is not None for r in chosen),
        )
        expected = summary[arm]["known_cost"]
        totals["passed"] = totals["denominator"] == summary[arm]["intended_denominator"] and (
            expected is None
            or abs(totals["numerator"] / totals["known_denominator"] - expected) < 1e-12
        )
        independent_totals[arm] = totals
    return dict(
        rows=rows,
        summary=summary,
        all_intended_bounds=bounds,
        paired_comparisons=comparisons,
        adjusted_p_values=adjusted,
        confidence_intervals={a: c["interval"] for a, c in comparisons.items()},
        evaluation_support=support,
        benefit=benefit(summary, comparisons, support),
        genuine_headroom=headroom > 0,
        independently_reduced_totals=independent_totals,
        equivalent_classifier_identity=dict(
            max_absolute_error=identity_error, passed=identity_error <= 1e-10
        ),
        sample_size_budget=dict(
            intended=len(units),
            eligible=len(usable),
            started=sum(r["status"] != "excluded" for r in units),
            completed=sum(r["probability"] is not None for r in units),
            excluded=sum(r["status"] == "excluded" for r in units),
            failed=sum(r["failure_status"] for r in units),
            censored=sum(r["censor_status"] for r in units),
            independent=independent,
            seeds_are_independent=False,
        ),
    )


def controls() -> Json:
    """Oracle fixtures exercise registered gates without becoming natural evidence."""
    outcomes = {}
    for name in ("known_benefit", "no_headroom"):
        predictions, targets = [], []
        for i in range(256):
            y = i % 2
            targets.append(dict(family_id=f"oracle-{i}", y=y))
            for arm in m.ARMS:
                p = 0.001 if y == 0 else 0.999
                if name == "known_benefit" and arm != "spline":
                    p = 0.5
                predictions.append(
                    dict(
                        family_id=f"oracle-{i}",
                        source_cluster_id=f"oracle-{i}",
                        arm=arm,
                        seed=17,
                        primary=True,
                        probability=p,
                        identity_probability=p if arm == "spline" else None,
                        failure_status=False,
                        censor_status=False,
                        status="completed",
                    )
                )
        result = evaluate(predictions, targets)
        outcomes[name] = dict(
            benefit=result["benefit"],
            genuine_headroom=result["genuine_headroom"],
            verdict_class="circular_positive",
            verifier_is_oracle=True,
            oracle_targets_separate=True,
            independent_natural_evidence=False,
            comparisons=result["paired_comparisons"],
        )
    outcomes["passed"] = bool(
        outcomes["known_benefit"]["benefit"]
        and outcomes["known_benefit"]["genuine_headroom"]
        and not outcomes["no_headroom"]["benefit"]
        and not outcomes["no_headroom"]["genuine_headroom"]
    )
    return outcomes
