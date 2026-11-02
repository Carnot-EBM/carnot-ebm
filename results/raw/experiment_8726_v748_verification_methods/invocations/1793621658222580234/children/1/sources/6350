"""REQ-REPORT-8432: reduce fixed decisions without treating siblings as independent.

Losses average over seeds and answers before whole-question resampling. Missing
probabilities remain visible in intended bounds rather than improving scores.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, cast

import numpy as np

from carnot.verify.v726_criterion_energy_fit import ARMS as ARMS, SEEDS as SEEDS

Json = dict[str, Any]
RANDOM_SEED, DRAWS = 7258419, 10000
CONFIG = dict(
    accept_max=0.2,
    reject_min=0.8,
    escalation_loss=0.2,
    bootstrap_draws=DRAWS,
    bootstrap_seed=RANDOM_SEED,
    calibration_edges=list(np.linspace(0, 1, 11)),
)


def action(p: float) -> str:
    """Fixed boundaries prevent evaluator outcomes from choosing a better cutoff."""
    return "accept" if p <= 0.2 else "reject" if p >= 0.8 else "escalate"


def loss(p: float, y: int) -> float:
    """Escalation spends a fixed review cost; definitive mistakes spend one unit."""
    decision = action(p)
    return 0.2 if decision == "escalate" else float((decision == "reject") != bool(y))


def summary(rows: list[Json], target: str) -> Json:
    """Bound unknown predictions while distinguishing absent sensitivity labels."""
    observed = [r for r in rows if r.get(target) in (0, 1)]
    valid = [r for r in observed if r["probability"] is not None]
    n, total = len(valid), len(rows) if target == "y" else len(observed)
    errors = sum(loss(r["probability"], r[target]) for r in valid)
    brier_sum = sum((r["probability"] - r[target]) ** 2 for r in valid)
    definitive = [r for r in valid if action(r["probability"]) != "escalate"]
    accepts = [r for r in definitive if action(r["probability"]) == "accept"]
    false_accepts = sum(r[target] == 1 for r in accepts)
    misses = total - n
    ratios = dict(
        brier=(brier_sum, n),
        log_loss=(
            sum(
                -math.log(
                    max(1e-15, min(1 - 1e-15, r["probability"]))
                    if r[target]
                    else max(1e-15, 1 - min(1 - 1e-15, r["probability"]))
                )
                for r in valid
            ),
            n,
        ),
        decision_loss=(errors, n),
        coverage=(len(definitive), total),
        selective_risk=(
            sum(loss(r["probability"], r[target]) for r in definitive),
            len(definitive),
        ),
        false_accept_rate=(false_accepts, len(accepts)),
    )
    metrics = {k: a / b if b else None for k, (a, b) in ratios.items()}
    calibration = []
    for bin_id in range(10):
        members = [r for r in valid if min(9, int(r["probability"] * 10)) == bin_id]
        calibration.append(
            dict(
                lower=bin_id / 10,
                upper=(bin_id + 1) / 10,
                count=len(members),
                mean_probability=sum(r["probability"] for r in members) / len(members)
                if members
                else None,
                observed_frequency=sum(r[target] for r in members) / len(members)
                if members
                else None,
                undefined_reason=None if members else "empty_fixed_bin",
            )
        )
    return dict(
        **metrics,
        intended_prediction_count=len(rows),
        observed_target_count=len(observed),
        observed_answer_count=len({r["unit_id"] for r in observed}),
        valid_prediction_count=n,
        missing_prediction_count=sum(
            r["probability"] is None for r in (rows if target == "y" else observed)
        ),
        missing_target_count=len(rows) - len(observed),
        false_accepts=false_accepts,
        definitive_count=len(definitive),
        accepted_count=len(accepts),
        raw_operands={k: dict(numerator=a, denominator=b) for k, (a, b) in ratios.items()},
        undefined_reasons={
            k: "zero_valid_predictions"
            if k in {"brier", "log_loss", "decision_loss"}
            else "zero_observed_targets"
            if k == "coverage"
            else "zero_definitive_decisions"
            if k == "selective_risk"
            else "zero_accepts"
            for k, (_, b) in ratios.items()
            if not b
        },
        bounds=dict(
            brier=[brier_sum / total, (brier_sum + misses) / total] if total else None,
            decision_loss=[errors / total, (errors + misses) / total] if total else None,
            missing_prediction_count=misses,
            denominator=total,
            reason=None if total else "zero_observed_targets",
        ),
        calibration=calibration,
    )


def contrast(rows: list[Json], target: str) -> Json:
    """Use the same observed siblings and all seeds in each paired question."""
    groups: dict[str, list[Json]] = defaultdict(list)
    for row in rows:
        groups[row["question_hash"]].append(row)
    paired, omitted = [], []
    for q, group in sorted(groups.items()):
        arms = {
            a: [r for r in group if r["arm"] == a and r.get(target) in (0, 1)]
            for a in ("logistic9", "spline9")
        }
        keys = [{(r["unit_id"], r["seed"]) for r in arms[a]} for a in arms]
        if (
            not keys[0]
            or keys[0] != keys[1]
            or any(r["probability"] is None for a in arms for r in arms[a])
        ):
            omitted.append(
                dict(question_hash=q, reason="missing_paired_probability_or_observed_target")
            )
            continue
        scores = {a: summary(arms[a], target) for a in arms}
        paired.append(
            dict(
                question_hash=q,
                brier_gain=scores["logistic9"]["brier"] - scores["spline9"]["brier"],
                decision_loss_delta=scores["spline9"]["decision_loss"]
                - scores["logistic9"]["decision_loss"],
            )
        )
    if not paired:
        return dict(
            gain=None,
            ci95=None,
            decision_delta=None,
            independent_questions=0,
            draws=DRAWS,
            seed=RANDOM_SEED,
            conditional_reason="zero_complete_paired_questions",
            paired_question_rows=[],
            omitted_questions=omitted,
        )
    vectors = np.asarray([r["brier_gain"] for r in paired])
    rng = np.random.default_rng(RANDOM_SEED)
    draws = []
    for completed in range(0, DRAWS, 1000):
        print(
            f"[exp8432] phase=bootstrap_{target} completed={completed} pending={DRAWS - completed}",
            flush=True,
        )
        draws.extend(
            vectors[rng.integers(0, len(vectors), size=(1000, len(vectors)))].mean(axis=1).tolist()
        )
    return dict(
        gain=float(vectors.mean()),
        ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
        decision_delta=float(np.mean([r["decision_loss_delta"] for r in paired])),
        independent_questions=len(paired),
        draws=DRAWS,
        seed=RANDOM_SEED,
        conditional_reason="conditional_on_observed_labels_and_complete_paired_questions",
        paired_question_rows=paired,
        omitted_questions=omitted,
    )


def evaluate(predictions: list[Json], targets: list[Json]) -> Json:
    """Keep every intended arm and seed, plus fixed reporting strata and controls."""
    target_map = {r["unit_id"]: r for r in targets}
    keys = {(r["unit_id"], r["arm"], r["seed"]) for r in predictions}
    if len(keys) != len(predictions) or len(target_map) != len(targets):
        raise ValueError("duplicate_prediction_or_target")
    rows = []
    for row in predictions:
        p = row["probability"]
        if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError("invalid_probability")
        label = target_map.get(row["unit_id"], {})
        y = label.get("y")
        missing = row.get("missing_reason") or ("A1_target_missing" if y not in (0, 1) else None)
        valid = p is not None and y in (0, 1)
        rows.append(
            dict(
                row,
                y=y,
                y_sensitivity=label.get("y_sensitivity"),
                family=row["question_hash"],
                decision=action(p) if p is not None else None,
                status="completed" if valid else "censored",
                censored=not valid,
                excluded=False,
                missing_reason=missing,
                absolute_metric=(p - y) ** 2 if valid else None,
                raw_numerator=(p - y) ** 2 if valid else None,
                raw_denominator=1,
                metric=dict(brier=(p - y) ** 2, decision_loss=loss(p, cast(int, y)))
                if valid
                else None,
            )
        )
    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}
    metrics = {
        target: {a: summary(rs, field) for a, rs in by_arm.items()}
        for target, field in [("A1", "y"), ("A2", "y_sensitivity")]
    }
    seeds = {
        str(seed): {
            a: summary([r for r in rs if r["seed"] == seed], "y") for a, rs in by_arm.items()
        }
        for seed in SEEDS
    }
    strata = {}
    for fields in [("dataset",), ("generator",), ("dataset", "generator")]:
        roster = sorted({tuple(r[k] for k in fields) for r in rows})
        for values in roster:
            selected = [r for r in rows if tuple(r[k] for k in fields) == values]
            key = "|".join(f"{k}={v}" for k, v in zip(fields, values, strict=True))
            strata[key] = dict(
                roster=dict(zip(fields, values, strict=True)),
                independent_questions=len({r["question_hash"] for r in selected}),
                metrics={
                    target: {
                        a: summary([r for r in selected if r["arm"] == a], field) for a in ARMS
                    }
                    for target, field in [("A1", "y"), ("A2", "y_sensitivity")]
                },
                contrast=contrast(selected, "y"),
            )
    primary, sensitivity = contrast(rows, "y"), contrast(rows, "y_sensitivity")
    control = {
        (r["unit_id"], r["seed"]): r["probability"] for r in by_arm["source_permuted_spline9"]
    }
    differences = [
        abs(r["probability"] - control[(r["unit_id"], r["seed"])])
        for r in by_arm["spline9"]
        if r["probability"] is not None and control.get((r["unit_id"], r["seed"])) is not None
    ]
    inert = bool(differences) and max(differences) == 0
    exploratory = bool(
        primary["ci95"] and primary["ci95"][0] > 0 and primary["decision_delta"] <= 0 and not inert
    )
    return dict(
        rows=rows,
        A1_metrics=metrics["A1"],
        A2_sensitivity=metrics["A2"],
        per_seed_metrics=seeds,
        brier_gain_vs_logistic=primary["gain"],
        brier_gain_ci95=primary["ci95"],
        decision_loss_delta=primary["decision_delta"],
        exploratory_benefit_score=int(exploratory),
        bootstrap=primary,
        sensitivity_bootstrap=sensitivity,
        per_stratum_metrics=strata,
        confidence_calibration={a: s["calibration"] for a, s in metrics["A1"].items()},
        risk_coverage={
            a: {
                k: s[k]
                for k in ["coverage", "selective_risk", "false_accepts", "undefined_reasons"]
            }
            for a, s in metrics["A1"].items()
        },
        missing_worst_case_bounds={
            t: {a: s["bounds"] for a, s in m.items()} for t, m in metrics.items()
        },
        negative_control_degeneracy=dict(
            identical_predictions=inert,
            paired_count=len(differences),
            maximum_absolute_prediction_difference=max(differences) if differences else None,
            explanation="Source permutation is inert on these cached features/heads; identical predictions supply no control benefit."
            if inert
            else "Source-permuted predictions differ; the control remains descriptive without independent factuality credit.",
        ),
    )
