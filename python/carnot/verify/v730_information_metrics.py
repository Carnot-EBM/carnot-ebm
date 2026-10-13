"""REQ-REPORT-8488: pair whole questions without fitting a curve to test labels.

Shipped loss functions retain their fixed thresholds. The two intervals correct
for the registered family of contrasts, not for an estimated population model.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product
import math
from typing import Any

import numpy as np

from carnot.verify import v726_decision_metrics as shipped
from carnot.verify.v730_source_control_fit import ARMS as ARMS, SEEDS as SEEDS
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
SCALAR, FULL, PERMUTED = "scalar_cubic_spline", "full_spline9", "source_permuted_spline9"
RANDOM_SEED, DRAWS = 7278446, 10000
CONFIG = dict(
    shipped.CONFIG,
    bootstrap_seed=RANDOM_SEED,
    confidence_level=0.975,
    contrasts=[SCALAR, PERMUTED],
    family_size=2,
    population_grouping_loss_estimate=False,
)


def contrast(rows: list[Json], target: str) -> Json:
    """Average the same siblings and seeds before shared whole-question draws."""
    groups: dict[str, list[Json]] = defaultdict(list)
    for row in rows:
        groups[row["question_hash"]].append(row)
    paired: list[Json] = []
    omitted: list[Json] = []
    for question, group in sorted(groups.items()):
        arms = {
            a: [r for r in group if r["arm"] == a and r.get(target) in (0, 1)]
            for a in (SCALAR, PERMUTED, FULL)
        }
        keys = [{(r["unit_id"], r["seed"]) for r in arms[a]} for a in arms]
        if (
            not keys[0]
            or any(k != keys[0] for k in keys)
            or any(r["probability"] is None for a in arms for r in arms[a])
        ):
            omitted.append(
                dict(question_hash=question, reason="missing_paired_probability_or_observed_target")
            )
            continue
        scores = {a: shipped.summary(rs, target) for a, rs in arms.items()}
        paired.append(
            dict(
                question_hash=question,
                sibling_seed_count=len(keys[0]),
                gains={a: scores[a]["brier"] - scores[FULL]["brier"] for a in (SCALAR, PERMUTED)},
                loss_deltas={
                    a: scores[FULL]["decision_loss"] - scores[a]["decision_loss"]
                    for a in (SCALAR, PERMUTED)
                },
            )
        )
    gains: Json = {a: None for a in (SCALAR, PERMUTED)}
    intervals: Json = {a: None for a in (SCALAR, PERMUTED)}
    deltas: Json = {a: None for a in (SCALAR, PERMUTED)}
    if paired:
        vectors = np.asarray([[r["gains"][a] for a in (SCALAR, PERMUTED)] for r in paired])
        rng = np.random.default_rng(RANDOM_SEED)
        draws = []
        for done in range(0, DRAWS, 1000):
            print(
                f"[exp8488] phase=bootstrap_{target} completed={done} pending={DRAWS - done}",
                flush=True,
            )
            draws.append(
                vectors[rng.integers(0, len(paired), size=(1000, len(paired)))].mean(axis=1)
            )
        sampled = np.concatenate(draws)
        for i, arm in enumerate((SCALAR, PERMUTED)):
            gains[arm] = float(vectors[:, i].mean())
            intervals[arm] = np.quantile(sampled[:, i], [0.0125, 0.9875]).tolist()
            deltas[arm] = float(np.mean([r["loss_deltas"][arm] for r in paired]))
    return dict(
        contrasts=gains,
        familywise_intervals=intervals,
        decision_loss_deltas=deltas,
        paired_question_rows=paired,
        omitted_questions=omitted,
        independent_questions=len(paired),
        intended_questions=len(groups),
        draws=DRAWS,
        seed=RANDOM_SEED,
        confidence_level=0.975,
        family_size=2,
        conditional_reason="conditional_on_observed_labels_and_complete_paired_questions"
        if paired
        else "zero_complete_paired_questions",
    )


def matched_risk(rows: list[Json], count: int) -> Json:
    """Fractional boundary ties avoid selecting favorable labels or metadata identifiers."""
    valid = [r for r in rows if r["probability"] is not None and r.get("y") in (0, 1)]
    groups: dict[float, list[Json]] = defaultdict(list)
    for row in valid:
        groups[max(row["probability"], 1 - row["probability"])].append(row)
    remaining, errors = count, 0.0
    for confidence in sorted(groups, reverse=True):
        members = groups[confidence]
        weight = min(remaining, len(members)) / len(members)
        errors += weight * sum((r["probability"] >= 0.5) != bool(r["y"]) for r in members)
        remaining -= min(remaining, len(members))
    return dict(
        selected_count=count,
        available_count=len(valid),
        intended_count=len(rows),
        coverage=count / len(rows) if rows else None,
        risk=errors / count if count else None,
        error_numerator=errors,
        undefined_reason=None if count else "zero_matched_coverage",
        boundary_ties="fractional_label_blind",
    )


def repeat_audit(by_arm: dict[str, list[Json]]) -> Json:
    """Measure duplicate fitted predictions and flips without counting repeats as new questions."""
    result = {}
    for arm, rows in by_arm.items():
        seeds = {
            s: sorted([r for r in rows if r["seed"] == s], key=lambda r: r["unit_id"])
            for s in SEEDS
        }
        hashes = [canonical_hash([r["probability"] for r in seeds[s]]) for s in SEEDS]
        pairs, flips = 0, 0
        for a, b in combinations(SEEDS, 2):
            for first, second in zip(seeds[a], seeds[b], strict=True):
                if first["probability"] is not None and second["probability"] is not None:
                    pairs += 1
                    flips += shipped.action(first["probability"]) != shipped.action(
                        second["probability"]
                    )
        result[arm] = dict(
            unique_prediction_vectors=len(set(hashes)),
            repeat_count=len(hashes),
            deterministic_duplicate_rate=1 - len(set(hashes)) / len(hashes),
            decision_flip_rate=flips / pairs if pairs else None,
            paired_repeat_decisions=pairs,
            flips=flips,
            interpretation="frozen repeated fits; not independent replications",
        )
    return result


def evaluate(predictions: list[Json], targets: list[Json]) -> Json:
    """Keep the full intended panel while reporting primary and sensitivity losses separately."""
    labels = {r["unit_id"]: r for r in targets}
    if len(labels) != len(targets) or len(
        {(r["unit_id"], r["arm"], r["seed"]) for r in predictions}
    ) != len(predictions):
        raise ValueError("duplicate_prediction_or_target")
    rows = []
    for row in predictions:
        p, label = row["probability"], labels.get(row["unit_id"], {})
        if p is not None and (not math.isfinite(p) or not 0 <= p <= 1):
            raise ValueError("invalid_probability")
        y = label.get("y")
        valid = p is not None and y in (0, 1)
        brier = (p - y) ** 2 if valid else None
        rows.append(
            dict(
                row,
                y=y,
                y_sensitivity=label.get("y_sensitivity"),
                family=row["question_hash"],
                decision=shipped.action(p) if p is not None else None,
                status="completed" if valid else "censored",
                censored=not valid,
                excluded=False,
                missing_reason=row.get("missing_reason")
                or ("A1_target_missing" if y not in (0, 1) else None),
                absolute_metric=brier,
                raw_numerator=brier,
                raw_denominator=1,
                metric=dict(brier=brier, decision_loss=shipped.loss(p, y)) if valid else None,
            )
        )
    by_arm = {a: [r for r in rows if r["arm"] == a] for a in ARMS}
    metrics = {
        t: {a: shipped.summary(rs, field) for a, rs in by_arm.items()}
        for t, field in [("A1", "y"), ("A2", "y_sensitivity")]
    }
    primary, sensitivity = contrast(rows, "y"), contrast(rows, "y_sensitivity")
    benefit = bool(
        all(v and v[0] > 0 for v in primary["familywise_intervals"].values())
        and primary["decision_loss_deltas"][SCALAR] is not None
        and primary["decision_loss_deltas"][SCALAR] <= 0
    )
    strata = {}
    datasets = sorted({r["dataset"] for r in rows})
    generators = sorted({r["generator"] for r in rows})
    for fields, roster in [("dataset", datasets), ("generator", generators)]:
        for value in roster:
            selected = [r for r in rows if r[fields] == value]
            strata[f"{fields}={value}"] = dict(
                independent_questions=len({r["question_hash"] for r in selected}),
                metrics={
                    t: {
                        a: shipped.summary([r for r in selected if r["arm"] == a], field)
                        for a in ARMS
                    }
                    for t, field in [("A1", "y"), ("A2", "y_sensitivity")]
                },
            )
    for dataset, generator in product(datasets, generators):
        selected = [r for r in rows if (r["dataset"], r["generator"]) == (dataset, generator)]
        strata[f"dataset={dataset}|generator={generator}"] = dict(
            independent_questions=len({r["question_hash"] for r in selected}),
            missing_cell=not selected,
            metrics={
                t: {a: shipped.summary([r for r in selected if r["arm"] == a], field) for a in ARMS}
                for t, field in [("A1", "y"), ("A2", "y_sensitivity")]
            },
        )
    matched_count = min(s["definitive_count"] for s in metrics["A1"].values())
    return dict(
        rows=rows,
        A1_metrics=metrics["A1"],
        A2_sensitivity=metrics["A2"],
        bootstrap=primary,
        sensitivity_bootstrap=sensitivity,
        **{
            k: primary[k]
            for k in [
                "contrasts",
                "familywise_intervals",
                "decision_loss_deltas",
                "paired_question_rows",
            ]
        },
        exploratory_benefit_score=int(benefit),
        brier_gain_vs_logistic=primary["contrasts"][SCALAR],
        brier_gain_ci95=primary["familywise_intervals"][SCALAR],
        decision_loss_delta=primary["decision_loss_deltas"][SCALAR],
        per_seed_metrics={
            str(s): {
                a: shipped.summary([r for r in rs if r["seed"] == s], "y")
                for a, rs in by_arm.items()
            }
            for s in SEEDS
        },
        per_stratum_metrics=strata,
        repeat_fit_audit=repeat_audit(by_arm),
        matched_coverage_risk={a: matched_risk(rs, matched_count) for a, rs in by_arm.items()},
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
        population_grouping_loss_estimate=None,
    )
