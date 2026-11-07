"""REQ-REPORT-8885: compare paid labels only after both certificates are fixed.

Exact constructed truth measures stationary census risk. It cannot establish
natural-text validity or independent generalization from the oracle.
"""

import math
from typing import Any

from carnot.reporting import sequential_certificate_methods_8880 as frozen
from carnot.reporting.selective_availability_8844 import interval

Json = dict[str, Any]
__all__ = ["frozen"]
ROOT, SEED = frozen.ROOT, frozen.SEED
NAME, TASK = "experiment_8885_v760_sequential_label_cost", "exp8885-sequential-label-cost"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/sequential_label_cost_8885.py",
    "python/carnot/reporting/sequential_label_cost_runner_8885.py",
    CLI,
]
TESTS = ["tests/python/test_sequential_label_cost_8885.py"]
ARMS = list(dict.fromkeys(a for a, _ in frozen.base.POLICIES))


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual completed counts expose bounded work without extending its deadline."""
    print(f"[exp8885] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Freeze all comparisons before opening census outcomes or cost headlines."""
    return dict(
        methods=frozen.protocol(),
        mixtures=frozen.base.MIXTURES,
        family_size=23,
        family_error=0.01,
        interval_method="Bonferroni bounded Hoeffding",
        cost_difference_range=[-3360, 0],
        coverage_difference_range=[-1, 1],
        primary_family="learned_energy",
        coverage_noninferiority=-0.01,
        savings_lower_gt=0,
        risk_ceiling=0.05,
        seed_aggregation="average five learned seeds inside each independent replicate",
        access_order=[
            "protocol",
            "frozen_operands",
            "selection",
            "certificates",
            "census_labels",
            "evaluation",
        ],
        runtime_budget_s=4200,
    )


def evaluate(
    cohort: Json,
    scores: list[Json],
    truth: Json,
    trace: Json,
    certificates: Json,
    *,
    calibration_truth: Json | None = None,
) -> Json:
    """Audit every possible bound, while only issued certificates can serve census answers."""
    base = frozen.base
    calibration_truth = truth if calibration_truth is None else calibration_truth
    data, indices = base.registry(cohort, scores)
    rows, census, bound_rows, violations, availability = [], [], [], [], []
    for stop in certificates["rows"]:
        policy = base.POLICIES.index((stop["arm"], stop["seed"]))
        g = stop["group"]
        selected = base.count(data, truth, indices[g], policy, stop["cutoff"])
        served = (
            selected
            if stop["certificate_issued"]
            else dict(accepted=0, errors=0, denominator=len(indices[g]), risk=None)
        )
        rows.append(
            dict(
                stop,
                census_denominator=len(indices[g]),
                population_accepted=selected["accepted"],
                population_errors=selected["errors"],
                population_risk=selected["risk"],
                served_accepted=served["accepted"],
                served_errors=served["errors"],
                served_risk=served["risk"],
            )
        )
        stream = trace["replicates"][stop["replicate"]]["groups"][g]["certification"]
        for look in frozen.LOOKS if stop["protocol"] == "sequential" else [960]:
            counts = base.count(
                data,
                calibration_truth,
                [indices[g][i] for i in stream[:look]],
                policy,
                stop["cutoff"],
            )
            upper = base.old.upper(
                counts["accepted"], counts["errors"], frozen.ALPHAS[stop["protocol"]]
            )
            bound_rows.append(
                dict(
                    unit_id=stop["unit_id"] + f"/look/{look}",
                    replicate=stop["replicate"],
                    protocol=stop["protocol"],
                    arm=stop["arm"],
                    seed=stop["seed"],
                    group=g,
                    candidate_draws=look,
                    visited=look <= stop["candidate_draws"],
                    risk_upper=upper,
                    population_risk=selected["risk"],
                    **counts,
                    bound_violation=selected["risk"] is not None and selected["risk"] > upper,
                )
            )
        if len(rows) % 56 == 0:
            progress("calibration_family_audit", len(rows), len(certificates["rows"]) - len(rows))
    reps = sorted({x["replicate"] for x in rows})
    for rep in reps:
        local = [x for x in rows if x["replicate"] == rep]
        violations.append(
            dict(
                replicate=rep,
                violation=any(x["bound_violation"] for x in bound_rows if x["replicate"] == rep),
                tested_bounds=sum(x["replicate"] == rep for x in bound_rows),
            )
        )
        for name in frozen.ALPHAS:
            for arm, seed in base.POLICIES:
                cells = {
                    x["group"]: dict(
                        accepted=x["served_accepted"],
                        errors=x["served_errors"],
                        denominator=x["census_denominator"],
                    )
                    for x in local
                    if (x["protocol"], x["arm"], x["seed"]) == (name, arm, seed)
                }
                for mixture, weights in base.MIXTURES.items():
                    observed = base.mixture(cells, weights)
                    census.append(
                        dict(
                            unit_id=f"{rep}/{name}/{arm}/{seed}/{mixture}",
                            cluster_id=f"rep/{rep}",
                            replicate=rep,
                            protocol=name,
                            arm=arm,
                            seed=seed,
                            mixture=mixture,
                            metric=observed["coverage"],
                            served_risk=observed["risk"],
                            error_mass=observed["error_mass"],
                            denominator=sum(len(v) for v in indices.values()),
                            status="completed",
                            missing_reason=None,
                        )
                    )
        progress("census_replicates", len(violations), len(reps) - len(violations))
    for name in frozen.ALPHAS:
        for arm, seed in base.POLICIES:
            for g in base.GROUPS:
                local = [
                    x
                    for x in rows
                    if (x["protocol"], x["arm"], x["seed"], x["group"]) == (name, arm, seed, g)
                ]
                availability.append(
                    dict(
                        protocol=name,
                        arm=arm,
                        seed=seed,
                        group=g,
                        **interval(len(local), sum(x["certificate_issued"] for x in local)),
                    )
                )
    result = dict(
        rows=rows,
        stopping_rows=certificates["rows"],
        census_risk_rows=census,
        calibration_bound_rows=bound_rows,
        family_violation_rows=violations,
        observed_family_violation_count=sum(x["violation"] for x in violations),
        theoretical_family_error_allocation=frozen.protocol()["alpha_allocation"],
        certificate_availability=availability,
        standalone_costs=certificates["standalone_costs"],
        shared_union_costs=certificates["shared_union_costs"],
        actual_label_ledger=dict(
            joint_calibration_labels=certificates["actual_research_labels"],
            selection_labels=len(reps) * 960,
            certification_labels=len(reps) * 3840,
            census_oracle_labels=sum(len(v) for v in indices.values()),
            current_training_labels=0,
            paid_draw_counts_include_repeats=True,
            fresh_external_annotations=0,
            label_source="authenticated retained oracle labels replayed per scheduled draw; fresh census oracle labels",
        ),
        numerical_training_calls=0,
    )
    return dict(result, **summarize(result))


def summarize(result: Json) -> Json:
    """Pair independent replicates, keeping shared policy seeds inside each average."""
    values: dict[str, list[float]] = {}
    pairs = []
    for rep in sorted({x["replicate"] for x in result["rows"]}):
        observations: Json = {}
        for arm in ARMS:
            local = [
                x for x in result["standalone_costs"] if x["replicate"] == rep and x["arm"] == arm
            ]
            difference = sum(x["sequential"] - x["fixed"] for x in local) / len(local)
            observations[arm + "/label_difference"] = difference
            observations[arm + "/label_savings"] = -difference
            for mixture in frozen.base.MIXTURES:
                local = [
                    x
                    for x in result["census_risk_rows"]
                    if x["replicate"] == rep and x["arm"] == arm and x["mixture"] == mixture
                ]
                seeds = sum(a == arm for a, _ in frozen.base.POLICIES)
                observations[arm + "/" + mixture + "/coverage_difference"] = (
                    sum(x["metric"] * (1 if x["protocol"] == "sequential" else -1) for x in local)
                    / seeds
                )
        shared = next(x for x in result["shared_union_costs"] if x["replicate"] == rep)
        difference = shared["sequential_total_labels"] - shared["fixed_total_labels"]
        observations["shared_union/label_difference"] = difference
        observations["shared_union/label_savings"] = -difference
        for key, sample in observations.items():
            values.setdefault(key, []).append(sample)
        pairs.append(dict(replicate=rep, cluster_id=f"rep/{rep}", observations=observations))
    intervals: list[Json] = []
    for key, samples in values.items():
        lo, hi = (
            (-1, 1)
            if key.endswith("coverage_difference")
            else (0, 3360)
            if key.endswith("label_savings")
            else (-3360, 0)
        )
        estimate = sum(samples) / len(samples)
        radius = (hi - lo) * math.sqrt(math.log(2 * 23 / 0.01) / (2 * len(samples)))
        intervals.append(
            dict(
                contrast=key,
                estimate=estimate,
                lower_99=max(lo, estimate - radius),
                upper_99=min(hi, estimate + radius),
                independent_count=len(samples),
                family_size=23,
                family_error=0.01,
                method="Bonferroni bounded Hoeffding",
            )
        )
    indexed = {x["contrast"]: x for x in intervals}
    risks = [x["served_risk"] for x in result["rows"] if x["served_risk"] is not None]
    safe = all(x <= 0.05 for x in risks)
    score = int(
        indexed["learned_energy/label_savings"]["lower_99"] > 0
        and all(
            indexed["learned_energy/" + mix + "/coverage_difference"]["lower_99"] >= -0.01
            for mix in frozen.base.MIXTURES
        )
        and safe
    )
    return dict(
        paired_intervals=intervals,
        per_replicate_results=pairs,
        worst_served_group_risk=max(risks) if risks else None,
        label_savings_score=score,
    )
