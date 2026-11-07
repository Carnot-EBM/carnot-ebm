"""REQ-REPORT-8870: fixed-group prevalence changes test conditional certificates.

All labels describe constructed relational worlds. Exact oracle checks cannot
establish natural-language correctness or independent generalization.
"""

import math
from typing import Any

import numpy as np

from carnot.reporting import conditional_risk_methods_8868 as base
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
__all__ = ["base"]
ROOT, SEED = base.ROOT, base.SEED
NAME, TASK = (
    "experiment_8870_v759_conditional_risk_evaluation",
    "exp8870-conditional-risk-evaluation",
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/conditional_risk_evaluation_8870.py",
    "python/carnot/reporting/conditional_risk_evaluation_runner_8870.py",
    CLI,
]
TESTS = ["tests/python/test_conditional_risk_evaluation_8870.py"]
ARMS = list(dict.fromkeys(a for a, _ in base.POLICIES))


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual completed counts expose bounded work without extending a deadline."""
    print(f"[exp8870] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Fix the contrasts and interval family before opening evaluation outcomes."""
    return dict(
        methods=base.protocol(),
        independent_count=40,
        primary_arm="learned_energy",
        primary_family_size=36,
        interval_method="Bonferroni bounded Hoeffding",
        family_error=0.01,
        coverage_floor=0.30,
        risk_ceiling=0.05,
        contrast_ranges=dict(coverage=[0, 1], paired_difference=[-1, 1]),
        seed_aggregation="mean within arm and replicate; five learned seeds share labels",
        runtime_budget_s=4200,
        current_training_calls=0,
        corruption_assumption_broken="Within-group exchangeability of the joint evidence/outcome distribution between certification draws and serving worlds; opposite signed facts change this distribution, not just group prevalence.",
    )


def evaluate(
    cohort: Json, scores: list[Json], truth: Json, certificates: Json, *, corrupted: bool = False
) -> Json:
    """Recompute every census mask and keep selected risks even for refused groups."""
    rows = base.census(cohort, scores, truth, certificates, corrupted=corrupted)
    data, indices = base.registry(cohort, scores)
    groups = []
    for choice in certificates["choices"]:
        policy = base.POLICIES.index((choice["arm"], choice["seed"]))
        for g in base.GROUPS:
            selected = base.count(data, truth, indices[g], policy, choice["cutoffs"][g])
            served = choice["served"][g]
            observed = (
                selected
                if served
                else dict(accepted=0, errors=0, denominator=len(indices[g]), risk=None)
            )
            bound = choice["bounds"][g]
            groups.append(
                dict(
                    replicate=choice["replicate"],
                    protocol=choice["protocol"],
                    arm=choice["arm"],
                    seed=choice["seed"],
                    group=g,
                    cutoff=choice["cutoffs"][g],
                    unit_id=f"{choice['replicate']}/{choice['protocol']}/{choice['arm']}/{choice['seed']}/{g}",
                    cluster_id=f"rep/{choice['replicate']}",
                    **observed,
                    metric=observed["risk"],
                    certificate_issued=served,
                    refusal_reason=None if served else "selected_group_or_pooled_bound_refused",
                    risk_upper=bound["risk_upper"],
                    calibration_accepted=bound["accepted"],
                    calibration_errors=bound["errors"],
                    calibration_denominator=bound["denominator"],
                    selected_population_accepted=selected["accepted"],
                    selected_population_errors=selected["errors"],
                    selected_population_risk=selected["risk"],
                    bound_violation=bool(
                        not corrupted
                        and selected["risk"] is not None
                        and selected["risk"] > bound["risk_upper"]
                    ),
                    unsafe_served_group=bool(
                        served and observed["risk"] is not None and observed["risk"] > 0.05
                    ),
                    status="completed",
                    missing_reason=None,
                )
            )
    return dict(rows=rows, group_count_rows=groups)


def summarize(rows: list[Json], groups: list[Json]) -> Json:
    """Bound a fixed family over independent replicates, never over seeds or mixtures."""
    reps = sorted({r["replicate"] for r in rows})
    paired: list[Json] = []
    values: dict[str, list[float]] = {}
    intervals: list[Json] = []
    summaries: list[Json] = []
    for rep in reps:
        observations = []
        for arm in ARMS:
            seeds = [s for a, s in base.POLICIES if a == arm]
            risks = {}
            for name in ["pooled", "grouped"]:
                local = [
                    r
                    for r in rows
                    if r["replicate"] == rep and r["arm"] == arm and r["protocol"] == name
                ]
                for mixture in base.MIXTURES:
                    selected = [r for r in local if r["scenario"] == mixture]
                    key = f"{arm}/{name}/{mixture}/coverage"
                    values.setdefault(key, []).append(
                        sum(r["metric"] for r in selected) / len(seeds)
                    )
                risks[name] = sum(
                    max([max((r["served_risk"] or 0) - 0.05, 0) for r in local if r["seed"] == s])
                    for s in seeds
                ) / len(seeds)
            gain = (
                values[f"{arm}/grouped/balanced/coverage"][-1]
                - values[f"{arm}/pooled/balanced/coverage"][-1]
            )
            reduction = risks["pooled"] - risks["grouped"]
            values.setdefault(f"{arm}/balanced_coverage_gain", []).append(gain)
            values.setdefault(f"{arm}/worst_mixture_excess_reduction", []).append(reduction)
            observations.append(
                dict(
                    arm=arm,
                    balanced_coverage_gain=gain,
                    worst_mixture_excess_reduction=reduction,
                    worst_mixture_excess_by_protocol=risks,
                )
            )
        paired.append(
            dict(
                unit_id=f"rep/{rep}",
                cluster_id=f"rep/{rep}",
                replicate=rep,
                observations=observations,
                metric=observations[-1]["worst_mixture_excess_reduction"],
                denominator=1,
                status="completed",
                missing_reason=None,
            )
        )
    for key, samples in values.items():
        lo, hi = (0.0, 1.0) if key.endswith("/coverage") else (-1.0, 1.0)
        radius = (hi - lo) * math.sqrt(math.log(2 * 36 / 0.01) / (2 * len(reps)))
        estimate = float(np.mean(samples))
        intervals.append(
            dict(
                contrast=key,
                estimate=estimate,
                lower_99=max(lo, estimate - radius),
                upper_99=min(hi, estimate + radius),
                independent_count=len(reps),
                family_size=36,
                family_error=0.01,
                method="Bonferroni bounded Hoeffding",
            )
        )
    by_key = {r["contrast"]: r for r in intervals}
    for arm in ARMS:
        gates = {}
        for name in ["pooled", "grouped"]:
            local = [g for g in groups if g["arm"] == arm and g["protocol"] == name]
            useful = all(
                by_key[f"{arm}/{name}/{mix}/coverage"]["lower_99"] >= 0.30 for mix in base.MIXTURES
            ) and not any(g["unsafe_served_group"] for g in local)
            gates[name] = int(useful)
        advantage = bool(
            gates["grouped"]
            and not gates["pooled"]
            and by_key[f"{arm}/worst_mixture_excess_reduction"]["lower_99"] > 0
        )
        summaries.append(
            dict(
                arm=arm,
                utility_by_protocol=gates,
                robustness_advantage=int(advantage),
                comparison="parity"
                if gates["grouped"] and gates["pooled"]
                else "robustness_advantage"
                if advantage
                else "terminal_null_low_coverage",
            )
        )
    primary = next(s for s in summaries if s["arm"] == "learned_energy")
    availability = []
    for arm, seed in base.POLICIES:
        for name in ["pooled", "grouped"]:
            for g in base.GROUPS:
                local = [
                    r
                    for r in groups
                    if (r["arm"], r["seed"], r["protocol"], r["group"]) == (arm, seed, name, g)
                ]
                availability.append(
                    dict(
                        arm=arm,
                        seed=seed,
                        protocol=name,
                        group=g,
                        **base.old.old.interval(
                            len(local), sum(r["certificate_issued"] for r in local)
                        ),
                    )
                )
    return dict(
        certificate_event_rows=groups,
        group_count_rows=groups,
        census_risk_rows=[
            dict(g, denominator=g["accepted"], population_denominator=g["denominator"])
            for g in groups
        ],
        mixture_rows=rows,
        paired_replicate_rows=paired,
        simultaneous_intervals=intervals,
        policy_summaries=summaries,
        certificate_availability=availability,
        calibration_family_violation_count=sum(
            any(g["bound_violation"] for g in groups if g["replicate"] == rep) for rep in reps
        ),
        calibration_family_violation_rows=[
            dict(
                replicate=rep,
                violation=any(g["bound_violation"] for g in groups if g["replicate"] == rep),
            )
            for rep in reps
        ],
        worst_served_group_risk=max(
            (g["risk"] for g in groups if g["risk"] is not None), default=None
        ),
        conditional_utility_score=primary["utility_by_protocol"]["grouped"],
        robustness_advantage_score=primary["robustness_advantage"],
        comparison=primary["comparison"],
        uncertainty_note="Simultaneous conservative 99-percent Hoeffding intervals over36 fixed quantities using40 independent calibration replicates, conditional on this population and fitted heads. No universal failure guarantee is inferred from zero violations. No service has null risk and zero excess, and cannot pass coverage.",
    )
