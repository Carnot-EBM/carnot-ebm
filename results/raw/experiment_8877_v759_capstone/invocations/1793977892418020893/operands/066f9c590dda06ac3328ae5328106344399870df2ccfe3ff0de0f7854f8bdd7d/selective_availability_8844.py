"""REQ-REPORT-8844: audit certificates conditional on a fixed constructed population.

Independent index draws support the risk bounds. Model seeds share draws and
never become extra independent replicates or evidence of text generalization.
"""

import math
from typing import Any

import numpy as np
from scipy.stats import beta

from carnot.reporting import selective_population_8842 as p
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
__all__ = ["p"]
ROOT = p.ROOT
NAME, TASK = "experiment_8844_v757_selective_availability", "exp8844-selective-availability"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/selective_availability{s}_8844.py" for s in ["", "_runner"]] + [
    CLI
]
TESTS = ["tests/python/test_selective_availability_8844.py"]
POLICIES = [("agreement_only", 0), ("fixed_evidence", 0)] + [("learned_energy", s) for s in p.SEEDS]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed real counts expose phase boundaries without extending deadlines."""
    print(f"[exp8844] phase={phase} completed={completed} pending={pending}", flush=True)


def interval(n: int, successes: int) -> Json:
    """Exact two-sided99-percent intervals expose uncertainty from only30 replicates."""
    if not 0 <= successes <= n:
        raise ValueError("binomial_counts")
    return dict(
        numerator=successes,
        denominator=n,
        estimate=successes / n if n else None,
        lower=float(beta.ppf(0.005, successes, n - successes + 1))
        if successes
        else 0.0
        if n
        else None,
        upper=float(beta.isf(0.005, successes + 1, n - successes))
        if successes < n
        else 1.0
        if n
        else None,
    )


def audit_fit(cohort: Json, fitted: Json) -> None:
    """Shuffled fitting labels cannot be admitted as correctly trained evidence."""
    truth = p.labels([u for u in cohort["inputs"] if u["split"] == "fit"])
    if canonical_hash(truth) != canonical_hash(fitted["training_labels"]):
        raise ValueError("fitting_label_permutation")


def fit(cohort: Json) -> Json:
    """Failed training retains its registered seed instead of silently shrinking the roster."""
    fitted: Json = dict(
        parameters=[], failed_seeds=[], numerical_training_calls=0, training_labels={}
    )
    for seed in p.SEEDS:
        try:
            result = p.fit(cohort, seeds=[seed])
            fitted["parameters"].extend(result["parameters"])
            fitted["training_labels"] = result["training_labels"]
            fitted["numerical_training_calls"] += result["numerical_training_calls"]
        except (ValueError, RuntimeError) as error:
            fitted["failed_seeds"].append(dict(seed=seed, error=str(error)))
    return fitted


def predict(cohort: Json, fitted: Json) -> list[Json]:
    """Missing scores stay explicitly missing; failed arms serve no answers."""
    rows = p.predict(cohort, fitted)
    for failure in fitted["failed_seeds"]:
        for unit in cohort["inputs"]:
            if unit["split"] == "population":
                rows.append(
                    dict(
                        unit_id=unit["unit_id"],
                        cluster_id=unit["cluster_id"],
                        candidate=unit["candidates"][unit["candidate_index"]],
                        probabilities=None,
                        arm="learned_energy",
                        seed=failure["seed"],
                        status="failed",
                        missing_reason=failure["error"],
                    )
                )
    return rows


def arrays(
    predictions: list[Json], size: int
) -> tuple[list[str], dict[tuple[str, int, str], Json]]:
    """A complete unique roster keeps failed or duplicated arms from changing denominators."""
    units = sorted({r["unit_id"] for r in predictions})
    lookup = {(r["arm"], r["seed"], r["unit_id"]): r for r in predictions}
    if (
        len(units) != size
        or len(lookup) != len(predictions)
        or set(lookup) != {(a, s, u) for a, s in POLICIES for u in units}
    ):
        raise ValueError("population_policy_roster")
    return units, lookup


def select(predictions: list[Json], calibration_truth: Json, trace: Json) -> Json:
    """Only calibration prefixes choose among the frozen six cutoffs and seven policies."""
    p.audit_draws(trace, trace["population_size"])
    units, lookup = arrays(predictions, trace["population_size"])
    menu, choices = [], []
    for rep in trace["replicates"]:
        for budget in p.BUDGETS:
            indices = rep["calibration"]["indices"][:budget]
            for arm, seed in POLICIES:
                sample = [lookup[arm, seed, units[i]] for i in indices]
                local = []
                for cutoff in p.THRESHOLDS:
                    actions = [
                        p.legacy.old.decide(r["probabilities"], cutoff)
                        if r["probabilities"] is not None
                        else "escalate"
                        for r in sample
                    ]
                    n = sum(a != "escalate" for a in actions)
                    errors = sum(
                        a != "escalate"
                        and a
                        != p.legacy.ACTIONS[
                            p.legacy.old.target(r["candidate"], calibration_truth[r["unit_id"]])
                        ]
                        for a, r in zip(actions, sample, strict=True)
                    )
                    item = dict(
                        replicate=rep["replicate"],
                        arm=arm,
                        seed=seed,
                        budget=budget,
                        threshold=cutoff,
                        accepted=n,
                        errors=errors,
                        denominator=budget,
                        risk_upper=p.upper(n, errors),
                        zero_error_upper=p.upper(n, 0),
                        eligible=n > 0 and p.upper(n, errors) <= 0.05,
                    )
                    local.append(item)
                safe = [r for r in local if r["eligible"]]
                best = max(safe, key=lambda r: (r["accepted"], -r["threshold"])) if safe else None
                choices.append(
                    dict(
                        replicate=rep["replicate"],
                        arm=arm,
                        seed=seed,
                        budget=budget,
                        threshold=best["threshold"] if best else None,
                        calibration_sha256=canonical_hash(sample),
                        refusal_reason=None
                        if best
                        else "insufficient_accepted_support_or_discriminative_errors",
                    )
                )
                menu.extend(local)
        progress("cutoffs_sealed", len(choices), len(trace["replicates"]) * 21 - len(choices))
    return dict(
        selected_cutoff_rows=choices,
        certificate_menu=menu,
        selection_sha256=canonical_hash(choices),
    )


def evaluate(predictions: list[Json], truth: Json, trace: Json, selection: Json) -> Json:
    """Open census truth only after selection; preserve every independent evaluation draw."""
    size = trace["population_size"]
    units, lookup = arrays(predictions, size)
    rows: list[Json] = []
    draws: list[Json] = []
    population: list[Json] = []
    expected = {
        u: p.legacy.ACTIONS[
            p.legacy.old.target(lookup["agreement_only", 0, u]["candidate"], truth[u])
        ]
        for u in units
    }
    for r in predictions:
        population.append(
            dict(
                r,
                truth=truth[r["unit_id"]],
                expected_action=expected[r["unit_id"]],
                metric=max(r["probabilities"]) if r["probabilities"] is not None else None,
                denominator=1,
                status=r.get("status", "completed"),
                missing_reason=r.get("missing_reason"),
            )
        )
    for j, choice in enumerate(selection["selected_cutoff_rows"]):
        arm, seed, cutoff = choice["arm"], choice["seed"], choice["threshold"]
        actions = [
            "escalate"
            if cutoff is None or lookup[arm, seed, u]["probabilities"] is None
            else p.legacy.old.decide(lookup[arm, seed, u]["probabilities"], cutoff)
            for u in units
        ]
        accepted = np.asarray([a != "escalate" for a in actions])
        errors = np.asarray(
            [a != "escalate" and a != expected[u] for a, u in zip(actions, units, strict=True)]
        )
        indices = trace["replicates"][choice["replicate"]]["evaluation"]["indices"]
        n, e, cn, ce = (
            int(accepted[indices].sum()),
            int(errors[indices].sum()),
            int(accepted.sum()),
            int(errors.sum()),
        )
        row = dict(
            choice,
            unit_id=f"rep/{choice['replicate']}/{choice['budget']}/{arm}/{seed}",
            cluster_id=f"rep/{choice['replicate']}",
            metric=n / len(indices),
            denominator=len(indices),
            status=lookup[arm, seed, units[0]].get("status", "completed"),
            missing_reason=lookup[arm, seed, units[0]].get("missing_reason"),
            accepted=n,
            errors=e,
            served_risk=e / n if n else None,
            census_accepted=cn,
            census_errors=ce,
            census_denominator=size,
            census_coverage=cn / size,
            census_risk=ce / cn if cn else None,
            certificate_issued=cutoff is not None,
            certificate_failure=cutoff is not None and cn > 0 and ce / cn > 0.05,
            operational_utility=cn / size >= 0.30 and cn > 0 and ce / cn <= 0.05,
        )
        rows.append(row)
        draws.extend(
            dict(
                replicate=choice["replicate"],
                budget=choice["budget"],
                arm=arm,
                seed=seed,
                draw_index=k,
                population_index=i,
                unit_id=units[i],
                cluster_id=f"rep/{choice['replicate']}",
                action=actions[i],
                error=bool(errors[i]),
                accepted=bool(accepted[i]),
                metric=int(errors[i]),
                denominator=1,
                status=row["status"],
                missing_reason=row["missing_reason"],
            )
            for k, i in enumerate(indices)
        )
        if (j + 1) % 21 == 0:
            progress("census_and_evaluation", j + 1, len(selection["selected_cutoff_rows"]) - j - 1)
    return dict(
        selection,
        rows=rows,
        per_draw_results=draws,
        population_score_rows=population,
        certificate_event_rows=[dict(r) for r in rows],
        served_risk_rows=[dict(r) for r in rows],
        **summarize(rows),
    )


def summarize(rows: list[Json]) -> Json:
    """Paired replicate intervals keep certificate calibration and superiority separate."""
    firing, failures, utility, comparisons = [], [], [], []
    for arm, seed in POLICIES:
        for budget in p.BUDGETS:
            local = [r for r in rows if (r["arm"], r["seed"], r["budget"]) == (arm, seed, budget)]
            n = len(local)
            issued = sum(r["certificate_issued"] for r in local)
            failed = sum(r["certificate_failure"] for r in local)
            identity = dict(arm=arm, seed=seed, budget=budget)
            firing.append(dict(identity, **interval(n, issued)))
            failures.append(
                dict(
                    identity,
                    unconditional=interval(n, failed),
                    conditional=interval(issued, failed),
                )
            )
            utility.append(
                dict(identity, **interval(n, sum(r["operational_utility"] for r in local)))
            )
    indexed = {(r["replicate"], r["arm"], r["seed"], r["budget"]): r for r in rows}
    reps = sorted({r["replicate"] for r in rows})
    for seed in p.SEEDS:
        for baseline in ["agreement_only", "fixed_evidence"]:
            pairs = [
                dict(
                    replicate=i,
                    coverage_gain=indexed[i, "learned_energy", seed, 360]["census_coverage"]
                    - indexed[i, baseline, 0, 360]["census_coverage"],
                    evaluation_coverage_gain=indexed[i, "learned_energy", seed, 360]["metric"]
                    - indexed[i, baseline, 0, 360]["metric"],
                    firing_gain=int(indexed[i, "learned_energy", seed, 360]["certificate_issued"])
                    - int(indexed[i, baseline, 0, 360]["certificate_issued"]),
                    utility_gain=int(indexed[i, "learned_energy", seed, 360]["operational_utility"])
                    - int(indexed[i, baseline, 0, 360]["operational_utility"]),
                    retained_risk=not indexed[i, "learned_energy", seed, 360][
                        "certificate_failure"
                    ],
                )
                for i in reps
            ]
            bounds = {}
            radius = math.sqrt(2 * math.log(10 / 0.01) / len(pairs))
            for key in ["coverage_gain", "evaluation_coverage_gain", "firing_gain", "utility_gain"]:
                mean = sum(row[key] for row in pairs) / len(pairs)
                bounds[key] = dict(
                    estimate=mean, lower=max(-1.0, mean - radius), upper=min(1.0, mean + radius)
                )
            comparisons.append(
                dict(
                    seed=seed,
                    baseline=baseline,
                    budget=360,
                    denominator=len(pairs),
                    paired_rows=pairs,
                    intervals=bounds,
                    lower_99_simultaneous=bounds["coverage_gain"]["lower"],
                    retained_risk=all(row["retained_risk"] for row in pairs),
                    method="Hoeffding paired differences in [-1,1]; one-sided .01/10 per comparison; upper bounds descriptive",
                )
            )
    useful = [r for r in utility if r["budget"] == 360 and r["numerator"] >= 24]
    return dict(
        firing_intervals=firing,
        conditional_failure_intervals=failures,
        utility_intervals=utility,
        paired_comparisons=comparisons,
        selective_utility_score=int(bool(useful)),
        learned_selective_benefit_score=int(
            all(r["retained_risk"] and r["lower_99_simultaneous"] >= 0.05 for r in comparisons)
        ),
        certificate_firing_count=sum(r["certificate_issued"] for r in rows),
        certificate_failure_count=sum(r["certificate_failure"] for r in rows),
    )
