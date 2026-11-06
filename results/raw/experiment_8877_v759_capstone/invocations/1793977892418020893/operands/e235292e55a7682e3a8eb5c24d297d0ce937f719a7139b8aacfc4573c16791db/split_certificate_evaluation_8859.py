"""REQ-REPORT-8859: distinguish certificate validity, availability and empirical utility.

Independent draw streams support inference for this constructed population only.
The five fitted seeds share draws and never count as additional replicates.
"""

from typing import Any

import numpy as np

from carnot.reporting import split_certificate_methods_8857 as registered
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
__all__ = ["registered", "old"]
ROOT, POLICIES, ALPHAS = registered.ROOT, registered.POLICIES, registered.ALPHAS
NAME, TASK = (
    "experiment_8859_v758_split_certificate_evaluation",
    "exp8859-split-certificate-evaluation",
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/split_certificate_evaluation_8859.py",
    "python/carnot/reporting/split_certificate_evaluation_runner_8859.py",
    CLI,
]
TESTS = ["tests/python/test_split_certificate_evaluation_8859.py"]
SEED = registered.SEED
METHOD_PROTOCOL_PIN = "sha256:8f569d7b5f5da65937793f90d38900cfd6993a4874af71eb1194982654a2889f"
old = registered.old
engine = namespace(registered, "evaluation8859")
draws = registered.draws


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counts keep each bounded phase visible without extending its deadline."""
    print(f"[exp8859] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Keep original certification rules while recording the requested evaluation bootstrap."""
    value = registered.protocol()
    value["bootstrap"]["draws"] = 10000
    value["original_method_protocol_hash"] = METHOD_PROTOCOL_PIN
    return value


def arrays(scores: list[Json], size: int) -> Any:
    """A failed fit receives an escalation score but retains its original failed evidence."""
    return registered.arrays(
        [
            dict(x, probabilities=[0, 0, 1], status="completed")
            if x.get("status") == "failed"
            else x
            for x in scores
        ],
        size,
    )


engine.arrays = arrays
engine.progress = progress
select, certify, census = engine.select, engine.certify, engine.census


def evaluate(scores: list[Json], truth: Json, trace: Json, certificates: Json) -> list[Json]:
    """Retain failed fits in the scheduled roster and keep their served coverage at zero."""
    failed = {(x["arm"], x["seed"]): x for x in scores if x.get("status") == "failed"}
    rows: list[Json] = engine.evaluate(scores, truth, trace, certificates)
    for row in rows:
        if (row["arm"], row["seed"]) in failed:
            row.update(
                status="failed", missing_reason=failed[row["arm"], row["seed"]]["missing_reason"]
            )
    return rows


def per_draw(scores: list[Json], truth: Json, trace: Json, certificates: Json) -> list[Json]:
    """Repeated population indices remain separate paid evaluation draws for every rule."""
    units, choices, accepted, candidates = arrays(scores, trace["population_size"])
    failed = {(x["arm"], x["seed"]): x for x in scores if x.get("status") == "failed"}
    rows = []
    for rule in certificates["choices"]:
        policy = POLICIES.index((rule["arm"], rule["seed"]))
        for draw in trace["replicates"][rule["replicate"]]["evaluation"]:
            i, cutoff = draw["index"], rule["threshold"]
            served = cutoff is not None and bool(
                accepted[registered.CUTOFFS.index(cutoff), policy, i]
            )
            expected = old.p.legacy.old.target(candidates[i], truth[units[i]])
            rows.append(
                dict(
                    unit_id=f"{draw['draw_id']}/{rule['protocol']}/{policy}",
                    cluster_id=f"rep/{rule['replicate']}",
                    draw_id=draw["draw_id"],
                    population_unit_id=units[i],
                    replicate=rule["replicate"],
                    protocol=rule["protocol"],
                    arm=rule["arm"],
                    seed=rule["seed"],
                    threshold=cutoff,
                    certificate_issued=rule["certificate_issued"],
                    accepted=served,
                    decision=old.p.legacy.ACTIONS[int(choices[policy, i])]
                    if served
                    else "escalate",
                    expected_action=old.p.legacy.ACTIONS[int(expected)],
                    error=bool(choices[policy, i] != expected) if served else None,
                    metric=int(served),
                    denominator=1,
                    status="failed" if (rule["arm"], rule["seed"]) in failed else "completed",
                    missing_reason=failed.get((rule["arm"], rule["seed"]), {}).get(
                        "missing_reason"
                    ),
                )
            )
    return rows


def summarize(rows: list[Json]) -> Json:
    """Resample whole replicate pairs and show exact uncertainty without a universal guarantee."""
    reps = sorted({r["replicate"] for r in rows})
    paired = [
        dict(
            unit_id=f"rep/{rep}",
            cluster_id=f"rep/{rep}",
            replicate=rep,
            metric=sum(
                r["metric"] * (1 if r["protocol"] == "split" else -1)
                for r in rows
                if r["replicate"] == rep
            )
            / 7,
            denominator=7,
            status="completed",
            missing_reason=None,
        )
        for rep in reps
    ]
    differences = [r["metric"] for r in paired]
    samples = (
        np.random.default_rng(SEED + 1).choice(differences, size=(10000, len(reps))).mean(axis=1)
    )
    lower, upper = np.quantile(samples, [0.005, 0.995]).tolist()
    firing, failure, conditional, policies = [], [], [], []
    for arm, seed in POLICIES:
        for name in ALPHAS:
            local = [r for r in rows if (r["arm"], r["seed"], r["protocol"]) == (arm, seed, name)]
            issued, bad = (
                sum(r["certificate_issued"] for r in local),
                sum(r["certificate_failure"] for r in local),
            )
            identity = dict(arm=arm, seed=seed, protocol=name)
            firing.append(dict(identity, **old.interval(len(local), issued)))
            failure.append(dict(identity, **old.interval(len(local), bad)))
            conditional.append(dict(identity, **old.interval(issued, bad)))
            policies.append(
                dict(
                    identity,
                    mean_coverage=float(np.mean([r["metric"] for r in local])),
                    issued=issued,
                    population_false_certifications=bad,
                    replicates=len(local),
                )
            )
    useful = [
        r
        for r in rows
        if r["certificate_issued"] and r["census_accepted"] / r["census_denominator"] >= 0.30
    ]
    safe = bool(useful) and all(
        r["census_risk"] is not None and r["census_risk"] <= 0.05 for r in useful
    )
    return dict(
        primary_contrast=dict(
            estimate=float(np.mean(differences)),
            lower_99=lower,
            upper_99=upper,
            independent_replicates=len(reps),
            bootstrap_draws=10000,
        ),
        split_protocol_benefit_score=int(lower > 0.05 and safe),
        paired_replicate_rows=paired,
        firing_intervals=firing,
        certificate_failure_intervals=failure,
        conditional_failure_intervals=conditional,
        policy_summaries=policies,
        certificate_event_rows=[
            dict(r, population_false_certification=r["certificate_failure"]) for r in rows
        ],
        served_risk_rows=[
            dict(
                unit_id=r["unit_id"],
                cluster_id=r["cluster_id"],
                protocol=r["protocol"],
                arm=r["arm"],
                seed=r["seed"],
                replicate=r["replicate"],
                metric=r["served_risk"],
                denominator=r["accepted"],
                errors=r["errors"],
                decisive_coverage=r["metric"],
                status=r["status"],
                missing_reason=r["missing_reason"],
            )
            for r in rows
        ],
        census_risk_rows=[
            dict(
                unit_id=r["unit_id"],
                cluster_id=r["cluster_id"],
                protocol=r["protocol"],
                arm=r["arm"],
                seed=r["seed"],
                replicate=r["replicate"],
                metric=r["census_risk"],
                denominator=r["census_accepted"],
                errors=r["census_errors"],
                decisive_coverage=r["census_accepted"] / r["census_denominator"],
                claimed_useful=r in useful
                and r["census_risk"] is not None
                and r["census_risk"] <= 0.05,
                status=r["status"],
                missing_reason=r["missing_reason"],
            )
            for r in rows
        ],
        uncertainty_note="Exact two-sided 99-percent binomial intervals. Sixty replicates cannot empirically establish a universal one-percent failure guarantee.",
        method_superiority_score=0,
        familywise_failure_interval=old.interval(
            len(reps),
            sum(
                any(r["certificate_failure"] for r in rows if r["replicate"] == rep) for rep in reps
            ),
        ),
    )
