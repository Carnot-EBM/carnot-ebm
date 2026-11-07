"""REQ-REPORT-8868: certify fixed query groups before changing their prevalence.

Labels describe constructed signed relational worlds, so these bounds establish
neither natural-language generalization nor independence from an exact oracle.
"""

from copy import deepcopy
import math
import random
import time
from typing import Any

import numpy as np

from carnot.reporting import split_certificate_methods_8857 as old
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
__all__ = ["old"]
ROOT, POLICIES, CUTOFFS = old.ROOT, old.POLICIES, old.CUTOFFS
NAME, TASK = "experiment_8868_v759_conditional_risk_methods", "exp8868-conditional-risk-methods"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/conditional_risk_methods_8868.py",
    "python/carnot/reporting/conditional_risk_runner_8868.py",
    CLI,
]
TESTS = ["tests/python/test_conditional_risk_methods_8868.py"]
SEED, ALPHA = 7598868, 0.01 / 56
FLOOR = math.ceil(math.log(ALPHA) / math.log(0.95))
GROUPS = dict(
    chain=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]],
    fork=[[0, 1], [0, 2], [0, 3], [3, 4], [3, 5]],
    cycle=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
    disconnected=[[0, 1], [1, 2], [3, 4], [4, 5], [5, 6]],
)
MIXTURES = dict(
    balanced=[0.25] * 4, **{"heavy_" + g: [0.7 if g == h else 0.1 for h in GROUPS] for g in GROUPS}
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Actual counts keep bounded work visible without extending its deadline."""
    print(f"[exp8868] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Freeze the valid construction before any population or calibration labels exist."""
    return dict(
        seed=SEED,
        groups=GROUPS,
        population_per_group=1024,
        replicates=40,
        selection_per_group=240,
        certification_per_group=480,
        labels_per_replicate=2880,
        policy_menu=POLICIES,
        cutoffs=CUTOFFS,
        fitting=dict(seeds=old.old.p.SEEDS, steps=240),
        alpha_allocation=dict(
            total=0.01,
            protocols=2,
            policies=7,
            groups=4,
            per_terminal_bound=ALPHA,
            scope="per independent replicate",
        ),
        exact_sample_requirements=dict(zero_error_minimum=FLOOR, certification_per_group=480),
        risk_bound_construction="independent group Clopper-Pearson upper bounds; pooled weights proportional to known mixture weight times exact input-only acceptance rate; no pooled binomial",
        mixture_weights=MIXTURES,
        risk_target=0.05,
        selection_rule="maximum accepted count with empirical risk<=.05; ties higher cutoff; pooled shared cutoff or one cutoff per group; certify once; empty refuses",
        expected_row_counts=dict(
            rows=3360,
            certificate_choices=560,
            terminal_bounds=2240,
            population=4096,
            paid_labels=115200,
            independent_replicates=40,
        ),
        access_order=[
            "protocol",
            "pilot",
            "population",
            "draws",
            "fit",
            "scores",
            "selection_labels",
            "selection",
            "certification_labels",
            "certificates",
            "census_labels",
            "census",
            "corruption",
        ],
        corruption="append opposite sign of first fact; refit/reselection forbidden; no validity guarantee",
        runtime_budget_s=4200,
        measurement_deadline_s=1800,
    )


def upper(n: int, errors: int) -> float:
    """Empty samples refuse; an exact one-sided bound accounts for unseen errors."""
    return old.upper(n, errors, ALPHA)


def group(query: list[list[str]]) -> str:
    """Canonical directed query structure uses no facts, labels, or entity spellings."""
    if len(query) != 5 or len({q[1] for q in query}) != 5:
        raise ValueError("five_predicate_query")
    names = sorted({q[i] for q in query for i in [0, 2]})
    signature = old.old.p.graph.topology([[names.index(q[0]), names.index(q[2])] for q in query])
    for name, edges in GROUPS.items():
        if signature == old.old.p.graph.topology(edges):
            return name
    raise ValueError("unregistered_query_topology")


def relevant(payload: Json) -> str:
    """Only queried predicates count; unused marker graphs cannot manufacture separation."""
    predicates = {q[1] for q in payload["query"]}
    atoms = [f for f in payload["facts"] if f[1] in predicates]
    names = sorted({f[i] for f in atoms for i in [0, 2]})
    edges = sorted({(names.index(f[0]), names.index(f[2])) for f in atoms})
    return str(old.old.p.graph.topology([list(e) for e in edges]))


def freeze(
    exposed: set[str], *, count: int = 1024, fit_count: int = 30, deadline_s: float = 120
) -> Json:
    """Use the existing signed variants and graph recipe, with no label-based rejection."""
    rng, started, seen = random.Random(SEED), time.monotonic(), set(exposed)
    ring = [(i, (i + 1) % 8) for i in range(8)]
    other = [(a, b) for a in range(8) for b in range(8) if a != b and (a, b) not in ring]
    worlds: list[Json] = []
    inputs: list[Json] = []
    for split, total in [("fit", fit_count), ("population", count)]:
        for name, query_edges in GROUPS.items():
            i = 0
            while i < total:
                if time.monotonic() - started > deadline_s:
                    raise ValueError("freeze_deadline")
                edges = ring + rng.sample(other, rng.randint(2, 12))
                variants = old.old.p.graph.original.variants(query_edges, 2 + i % 2, False)
                for v in variants:
                    # Both r0 endpoints are variables, so these are matching query facts.
                    v["input"]["facts"] += [[f"n{a}", "r0", f"n{b}", True] for a, b in edges]
                chosen = (
                    variants
                    if split == "fit"
                    else [min(variants, key=lambda v: canonical_hash(v["input"]))]
                )
                signatures = {relevant(v["input"]) for v in chosen}
                if signatures & seen:
                    continue
                seen.update(signatures)
                wid = f"v759/8868/{split}/{name}/{i:04d}"
                worlds.append(
                    dict(
                        world_id=wid,
                        split=split,
                        group=name,
                        relevant_structures=sorted(signatures),
                        edges=[list(e) for e in edges],
                    )
                )
                for v in chosen:
                    digest = canonical_hash(v["input"])
                    index = int(digest.split(":")[1][:8], 16) % 4
                    candidates = (
                        [old.old.p.legacy.old.LABELS[index]] * 3
                        if index < 3
                        else list(old.old.p.legacy.old.LABELS)
                    )
                    inputs.append(
                        dict(
                            v,
                            unit_id=wid + "/" + v["condition"],
                            world_id=wid,
                            cluster_id=wid,
                            split=split,
                            group=name,
                            input_sha256=digest,
                            candidates=candidates,
                            agreement="unanimous" if index < 3 else "disagreement",
                            candidate_index=int(canonical_hash(["8868", wid]).split(":")[1][:8], 16)
                            % 3,
                        )
                    )
                i += 1
                if i % 60 == 0:
                    progress("freeze_" + split + "_" + name, i, total - i)
            progress("sealed_" + split + "_" + name, i, 0)
    return dict(
        worlds=worlds,
        inputs=inputs,
        exposed_invariants=sorted(exposed),
        inclusion_rule="existing ring plus2..12 edges on queried r0; existing six signed variants; hash-min population variant; relevant graph exclusion before labels",
    )


def draws(size: int, *, replicates: int = 40) -> Json:
    """Independent spawned group/stage streams retain replacement repeats as paid draws."""
    rows = []
    for i, seed in enumerate(np.random.SeedSequence(SEED).spawn(replicates)):
        streams = seed.spawn(8)
        groups = {
            g: {
                stage: np.random.default_rng(streams[2 * j + k]).integers(0, size, n).tolist()
                for k, (stage, n) in enumerate([("selection", 240), ("certification", 480)])
            }
            for j, g in enumerate(GROUPS)
        }
        rows.append(dict(replicate=i, spawn_key=list(seed.spawn_key), groups=groups))
    return dict(seed=SEED, population_per_group=size, replacement=True, replicates=rows)


def registry(cohort: Json, scores: list[Json]) -> tuple[Any, Json]:
    """Validate the seven-policy roster before opening any requested labels."""
    data = old.arrays(scores, len({r["unit_id"] for r in scores}))
    names = {
        u["unit_id"]: u["group"]
        for u in cohort["inputs"]
        if u.get("split", "population") == "population"
    }
    return data, {g: [i for i, u in enumerate(data[0]) if names[u] == g] for g in GROUPS}


def count(data: Any, truth: Json, indices: list[int], policy: int, cutoff: float | None) -> Json:
    """A refused rule still retains its complete scheduled denominator."""
    return (
        old.counts(data, truth, indices, policy, CUTOFFS.index(cutoff))
        if cutoff is not None
        else dict(accepted=0, errors=0, denominator=len(indices), risk=None)
    )


def select(cohort: Json, scores: list[Json], truth: Json, trace: Json) -> Json:
    """Selection sees only independent selection labels, with higher-cutoff tie breaking."""
    data, indices = registry(cohort, scores)
    choices = []
    for rep in trace["replicates"]:
        for policy, (arm, seed) in enumerate(POLICIES):
            for protocol_name in ["pooled", "grouped"]:
                cutoffs = {}
                for key in ["all"] if protocol_name == "pooled" else list(GROUPS):
                    sample = [
                        indices[g][i]
                        for g in GROUPS
                        if key in ["all", g]
                        for i in rep["groups"][g]["selection"]
                    ]
                    candidates = [
                        dict(count(data, truth, sample, policy, c), cutoff=c) for c in CUTOFFS
                    ]
                    safe = [c for c in candidates if c["accepted"] and c["risk"] <= 0.05]
                    best = max(safe, key=lambda c: (c["accepted"], c["cutoff"])) if safe else None
                    cutoffs[key] = best["cutoff"] if best else None
                choices.append(
                    dict(
                        replicate=rep["replicate"],
                        arm=arm,
                        seed=seed,
                        protocol=protocol_name,
                        cutoffs={g: cutoffs.get(g, cutoffs.get("all")) for g in GROUPS},
                    )
                )
        progress("selection", len(choices), len(trace["replicates"]) * 14 - len(choices))
    return dict(choices=choices, selection_hash=canonical_hash(choices))


def certify(cohort: Json, scores: list[Json], truth: Json, trace: Json, selection: Json) -> Json:
    """Only the selected rules receive certification; exact serving mass supplies pooled weights."""
    if canonical_hash(selection["choices"]) != selection["selection_hash"]:
        raise ValueError("selection_commitment_drift")
    data, indices = registry(cohort, scores)
    choices = []
    for selected in selection["choices"]:
        policy = POLICIES.index((selected["arm"], selected["seed"]))
        rep = trace["replicates"][selected["replicate"]]
        bounds = {}
        for g in GROUPS:
            cutoff = selected["cutoffs"][g]
            sample = [indices[g][i] for i in rep["groups"][g]["certification"]]
            c = count(data, truth, sample, policy, cutoff)
            # Acceptance depends on sealed scores, so its full census needs no labels.
            accepted = (
                int(data[2][CUTOFFS.index(cutoff), policy, indices[g]].sum())
                if cutoff is not None
                else 0
            )
            bounds[g] = dict(
                c,
                risk_upper=upper(c["accepted"], c["errors"]),
                alpha=ALPHA,
                population_accepted=accepted,
                population_denominator=len(indices[g]),
                certification_draw_hash=canonical_hash(sample),
            )
        masses = [
            bounds[g]["population_accepted"] / bounds[g]["population_denominator"] / 4
            for g in GROUPS
        ]
        bound = (
            sum(masses[i] * bounds[g]["risk_upper"] for i, g in enumerate(GROUPS)) / sum(masses)
            if sum(masses)
            else 1.0
        )
        pooled_pass = bound <= 0.05 and all(b["accepted"] for b in bounds.values())
        served = {
            g: bool(
                pooled_pass
                if selected["protocol"] == "pooled"
                else bounds[g]["accepted"] and bounds[g]["risk_upper"] <= 0.05
            )
            for g in GROUPS
        }
        choices.append(
            dict(
                selected,
                bounds=bounds,
                pooled_risk_upper=bound,
                served=served,
                certificate_issued=any(served.values()),
                selected_rule_hash=canonical_hash(selected),
            )
        )
    progress("certificates_frozen", len(choices), 0)
    return dict(
        choices=choices,
        selection_hash=selection["selection_hash"],
        certificate_hash=canonical_hash(choices),
    )


def mixture(counts: Json, weights: list[float]) -> Json:
    """Conditional served risk weights errors and accepts by population mass separately."""
    accepted = sum(
        w * counts[g]["accepted"] / counts[g]["denominator"]
        for g, w in zip(GROUPS, weights, strict=True)
    )
    errors = sum(
        w * counts[g]["errors"] / counts[g]["denominator"]
        for g, w in zip(GROUPS, weights, strict=True)
    )
    return dict(coverage=accepted, error_mass=errors, risk=errors / accepted if accepted else None)


def census(
    cohort: Json, scores: list[Json], truth: Json, certificates: Json, *, corrupted: bool = False
) -> list[Json]:
    """Only frozen certificates reach the complete census and the separate corruption stress."""
    if canonical_hash(certificates["choices"]) != certificates["certificate_hash"]:
        raise ValueError("certificate_commitment_drift")
    data, indices = registry(cohort, scores)
    rows = []
    for choice in certificates["choices"]:
        policy = POLICIES.index((choice["arm"], choice["seed"]))
        counts = {
            g: count(
                data,
                truth,
                indices[g],
                policy,
                choice["cutoffs"][g] if choice["served"][g] else None,
            )
            for g in GROUPS
        }
        for name, weights in (
            dict(evidence_corruption=MIXTURES["balanced"]) if corrupted else MIXTURES
        ).items():
            observed = mixture(counts, weights)
            guarantee = not corrupted and (choice["protocol"] == "grouped" or name == "balanced")
            rows.append(
                dict(
                    choice,
                    scenario=name,
                    mixture_weights=weights,
                    group_counts=counts,
                    unit_id=f"{choice['replicate']}/{choice['protocol']}/{choice['arm']}/{choice['seed']}/{name}",
                    cluster_id=f"rep/{choice['replicate']}",
                    metric=observed["coverage"],
                    denominator=sum(len(v) for v in indices.values()),
                    served_risk=observed["risk"],
                    error_mass=observed["error_mass"],
                    validity_guarantee=guarantee,
                    certificate_failure=bool(
                        guarantee and observed["risk"] is not None and observed["risk"] > 0.05
                    ),
                    status="completed",
                    missing_reason=None,
                )
            )
    progress("census_corruption" if corrupted else "census_mixtures", len(rows), 0)
    return rows


def corrupt(cohort: Json) -> Json:
    """Opposite signed evidence changes the within-group distribution without changing query groups."""
    result = deepcopy(cohort)
    for unit in result["inputs"]:
        first = unit["input"]["facts"][0]
        unit["input"]["facts"].append([*first[:3], not first[3]])
    return result


def summarize(rows: list[Json]) -> Json:
    """Head seeds share draws; comparative headlines reduce only paired replicate means."""
    reps = sorted({r["replicate"] for r in rows})
    pairs = [
        sum(
            r["metric"] * (1 if r["protocol"] == "grouped" else -1)
            for r in rows
            if r["replicate"] == rep and r["scenario"] == "balanced"
        )
        / 7
        for rep in reps
    ]
    return dict(
        paired_coverage_difference=float(np.mean(pairs)) if pairs else None,
        paired_replicate_differences=pairs,
        independent_replicates=len(reps),
        certificate_failure_count=sum(r["certificate_failure"] for r in rows),
        scope="descriptive finite census; no superiority or universal guarantee claim",
    )


def controls() -> list[Json]:
    """Independent exact inversion and a prevalence counterexample reject invalid mathematics."""
    from scipy.stats import binomtest

    checks = [
        dict(
            name="exact_floor", passed=upper(FLOOR - 1, 0) > 0.05 >= upper(FLOOR, 0), minimum=FLOOR
        ),
        dict(
            name="independent_inversion",
            passed=abs(
                upper(480, 1) - binomtest(1, 480, alternative="less").proportion_ci(1 - ALPHA).high
            )
            < 1e-10,
        ),
    ]
    counts = {
        g: dict(accepted=100, errors=10 if i == 0 else 0, denominator=100)
        for i, g in enumerate(GROUPS)
    }
    balanced, stress = mixture(counts, [0.25] * 4), mixture(counts, [0.7, 0.1, 0.1, 0.1])
    checks.append(
        dict(
            name="pooled_counterexample",
            balanced=balanced,
            stress=stress,
            passed=balanced["risk"] <= 0.05 < stress["risk"],
        )
    )
    units = [dict(unit_id=f"control/{g}/{i}", group=g) for g in GROUPS for i in range(4)]
    cohort = dict(inputs=units)
    scores = [
        dict(
            unit_id=u["unit_id"],
            cluster_id=u["unit_id"],
            arm=a,
            seed=seed,
            candidate="support",
            probabilities=[0.96, 0.02, 0.02],
        )
        for a, seed in POLICIES
        for u in units
    ]
    truth = {u["unit_id"]: "support" for u in units}
    trace = draws(4, replicates=2)
    selected = select(cohort, scores, truth, trace)
    certificates = certify(cohort, scores, truth, trace, selected)
    rows = census(cohort, scores, truth, certificates)
    null = summarize(rows)
    checks.append(
        dict(
            name="identical_arm_null",
            observed=null["paired_coverage_difference"],
            passed=null["paired_coverage_difference"] == 0
            and all(c["certificate_issued"] for c in certificates["choices"]),
        )
    )
    checks.append(
        dict(
            name="query_only_grouping",
            passed=all(
                group([[f"?v{a}", f"r{i}", f"?v{b}"] for i, (a, b) in enumerate(edges)]) == name
                for name, edges in GROUPS.items()
            ),
        )
    )
    return checks
