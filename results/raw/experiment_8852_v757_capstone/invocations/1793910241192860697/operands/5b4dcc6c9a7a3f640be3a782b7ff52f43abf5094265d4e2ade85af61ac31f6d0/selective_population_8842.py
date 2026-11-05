"""REQ-REPORT-8842: certificates conditional on one immutable finite population.

Rejecting graphs constructs the population. Only subsequent independent uniform
index draws support binomial inference; neither step establishes text generalization.
"""

import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from scipy.stats import beta, binomtest

from carnot.reporting import selective_methods_8829 as graph
from carnot.reporting import consensus_selective_energy_8819 as legacy
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
__all__ = ["graph", "legacy"]
ROOT = Path(__file__).resolve().parents[3]
NAME, TASK = "experiment_8842_v757_selective_population", "exp8842-selective-population"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [f"python/carnot/reporting/selective_population{s}_8842.py" for s in ["", "_runner"]] + [
    CLI
]
TESTS = ["tests/python/test_selective_population_8842.py"]
SEED, SEEDS, BUDGETS = 7578842, [11, 23, 37, 47, 59], [60, 180, 360]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
ALPHA = 0.01 / (7 * 3 * 6)
FLOOR = math.ceil(math.log(ALPHA) / math.log(0.95))


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Real counts and flushed output make slow work visible without extending deadlines."""
    print(f"[exp8842] phase={phase} completed={completed} pending={pending}", flush=True)


def upper(n: int, errors: int) -> float:
    """A one-sided exact risk bound includes the empty and all-error boundaries."""
    if not 0 <= errors <= n:
        raise ValueError("binomial_counts")
    return 1.0 if not n or errors == n else float(beta.isf(ALPHA, errors + 1, n - errors))


def boundaries() -> list[Json]:
    """Independent SciPy inversion and logarithms must agree at the integer floor."""
    return [
        dict(
            n=n,
            errors=e,
            risk_upper=upper(n, e),
            scipy_upper=binomtest(e, n, alternative="less").proportion_ci(1 - ALPHA).high
            if n
            else 1.0,
            logarithmic_upper=-math.expm1(math.log(ALPHA) / n) if n and not e else None,
            passed=abs(upper(n, e) - (-math.expm1(math.log(ALPHA) / n))) < 1e-10
            if n and not e
            else upper(n, e) == 1.0,
        )
        for n, e in [(0, 0), (FLOOR - 1, 0), (FLOOR, 0), (FLOOR, FLOOR)]
    ]


def freeze(
    exposed: set[str], *, fit_count: int = 120, population_count: int = 4096, pilot_s: float = 300
) -> Json:
    """Freeze disjoint graphs and hash-fixed queries before drawing any indices or labels."""
    rng, started, seen = random.Random(SEED), time.monotonic(), set(exposed)
    worlds: list[Json] = []
    inputs: list[Json] = []
    canaries: list[Json] = []
    ring = [(i, (i + 1) % 8) for i in range(8)]
    other = [(a, b) for a in range(8) for b in range(8) if a != b and (a, b) not in ring]
    attempts = 0
    for split, count in [("fit", fit_count), ("population", population_count), ("canary", 12)]:
        progress("before_freeze_" + split, len(worlds), count)
        i = 0
        while i < count:
            if time.monotonic() - started >= pilot_s:
                raise ValueError("population_pilot_budget_exhausted")
            attempts += 1
            edges = [list(e) for e in sorted(ring + rng.sample(other, rng.randint(2, 12)))]
            invariant = graph.topology(edges)
            if invariant in seen:
                continue
            seen.add(invariant)
            wid = f"v757/8842/{split}/{i:04d}"
            worlds.append(
                dict(
                    world_id=wid,
                    structure=invariant,
                    query_invariant=invariant,
                    vertices=8,
                    edges=edges,
                    split=split,
                )
            )
            variants = graph.original.variants(edges, 2 + i % 2, split == "canary" and i % 4 == 3)
            chosen = (
                variants
                if split == "fit"
                else [min(variants, key=lambda v: canonical_hash(v["input"]))]
            )
            if split == "canary":
                condition = ["sign", "entity_binding", "missing_evidence", "valid_alternative"][
                    i % 4
                ]
                v = variants[[0, 5, 3, 0][i % 4]]
                # Preserve complete alternative witnesses while varying original fact polarity.
                if condition != "valid_alternative" and i // 4 == 1:
                    v["input"]["facts"][0][3] = False
                chosen = [dict(v, condition=condition, sign=v["input"]["facts"][0][3])]
            for v in chosen:
                unit = dict(
                    v,
                    unit_id=wid + "/" + v["condition"],
                    cluster_id=wid,
                    world_id=wid,
                    split=split,
                    input_sha256=canonical_hash(v["input"]),
                )
                index = int(unit["input_sha256"].split(":")[1][:8], 16) % 4
                unit["candidates"] = (
                    [legacy.old.LABELS[index]] * 3 if index < 3 else list(legacy.old.LABELS)
                )
                unit["agreement"] = "unanimous" if index < 3 else "disagreement"
                unit["candidate_index"] = (
                    int(canonical_hash(["8842/candidate", unit["unit_id"]]).split(":")[1][:8], 16)
                    % 3
                )
                (canaries if split == "canary" else inputs).append(unit)
            i += 1
            if i % 60 == 0:
                progress("freeze_" + split, i, count - i)
        progress("after_freeze_" + split, i, 0)
    return dict(
        worlds=worlds,
        inputs=inputs,
        canaries=canaries,
        attempts=attempts,
        exposed_invariants=sorted(exposed),
        inclusion_rule="8-vertex directed ring plus 2..12 uniformly proposed edges; exact topology exclusion before freezing; hash-min variant; no subsequent rejection",
    )


def draws(size: int, *, replicates: int = 30) -> Json:
    """Separate spawned streams preserve genuine replacement draws and all repeats."""
    result = []
    replicate_seeds = np.random.SeedSequence(SEED).spawn(replicates)
    for rep in range(replicates):
        row: Json = dict(
            replicate=rep, replicate_seed=replicate_seeds[rep].generate_state(4).tolist()
        )
        streams = replicate_seeds[rep].spawn(2)
        for offset, (split, count) in enumerate([("calibration", 360), ("evaluation", 240)]):
            sequence = streams[offset]
            rng = np.random.default_rng(sequence)
            initial = rng.bit_generator.state
            indices = rng.integers(0, size, size=count).tolist()
            row[split] = dict(
                spawn_key=list(sequence.spawn_key),
                initial_state=initial,
                indices=indices,
                final_state=rng.bit_generator.state,
            )
        result.append(row)
    return dict(seed=SEED, population_size=size, replacement=True, replicates=result)


def audit_draws(trace: Json, size: int) -> None:
    """Replaying the RNG rejects fabricated repeats and stream aliasing, while retaining real repeats."""
    if canonical_hash(trace) != canonical_hash(draws(size, replicates=len(trace["replicates"]))):
        raise ValueError("independent_draw_trace_drift")


def labels(inputs: list[Json]) -> Json:
    """Release exact labels through two separate evaluators only at their authorized phase."""
    result: Json = {}
    last = time.monotonic()
    progress("before_independent_labels", 0, len(inputs))
    for i, unit in enumerate(inputs):
        result[unit["unit_id"]] = graph.truth(unit["input"])
        if (i + 1) % 120 == 0 or time.monotonic() - last >= 30:
            progress("independent_labels", i + 1, len(inputs) - i - 1)
            last = time.monotonic()
    progress("after_independent_labels", len(inputs), 0)
    return result


def fit(cohort: Json, *, seeds: list[int] = SEEDS, steps: int = 240) -> Json:
    """Unchanged features, initialization and NCE updates consume fitting labels alone."""
    train = [u for u in cohort["inputs"] if u["split"] == "fit"]
    truth = labels(train)
    x = np.asarray([legacy.old.signals(u["input"]) for u in train])
    y = np.asarray([legacy.old.LABELS.index(truth[u["unit_id"]]) for u in train])
    parameters: list[Json] = []
    for seed in seeds:
        progress(
            "before_in_process_training_no_model_load",
            len(parameters),
            len(seeds) - len(parameters),
        )
        parameters.append(dict(seed=seed, models=legacy.old.train(x, y, seed, steps)))
        progress(
            "after_in_process_training_no_model_load", len(parameters), len(seeds) - len(parameters)
        )
    return dict(
        parameters=parameters,
        training_labels=truth,
        steps=steps,
        numerical_training_calls=3 * len(seeds),
        fitting_inputs_sha256=canonical_hash(train),
    )


def predict(cohort: Json, fitted: Json) -> list[Json]:
    """Reuse original action probabilities without opening any population label."""
    units = [u for u in cohort["inputs"] if u["split"] == "population"]
    chosen = {u["unit_id"]: u["candidate_index"] for u in units}
    progress("before_population_prediction", 0, len(units))
    rows, _ = legacy.predict(units, fitted["parameters"], [])
    result = [r for r in rows if r["candidate_index"] == chosen[r["unit_id"]]]
    progress("after_population_prediction", len(units), 0)
    return result


def audit_predictions(predictions: list[Json], cohort: Json, truth: Json, fitted: Json) -> None:
    """Compare actual input-only predictions and independent labels before granting readiness."""
    if canonical_hash(truth) != canonical_hash(labels(cohort["inputs"])):
        raise ValueError("label_permutation_or_drift")
    if canonical_hash(predictions) != canonical_hash(predict(cohort, fitted)):
        raise ValueError("prediction_roster_or_drift")


def reduce(predictions: list[Json], truth: Json, trace: Json) -> Json:
    """Select on calibration only, then audit accepted risk using the full sealed census."""
    size = trace["population_size"]
    audit_draws(trace, size)
    policies = sorted({(r["arm"], r["seed"]) for r in predictions})
    units = sorted({r["unit_id"] for r in predictions})
    if len(units) != size or len(predictions) != size * len(policies):
        raise ValueError("population_prediction_roster")
    lookup = {(r["arm"], r["seed"], r["unit_id"]): r for r in predictions}
    menu, choices, rows = [], [], []
    for rep in trace["replicates"]:
        for budget in BUDGETS:
            for arm, seed in policies:
                sample = [
                    lookup[arm, seed, units[i]] for i in rep["calibration"]["indices"][:budget]
                ]
                candidates = []
                for cutoff in THRESHOLDS:
                    accepted_rows = [
                        r
                        for r in sample
                        if legacy.old.decide(r["probabilities"], cutoff) != "escalate"
                    ]
                    errors = sum(
                        legacy.old.decide(r["probabilities"], cutoff)
                        != legacy.ACTIONS[legacy.old.target(r["candidate"], truth[r["unit_id"]])]
                        for r in accepted_rows
                    )
                    n = len(accepted_rows)
                    item = dict(
                        replicate=rep["replicate"],
                        arm=arm,
                        seed=seed,
                        budget=budget,
                        threshold=cutoff,
                        accepted=n,
                        errors=errors,
                        denominator=budget,
                        risk_upper=upper(n, errors),
                        eligible=n > 0 and upper(n, errors) <= 0.05,
                    )
                    menu.append(item)
                    candidates.append(item)
                safe = [r for r in candidates if r["eligible"]]
                best = max(safe, key=lambda r: (r["accepted"], -r["threshold"])) if safe else None
                choices.append(
                    dict(
                        replicate=rep["replicate"],
                        arm=arm,
                        seed=seed,
                        budget=budget,
                        threshold=best["threshold"] if best else None,
                        calibration_sha256=canonical_hash(sample),
                    )
                )
        progress(
            "calibration_rules_sealed",
            len(choices),
            len(trace["replicates"]) * len(BUDGETS) * len(policies) - len(choices),
        )
    selection_hash = canonical_hash(choices)
    for choice in choices:
        arm, seed, cutoff = choice["arm"], choice["seed"], choice["threshold"]
        rep = trace["replicates"][choice["replicate"]]
        counts: Json = {}
        for split, indices in [
            ("evaluation", rep["evaluation"]["indices"]),
            ("census", range(size)),
        ]:
            accepted = errors = 0
            for index in indices:
                r = lookup[arm, seed, units[index]]
                action = (
                    "escalate" if cutoff is None else legacy.old.decide(r["probabilities"], cutoff)
                )
                accepted += action != "escalate"
                errors += (
                    action != "escalate"
                    and action
                    != legacy.ACTIONS[legacy.old.target(r["candidate"], truth[r["unit_id"]])]
                )
            counts[split] = dict(
                accepted=accepted,
                errors=errors,
                denominator=len(indices),
                risk=errors / accepted if accepted else None,
            )
        ev, census = counts["evaluation"], counts["census"]
        rows.append(
            dict(
                choice,
                unit_id=f"rep/{choice['replicate']}/budget/{choice['budget']}/{arm}/{seed}",
                cluster_id=f"rep/{choice['replicate']}",
                metric=ev["accepted"] / 240,
                denominator=240,
                status="completed",
                missing_reason=None,
                accepted=ev["accepted"],
                errors=ev["errors"],
                served_risk=ev["risk"],
                census_accepted=census["accepted"],
                census_errors=census["errors"],
                census_denominator=size,
                census_risk=census["risk"],
                certificate_issued=cutoff is not None,
                certificate_failure=cutoff is not None
                and census["risk"] is not None
                and census["risk"] > 0.05,
            )
        )
    return dict(
        rows=rows,
        certificate_menu=menu,
        selection_sha256=selection_hash,
        census_released_after_selection=True,
        certificate_firing_count=sum(r["certificate_issued"] for r in rows),
        certificate_failure_count=sum(r["certificate_failure"] for r in rows),
    )
