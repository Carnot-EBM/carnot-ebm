"""REQ-REPORT-8857: compare independent certification at a fixed total label cost.

Uniform replacement draws support inference only for this constructed population.
The seven policies share draws; model seeds never add independent observations.
"""

import json
from typing import Any

import numpy as np
from scipy.stats import beta

from carnot.reporting import selective_availability_8844 as old
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
__all__ = ["old"]
ROOT, POLICIES, CUTOFFS = old.ROOT, old.POLICIES, old.p.THRESHOLDS
NAME, TASK = "experiment_8857_v758_split_certificate_methods", "exp8857-split-certificate-methods"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/split_certificate_methods_8857.py",
    "python/carnot/reporting/split_certificate_runner_8857.py",
    CLI,
]
TESTS = ["tests/python/test_split_certificate_methods_8857.py"]
SEED = 7588857
ALPHAS = dict(simultaneous=0.01 / (2 * 7 * 6), split=0.01 / (2 * 7))


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed real counts keep long phases visible without extending the deadline."""
    print(f"[exp8857] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Fix all analysis choices before constructing worlds or seeing any labels."""
    return dict(
        seed=SEED,
        population=4096,
        replicates=60,
        calibration=720,
        evaluation=240,
        selection=240,
        certification=480,
        policy_menu=POLICIES,
        cutoffs=CUTOFFS,
        familywise_allocation=dict(total=0.01, protocols=2, policies=7, alphas=ALPHAS),
        cutoff_rule="maximum accepted selection count with empirical error<=.05; ties higher cutoff; empty refuses; certify once, no replacement",
        simultaneous_rule="maximum accepted count among exact certified cutoffs; ties higher cutoff",
        risk_target=0.05,
        useful_coverage=0.30,
        benefit_lower_bound=0.05,
        primary_contrast="paired evaluation coverage split minus simultaneous; mean of all seven policies per replicate; refused serves zero",
        bootstrap=dict(seed=SEED + 1, draws=20000, quantiles=[0.005, 0.995]),
        access_order=[
            "population",
            "draws",
            "fit",
            "scores",
            "selection_labels",
            "selection",
            "certification_labels",
            "certificates",
            "evaluation_labels",
            "evaluation",
            "census_labels",
            "census",
        ],
        fitting=dict(seeds=old.p.SEEDS, steps=240, unchanged=True),
        expected_row_counts=dict(
            rows=840,
            selected_rules=420,
            certificates=840,
            calibration_draws=43200,
            evaluation_draws=14400,
        ),
    )


def freeze(exposed: set[str], *, population_count: int = 4096, fit_count: int = 120) -> Json:
    """Use the existing graph recipe with a fresh stream and exclude all exposed invariants."""
    builder = namespace(old.p, "split8857_population")
    builder.SEED = SEED
    value = builder.freeze(exposed, population_count=population_count, fit_count=fit_count)
    return dict(json.loads(json.dumps(value).replace("v757/8842/", "v758/8857/")))


def draws(size: int, *, replicates: int = 60) -> Json:
    """Draw identities count paid labels while repeated indices retain population support."""
    rows = []
    for rep, seed in enumerate(np.random.SeedSequence(SEED).spawn(replicates)):
        streams = seed.spawn(2)
        row: Json = dict(replicate=rep, replicate_seed=seed.generate_state(4).tolist())
        for i, (name, count) in enumerate([("calibration", 720), ("evaluation", 240)]):
            indices = np.random.default_rng(streams[i]).integers(0, size, size=count).tolist()
            row[name] = [
                dict(draw_id=f"{rep}/{name}/{j}", index=index) for j, index in enumerate(indices)
            ]
            row[name + "_support"] = len(set(indices))
        rows.append(row)
    return dict(seed=SEED, population_size=size, replacement=True, replicates=rows)


def upper(n: int, errors: int, alpha: float) -> float:
    """An empty denominator cannot certify a policy, even when it has no observed errors."""
    if not 0 <= errors <= n or not 0 < alpha < 1:
        raise ValueError("binomial_counts_or_alpha")
    return 1.0 if not n or n == errors else float(beta.isf(alpha, errors + 1, n - errors))


def arrays(scores: list[Json], size: int) -> tuple[list[str], Any, Any, list[str]]:
    """Validate every policy score before labels can affect cutoff decisions."""
    units, lookup = old.arrays(scores, size)
    local = [[lookup[a, s, u] for u in units] for a, s in POLICIES]
    probabilities = np.asarray(
        [[x["probabilities"] for x in policy] for policy in local], dtype=float
    )
    if (
        probabilities.shape != (7, size, 3)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(axis=2), 1)
        or any(x.get("status", "completed") != "completed" for x in scores)
    ):
        raise ValueError("invalid_or_failed_score")
    choices = probabilities.argmax(axis=2)
    accepted = np.asarray([(choices != 2) & (probabilities.max(axis=2) >= c) for c in CUTOFFS])
    return units, choices, accepted, [lookup["agreement_only", 0, u]["candidate"] for u in units]


def counts(
    data: tuple[list[str], Any, Any, list[str]],
    truth: Json,
    indices: list[int],
    policy: int,
    cutoff: int,
) -> Json:
    """Count draws rather than unique support; labels are opened only for requested indices."""
    units, choices, accepted, candidates = data
    mask = accepted[cutoff, policy, indices]
    expected = np.asarray(
        [old.p.legacy.old.target(candidates[i], truth[units[i]]) for i in indices]
    )
    n = int(mask.sum())
    errors = int((mask & (choices[policy, indices] != expected)).sum())
    return dict(accepted=n, errors=errors, denominator=len(indices), risk=errors / n if n else None)


def select(scores: list[Json], truth: Json, trace: Json) -> Json:
    """Only the first240 labels choose a rule; higher cutoffs win equal-coverage ties."""
    data = arrays(scores, trace["population_size"])
    choices: list[Json] = []
    menu: list[Json] = []
    for rep in trace["replicates"]:
        sample = rep["calibration"][:240]
        indices = [x["index"] for x in sample]
        for policy, (arm, seed) in enumerate(POLICIES):
            local = [
                dict(counts(data, truth, indices, policy, c), threshold=cutoff)
                for c, cutoff in enumerate(CUTOFFS)
            ]
            safe = [x for x in local if x["accepted"] and x["risk"] <= 0.05]
            best = max(safe, key=lambda x: (x["accepted"], x["threshold"])) if safe else None
            choices.append(
                dict(
                    replicate=rep["replicate"],
                    arm=arm,
                    seed=seed,
                    threshold=best["threshold"] if best else None,
                    selection_sample_hash=canonical_hash(sample),
                )
            )
            menu.extend(dict(x, replicate=rep["replicate"], arm=arm, seed=seed) for x in local)
        progress(
            "selection_rules_frozen", len(choices), len(trace["replicates"]) * 7 - len(choices)
        )
    return dict(choices=choices, menu=menu, selection_hash=canonical_hash(choices))


def certify(scores: list[Json], truth: Json, trace: Json, selection: Json) -> Json:
    """Certification can refuse its one selected cutoff but cannot search for a replacement."""
    if canonical_hash(selection["choices"]) != selection["selection_hash"]:
        raise ValueError("selection_commitment_drift")
    data = arrays(scores, trace["population_size"])
    choices: list[Json] = []
    menu: list[Json] = []
    for j, selected in enumerate(selection["choices"]):
        rep = trace["replicates"][selected["replicate"]]
        policy = POLICIES.index((selected["arm"], selected["seed"]))
        for name in ALPHAS:
            sample = rep["calibration"] if name == "simultaneous" else rep["calibration"][240:]
            indices = [x["index"] for x in sample]
            cutoffs = CUTOFFS if name == "simultaneous" else [selected["threshold"]]
            local = []
            for cutoff in cutoffs:
                if cutoff is None:
                    continue
                item = dict(
                    counts(data, truth, indices, policy, CUTOFFS.index(cutoff)), threshold=cutoff
                )
                item["risk_upper"] = upper(item["accepted"], item["errors"], ALPHAS[name])
                local.append(item)
            safe = [x for x in local if x["accepted"] and x["risk_upper"] <= 0.05]
            best = max(safe, key=lambda x: (x["accepted"], x["threshold"])) if safe else None
            choices.append(
                dict(
                    selected,
                    protocol=name,
                    selected_threshold=selected["threshold"] if name == "split" else None,
                    threshold=best["threshold"] if best else None,
                    certificate_issued=best is not None,
                    certificate_sample_hash=canonical_hash(sample),
                    selected_rule_hash=canonical_hash(selected),
                    alpha=ALPHAS[name],
                    certificate=best,
                    refusal_reason=None if best else "no_valid_certificate",
                )
            )
            menu.extend(
                dict(
                    x,
                    protocol=name,
                    replicate=selected["replicate"],
                    arm=selected["arm"],
                    seed=selected["seed"],
                )
                for x in local
            )
        if (j + 1) % 7 == 0:
            progress(
                "certificates_frozen", len(choices), len(selection["choices"]) * 2 - len(choices)
            )
    return dict(
        choices=choices,
        menu=menu,
        selection_hash=selection["selection_hash"],
        certificate_hash=canonical_hash(choices),
    )


def evaluate(scores: list[Json], truth: Json, trace: Json, certificates: Json) -> list[Json]:
    """Independent evaluation follows certificate freezing; refused certificates serve zero."""
    data = arrays(scores, trace["population_size"])
    rows = []
    for choice in certificates["choices"]:
        rep = trace["replicates"][choice["replicate"]]
        indices = [x["index"] for x in rep["evaluation"]]
        cutoff = choice["threshold"]
        observed: Json = (
            counts(
                data,
                truth,
                indices,
                POLICIES.index((choice["arm"], choice["seed"])),
                CUTOFFS.index(cutoff),
            )
            if cutoff is not None
            else dict(accepted=0, errors=0, denominator=len(indices), risk=None)
        )
        rows.append(
            dict(
                choice,
                unit_id=f"{choice['replicate']}/{choice['protocol']}/{choice['arm']}/{choice['seed']}",
                cluster_id=f"rep/{choice['replicate']}",
                metric=observed["accepted"] / len(indices),
                denominator=len(indices),
                accepted=observed["accepted"],
                errors=observed["errors"],
                served_risk=observed["risk"],
                evaluation_sample_hash=canonical_hash(rep["evaluation"]),
                status="completed",
                missing_reason=None,
            )
        )
    return rows


def census(scores: list[Json], truth: Json, rows: list[Json]) -> list[Json]:
    """The last label release diagnoses issued certificates without changing their rules."""
    size = len(truth)
    data = arrays(scores, size)
    result = []
    for row in rows:
        cutoff = row["threshold"]
        observed: Json = (
            counts(
                data,
                truth,
                list(range(size)),
                POLICIES.index((row["arm"], row["seed"])),
                CUTOFFS.index(cutoff),
            )
            if cutoff is not None
            else dict(accepted=0, errors=0, denominator=size, risk=None)
        )
        result.append(
            dict(
                row,
                census_accepted=observed["accepted"],
                census_errors=observed["errors"],
                census_denominator=size,
                census_risk=observed["risk"],
                certificate_failure=observed["risk"] is not None and observed["risk"] > 0.05,
            )
        )
    return result


def summarize(rows: list[Json]) -> Json:
    """Resample paired replicate means so seven correlated policies never count as seven trials."""
    replicates = sorted({r["replicate"] for r in rows})
    differences = [
        float(
            np.mean(
                [
                    r["metric"] * (1 if r["protocol"] == "split" else -1)
                    for r in rows
                    if r["replicate"] == rep
                ]
            )
            * 2
        )
        for rep in replicates
    ]
    samples = (
        np.random.default_rng(SEED + 1)
        .choice(differences, size=(20000, len(differences)))
        .mean(axis=1)
    )
    lower, upper_bound = np.quantile(samples, [0.005, 0.995]).tolist()
    policies = []
    for arm, seed in POLICIES:
        local = [r for r in rows if (r["arm"], r["seed"]) == (arm, seed)]
        policies.append(
            dict(
                arm=arm,
                seed=seed,
                protocols={
                    name: dict(
                        mean_coverage=float(
                            np.mean([r["metric"] for r in local if r["protocol"] == name])
                        ),
                        certificates=sum(
                            r["certificate_issued"] for r in local if r["protocol"] == name
                        ),
                        certificate_failures=sum(
                            r["certificate_failure"] for r in local if r["protocol"] == name
                        ),
                        useful_certificates=sum(
                            r["certificate_issued"]
                            and r["census_accepted"] / r["census_denominator"] >= 0.30
                            and r["census_risk"] is not None
                            and r["census_risk"] <= 0.05
                            for r in local
                            if r["protocol"] == name
                        ),
                    )
                    for name in ALPHAS
                },
            )
        )
    return dict(
        primary_contrast=dict(
            estimate=float(np.mean(differences)),
            lower_99=lower,
            upper_99=upper_bound,
            paired_replicate_differences=differences,
            independent_replicates=len(replicates),
            bootstrap_draws=20000,
        ),
        protocol_benefit_score=int(lower > 0.05),
        policy_summaries=policies,
    )
