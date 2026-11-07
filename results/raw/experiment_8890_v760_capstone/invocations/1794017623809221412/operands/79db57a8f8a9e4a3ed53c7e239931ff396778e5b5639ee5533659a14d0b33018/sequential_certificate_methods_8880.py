"""REQ-REPORT-8880: finite-look certificates count every acquired candidate label.

The guarantee is conditional on the frozen constructed population and policies.
Repeated looks spend a finite error budget; they are not an anytime confidence sequence.
"""

import json
import math
from typing import Any

import numpy as np

from carnot.reporting import conditional_risk_methods_8868 as base
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
__all__ = ["base"]
ROOT = base.ROOT
NAME, TASK = (
    "experiment_8880_v760_sequential_certificate_methods",
    "exp8880-sequential-certificate-methods",
)
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/sequential_certificate_methods_8880.py",
    "python/carnot/reporting/sequential_certificate_runner_8880.py",
    CLI,
]
TESTS = ["tests/python/test_sequential_certificate_methods_8880.py"]
SEED, LOOKS = 7608880, [120, 240, 480, 960]
ALPHAS = dict(sequential=0.01 / (2 * 7 * 4 * 4), fixed=0.01 / (2 * 7 * 4))


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed actual counts keep bounded work visible without moving its deadline."""
    print(f"[exp8880] phase={phase} completed={completed} pending={pending}", flush=True)


def minimum(alpha: float) -> int:
    """Zero errors still require enough accepted examples to rule out unseen errors."""
    return math.ceil(math.log(alpha) / math.log(0.95))


def protocol() -> Json:
    """Freeze selection, spending and costs before opening population outcomes."""
    return dict(
        seed=SEED,
        population_per_group=1024,
        replicates=40,
        selection_per_group=240,
        groups=base.GROUPS,
        policy_menu=base.POLICIES,
        cutoffs=base.CUTOFFS,
        look_schedule=LOOKS,
        fixed_budget=960,
        risk_target=0.05,
        alpha_allocation=dict(
            total=0.01,
            per_bound=ALPHAS,
            sequential_total=7 * 4 * 4 * ALPHAS["sequential"],
            fixed_total=7 * 4 * ALPHAS["fixed"],
            scope="per replicate, conditional on selection and fixed population",
        ),
        exact_sample_requirements={k: minimum(a) for k, a in ALPHAS.items()},
        bound_proof="Condition on the frozen population, policies and independent selection labels. Each group uses iid uniform replacement candidates; acceptance is input-only and fixed. Conditional on the accepted count, errors are binomial at its fixed group risk. Each exact upper bound undercovers with probability at most its allocated alpha, including random accepted counts; n=0 refuses with upper=1. A union bound over 7*4*4 sequential events costs .005 and 7*4 fixed events costs .005. Stopping at the first valid bound selects a subset of this finite family, so unsafe certification probability is at most .01 per replicate. Dependence across looks, policies and protocols requires no extra independence. This is not a guarantee across forty replicates or arbitrary looks. No lower-bound early rejection is used.",
        label_cost_definition="Standalone cell pays 240 selection plus all candidate draws through its terminal look, including repeats and unaccepted draws. Standalone policy sums four cells. Shared sequential union pays selection once per group plus the largest terminal look among seven policies. Fixed shared union pays four*(240+960). Actual joint research acquisition shares selection and the full fixed stream, paying 4800 labels per replicate, not the sum of standalone costs. Oracle census and pilot labels are separate research overhead; retained historical fit labels are not current labels.",
        selection_rule="Group cutoffs maximize accepted selection draws at empirical risk<=.05; ties higher cutoff. Both protocols share the same rule; empty refuses after full budget.",
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
        ],
        runtime_budget_s=4200,
        measurement_deadline_s=1800,
        scientific_benefit_claim=False,
        motivation=dict(
            url="https://arxiv.org/abs/2602.01637",
            scope="Sequential feasibility motivation only; the proof here is a finite union bound.",
        ),
    )


def freeze(exposed: set[str], *, count: int = 1024) -> Json:
    """Reuse the query-relevant recipe with a new stream and no new fitted heads."""
    builder = namespace(base, "sequential8880_population")
    builder.SEED = SEED
    value = builder.freeze(exposed, count=count, fit_count=0)
    return dict(json.loads(json.dumps(value).replace("v759/8868/", "v760/8880/")))


def draws(size: int, *, replicates: int = 40) -> Json:
    """Spawn independent stage streams while retaining replacement repeats as paid draws."""
    rows = []
    for i, seed in enumerate(np.random.SeedSequence(SEED).spawn(replicates)):
        streams = seed.spawn(8)
        groups = {
            g: {
                stage: np.random.default_rng(streams[2 * j + k]).integers(0, size, n).tolist()
                for k, (stage, n) in enumerate([("selection", 240), ("certification", 960)])
            }
            for j, g in enumerate(base.GROUPS)
        }
        rows.append(dict(replicate=i, spawn_key=list(seed.spawn_key), groups=groups))
    return dict(seed=SEED, population_per_group=size, replacement=True, replicates=rows)


def select(cohort: Json, scores: list[Json], truth: Json, trace: Json) -> Json:
    """Only selection labels can choose cutoffs shared by both certification protocols."""
    choices = [
        x
        for x in base.select(cohort, scores, truth, trace)["choices"]
        if x["protocol"] == "grouped"
    ]
    return dict(choices=choices, selection_hash=canonical_hash(choices))


def certify(cohort: Json, scores: list[Json], truth: Json, trace: Json, selection: Json) -> Json:
    """Inspect prefixes only; later labels acquired for the fixed arm cannot alter a stop."""
    if canonical_hash(selection["choices"]) != selection["selection_hash"]:
        raise ValueError("selection_commitment_drift")
    data, indices = base.registry(cohort, scores)
    rows = []
    for choice in selection["choices"]:
        policy = base.POLICIES.index((choice["arm"], choice["seed"]))
        rep = trace["replicates"][choice["replicate"]]
        for g in base.GROUPS:
            for name, alpha in ALPHAS.items():
                events = []
                for look in LOOKS if name == "sequential" else [960]:
                    sample = [indices[g][i] for i in rep["groups"][g]["certification"][:look]]
                    counts = base.count(data, truth, sample, policy, choice["cutoffs"][g])
                    bound = base.old.upper(counts["accepted"], counts["errors"], alpha)
                    issued = counts["accepted"] > 0 and bound <= 0.05
                    events.append(
                        dict(
                            candidate_draws=look,
                            **counts,
                            alpha=alpha,
                            risk_upper=bound,
                            certificate_issued=issued,
                        )
                    )
                    if issued:
                        break
                last = events[-1]
                rows.append(
                    dict(
                        unit_id=f"{rep['replicate']}/{name}/{choice['arm']}/{choice['seed']}/{g}",
                        cluster_id=f"rep/{rep['replicate']}",
                        replicate=rep["replicate"],
                        protocol=name,
                        arm=choice["arm"],
                        seed=choice["seed"],
                        group=g,
                        cutoff=choice["cutoffs"][g],
                        looks=events,
                        **last,
                        selection_labels=240,
                        total_labels=240 + last["candidate_draws"],
                        metric=240 + last["candidate_draws"],
                        status="completed",
                        missing_reason=None,
                        refusal_reason=None if issued else "no_valid_certificate_at_final_look",
                    )
                )
        progress("certification", len(rows), len(selection["choices"]) * 8 - len(rows))
    shared, standalone = [], []
    for rep in trace["replicates"]:
        local = [x for x in rows if x["replicate"] == rep["replicate"]]
        union = {
            g: max(
                x["candidate_draws"]
                for x in local
                if x["group"] == g and x["protocol"] == "sequential"
            )
            for g in base.GROUPS
        }
        shared.append(
            dict(
                replicate=rep["replicate"],
                sequential_terminal_draws=union,
                sequential_total_labels=960 + sum(union.values()),
                fixed_total_labels=4800,
                actual_joint_research_labels=4800,
            )
        )
        for arm, seed in base.POLICIES:
            standalone.append(
                dict(
                    replicate=rep["replicate"],
                    arm=arm,
                    seed=seed,
                    **{
                        name: sum(
                            x["total_labels"]
                            for x in local
                            if (x["arm"], x["seed"], x["protocol"]) == (arm, seed, name)
                        )
                        for name in ALPHAS
                    },
                )
            )
    return dict(
        rows=rows,
        standalone_costs=standalone,
        shared_union_costs=shared,
        actual_research_labels=4800 * len(trace["replicates"]),
        numerical_training_calls=0,
    )


def controls() -> list[Json]:
    """Independent inversion and constant policies expose invalid stopping mathematics."""
    from scipy.stats import binomtest

    controls = []
    for name, alpha in ALPHAS.items():
        floor = minimum(alpha)
        for n, errors in [(0, 0), (floor - 1, 0), (floor, 0), (960, 960)]:
            bound = base.old.upper(n, errors, alpha)
            independent = (
                binomtest(errors, n, alternative="less").proportion_ci(1 - alpha).high if n else 1.0
            )
            controls.append(
                dict(
                    control=f"{name}/{n}/{errors}",
                    n=n,
                    errors=errors,
                    risk_upper=bound,
                    independent_upper=independent,
                    passed=abs(bound - independent) < 1e-10 and (bound <= 0.05) == (n == floor),
                )
            )
    units = [dict(unit_id=g, group=g) for g in base.GROUPS]
    scores = [
        dict(
            unit_id=g,
            cluster_id=g,
            arm=a,
            seed=s,
            candidate="support",
            probabilities=[0.96, 0.02, 0.02],
        )
        for a, s in base.POLICIES
        for g in base.GROUPS
    ]
    truth = dict.fromkeys(base.GROUPS, "support")
    cohort, trace = dict(inputs=units), draws(1, replicates=1)
    selection = select(cohort, scores, truth, trace)
    safe = certify(cohort, scores, truth, trace, selection)
    unsafe = certify(cohort, scores, dict.fromkeys(truth, "refute"), trace, selection)
    from carnot.reporting import sequential_certificate_methods_8880 as source

    twin = namespace(source, "sequential8880_identical_protocols")
    twin.LOOKS = [960]
    twin.ALPHAS = dict.fromkeys(ALPHAS, ALPHAS["fixed"])
    identical = twin.certify(cohort, scores, truth, trace, selection)["rows"]
    changed = dict(
        trace,
        replicates=[
            dict(
                trace["replicates"][0],
                groups={
                    g: dict(v, certification=[0] * 960)
                    for g, v in trace["replicates"][0]["groups"].items()
                },
            )
        ],
    )
    controls.extend(
        [
            dict(
                control="identical_policy_arms",
                passed=len(
                    {x["candidate_draws"] for x in safe["rows"] if x["protocol"] == "sequential"}
                )
                == 1
                and all(x["certificate_issued"] for x in safe["rows"]),
            ),
            dict(
                control="unsafe_constant_repeated_look_rejection",
                passed=all(
                    not x["certificate_issued"] and x["candidate_draws"] == 960
                    for x in unsafe["rows"]
                ),
            ),
            dict(
                control="selection_isolation",
                passed=selection == select(cohort, scores, truth, changed),
            ),
            dict(
                control="same_schedule_identical_protocol_control",
                passed=all(
                    a["looks"] == b["looks"] and a["total_labels"] == b["total_labels"]
                    for a, b in zip(identical[::2], identical[1::2])
                ),
            ),
        ]
    )
    return controls
