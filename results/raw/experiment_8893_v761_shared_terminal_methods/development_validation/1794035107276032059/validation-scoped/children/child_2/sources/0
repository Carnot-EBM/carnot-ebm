"""REQ-REPORT-8893: share the final event to prove inclusion for every draw path.

This local finite-look construction uses an exact constructed oracle. Its
statistical guarantee assumes stationary independent draws within each group.
"""

import json
from typing import Any

from carnot.reporting import sequential_certificate_methods_8880 as old
from carnot.reporting import sequential_label_cost_8885 as cost
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
base, ROOT = old.base, old.ROOT
NAME, TASK = "experiment_8893_v761_shared_terminal_methods", "exp8893-shared-terminal-methods"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/shared_terminal_methods_8893.py",
    "python/carnot/reporting/shared_terminal_runner_8893.py",
    CLI,
]
TESTS = ["tests/python/test_shared_terminal_methods_8893.py"]
SEED, LOOKS = 7618893, [120, 240, 480, 960]
EARLY, FINAL, SENSITIVITY = 0.005 / (7 * 4 * 3), 0.005 / (7 * 4), 0.01 / (7 * 4)
ALPHAS = dict(sequential=EARLY, fixed=FINAL)
INCLUSION = (
    "For every possible candidate sequence and frozen selected cutoff, let F be "
    "the common final test and E1,E2,E3 the early tests. Fixed certifies iff F. "
    "Sequential certifies iff E1 or E2 or E3 or F. Therefore F implies sequential "
    "certification, even when sequential stopped earlier. The certified frozen "
    "policy serves the same input-only accepted set, so fixed served sets are "
    "subsets of sequential served sets in every group and every nonnegative mixture. "
    "This logical inclusion requires no probabilistic assumption and says nothing "
    "about correctness of answers. Risk validity separately requires iid stationary "
    "within-group draws independent of selection and policies."
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Expose completed work without changing the fixed execution deadline."""
    print(f"[exp8893] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Freeze both comparisons before any selection or population outcomes are opened."""
    p = old.protocol()
    p.update(
        seed=SEED,
        alpha_allocation=dict(
            total=0.01,
            early_per_event=EARLY,
            final_per_event=FINAL,
            early_event_count=84,
            final_event_count=28,
            distinct_total=0.01,
            shared_final_counted_once=True,
            scope="per replicate conditional on frozen selection",
            exact_fraction_sum="84/16800 + 28/5600 = 1/100",
        ),
        coverage_inclusion_proof=INCLUSION,
        fixed_final_test=dict(
            candidate_draws=960,
            alpha=FINAL,
            risk_upper_le=0.05,
            accepted_n_zero="refuse",
            identical_prefix=True,
        ),
        fixed_alpha_sensitivity=dict(
            alpha=SENSITIVITY, descriptive_only=True, excluded_from_joint_guarantee=True
        ),
        exact_sample_requirements=dict(early=old.minimum(EARLY), final=old.minimum(FINAL)),
        bound_proof=(
            "Condition on frozen policies, population and independent selection. "
            "Accepted errors conditional on accepted n are binomial at the fixed "
            "group risk. Each exact one-sided bound undercovers with probability "
            "at most its alpha; n=0 refuses. Union bound over 84 early events "
            "and 28 distinct shared final events is .005+.005=.01. Stopping "
            "only selects events in this family. Shared or dependent policy "
            "tests need no independence. This is a per-replicate guarantee, "
            "not a guarantee across all forty replicates or nonstationary draws. "
            "This is a new local construction, not the cited paper theorem."
        ),
        interval_contract=dict(
            family_size=23,
            family_error=0.01,
            confidence_percent=99,
            method="Bonferroni bounded Hoeffding",
            label_savings_range=[0, 3360],
            coverage_difference_range=[-1, 1],
            primary_family="learned_energy",
            coverage_gate="pathwise inclusion proof, not descriptive interval",
        ),
        access_order=p["access_order"] + ["census_labels", "evaluation"],
    )
    return p


def freeze(exposed: set[str], *, count: int = 1024) -> Json:
    """New population structures are rejected before labels, preserving the old fit."""
    builder = namespace(old, "shared8893_population")
    builder.SEED = SEED
    value = builder.freeze(exposed, count=count)
    return dict(json.loads(json.dumps(value).replace("v760/8880/", "v761/8893/")))


def draws(size: int, *, replicates: int = 40) -> Json:
    """Independent group/stage streams pair protocols without erasing paid repeats."""
    builder = namespace(old, "shared8893_draws")
    builder.SEED = SEED
    return dict(builder.draws(size, replicates=replicates))


select = old.select


def common_final(a: float, b: float, left: list[int], right: list[int]) -> bool:
    """Equal spending and the same ordered full prefix are premises of inclusion."""
    return a == b == FINAL and len(left) == len(right) == 960 and left == right


def decision_path(statistics: list[tuple[int, int]]) -> list[Json]:
    """Validate cumulative counts before using unseen-error bounds at each candidate look."""
    if len(statistics) != 4:
        raise ValueError("sufficient_statistic_schedule")
    events, previous = [], (0, 0, 0)
    for look, (n, errors) in zip(LOOKS, statistics):
        last_n, last_errors, last_look = previous
        if (
            type(n) is not int
            or type(errors) is not int
            or not 0 <= errors <= n <= look
            or not 0 <= errors - last_errors <= n - last_n <= look - last_look
        ):
            raise ValueError("sufficient_statistic_transition")
        alpha = FINAL if look == 960 else EARLY
        upper = base.old.upper(n, errors, alpha)
        events.append(
            dict(
                candidate_draws=look,
                accepted=n,
                errors=errors,
                denominator=look,
                risk=errors / n if n else None,
                alpha=alpha,
                risk_upper=upper,
                certificate_issued=n > 0 and upper <= 0.05,
            )
        )
        previous = n, errors, look
    return events


def ledgers(rows: list[Json]) -> tuple[list[Json], list[Json]]:
    """An unfinished policy consumes the full prefix even when other policies stop early."""
    standalone, shared = [], []
    for rep in sorted({x["replicate"] for x in rows}):
        local = [x for x in rows if x["replicate"] == rep]
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
                replicate=rep,
                sequential_terminal_draws=union,
                sequential_total_labels=960 + sum(union.values()),
                fixed_total_labels=4800,
                actual_joint_research_labels=4800,
            )
        )
        for arm, seed in base.POLICIES:
            standalone.append(
                dict(
                    replicate=rep,
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
    return standalone, shared


def certify(cohort: Json, scores: list[Json], truth: Json, trace: Json, selection: Json) -> Json:
    """Use one final event in both arms; sensitivity cannot change the registered comparator."""
    if canonical_hash(selection["choices"]) != selection["selection_hash"]:
        raise ValueError("selection_commitment_drift")
    data, indices = base.registry(cohort, scores)
    rows, sensitivity = [], []
    for choice in selection["choices"]:
        policy = base.POLICIES.index((choice["arm"], choice["seed"]))
        rep = choice["replicate"]
        for g in base.GROUPS:
            stream = trace["replicates"][rep]["groups"][g]["certification"]
            if not common_final(FINAL, FINAL, stream, stream):
                raise ValueError("candidate_prefix_contract")
            stats = [
                base.count(
                    data,
                    truth,
                    [indices[g][i] for i in stream[:look]],
                    policy,
                    choice["cutoffs"][g],
                )
                for look in LOOKS
            ]
            events = decision_path([(s["accepted"], s["errors"]) for s in stats])
            terminal = events[-1]
            stopped = next((i for i, e in enumerate(events) if e["certificate_issued"]), 3)
            for name, visited in [("sequential", events[: stopped + 1]), ("fixed", [terminal])]:
                last = visited[-1]
                rows.append(
                    dict(
                        unit_id=f"{rep}/{name}/{choice['arm']}/{choice['seed']}/{g}",
                        cluster_id=f"rep/{rep}",
                        replicate=rep,
                        protocol=name,
                        arm=choice["arm"],
                        seed=choice["seed"],
                        group=g,
                        cutoff=choice["cutoffs"][g],
                        looks=visited,
                        scheduled_looks=events if name == "sequential" else [terminal],
                        final_prefix_hash=canonical_hash(stream),
                        **last,
                        selection_labels=240,
                        total_labels=240 + last["candidate_draws"],
                        metric=240 + last["candidate_draws"],
                        status="completed",
                        missing_reason=None,
                    )
                )
            bound = base.old.upper(terminal["accepted"], terminal["errors"], SENSITIVITY)
            sensitivity.append(
                dict(
                    rows[-1],
                    unit_id=rows[-1]["unit_id"] + "/sensitivity",
                    protocol="fixed_alpha_sensitivity",
                    alpha=SENSITIVITY,
                    risk_upper=bound,
                    scheduled_looks=[],
                    looks=[],
                    certificate_issued=terminal["accepted"] > 0 and bound <= 0.05,
                    descriptive_only=True,
                )
            )
        progress("certification", len(rows), len(selection["choices"]) * 8 - len(rows))
    standalone, shared = ledgers(rows)
    return dict(
        rows=rows,
        standalone_costs=standalone,
        shared_union_costs=shared,
        fixed_alpha_sensitivity_rows=sensitivity,
        actual_research_labels=4800 * len(trace["replicates"]),
        numerical_training_calls=0,
    )


def evaluate(cohort: Json, scores: list[Json], truth: Json, trace: Json, cert: Json) -> Json:
    """Open census truth only after certificates, keeping exact coverage separate from risk validity."""
    data, indices = base.registry(cohort, scores)
    rows: list[Json] = []
    sensitivity: list[Json] = []
    census: list[Json] = []
    violations: list[Json] = []
    for source, target in [
        (cert["rows"], rows),
        (cert["fixed_alpha_sensitivity_rows"], sensitivity),
    ]:
        for stop in source:
            policy = base.POLICIES.index((stop["arm"], stop["seed"]))
            counts = base.count(data, truth, indices[stop["group"]], policy, stop["cutoff"])
            issued = stop["certificate_issued"]
            target.append(
                dict(
                    stop,
                    census_denominator=counts["denominator"],
                    population_accepted=counts["accepted"],
                    population_errors=counts["errors"],
                    population_risk=counts["risk"],
                    served_accepted=counts["accepted"] if issued else 0,
                    served_errors=counts["errors"] if issued else 0,
                    served_risk=counts["risk"] if issued else None,
                )
            )
    inclusion = sum(
        b["certificate_issued"] and not a["certificate_issued"]
        for a, b in zip(rows[::2], rows[1::2])
    )
    for rep in trace["replicates"]:
        local = [x for x in rows if x["replicate"] == rep["replicate"]]
        violations.append(
            dict(
                replicate=rep["replicate"],
                violation=any(
                    x["population_risk"] is not None and x["population_risk"] > e["risk_upper"]
                    for x in local
                    if x["protocol"] == "sequential"
                    for e in x["scheduled_looks"]
                ),
            )
        )
        for name in ALPHAS:
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
                            unit_id=f"{rep['replicate']}/{name}/{arm}/{seed}/{mixture}",
                            cluster_id=f"rep/{rep['replicate']}",
                            replicate=rep["replicate"],
                            protocol=name,
                            arm=arm,
                            seed=seed,
                            mixture=mixture,
                            metric=observed["coverage"],
                            served_risk=observed["risk"],
                            error_mass=observed["error_mass"],
                            denominator=len(data[0]),
                            status="completed",
                            missing_reason=None,
                        )
                    )
        progress("census", rep["replicate"] + 1, len(trace["replicates"]) - rep["replicate"] - 1)
    result = dict(
        rows=rows,
        stopping_rows=cert["rows"],
        census_risk_rows=census,
        fixed_alpha_sensitivity_rows=sensitivity,
        pathwise_inclusion_violations=inclusion,
        family_violation_rows=violations,
        observed_family_violation_count=sum(v["violation"] for v in violations),
        standalone_costs=cert["standalone_costs"],
        standalone_learned_policy_costs=[
            x for x in cert["standalone_costs"] if x["arm"] == "learned_energy"
        ],
        shared_union_costs=cert["shared_union_costs"],
        actual_label_ledger=dict(
            joint_calibration_labels=cert["actual_research_labels"],
            selection_labels=960 * len(trace["replicates"]),
            certification_labels=3840 * len(trace["replicates"]),
            census_oracle_labels=len(data[0]),
            current_training_labels=0,
            paid_draw_counts_include_repeats=True,
            fresh_external_annotations=0,
        ),
        numerical_training_calls=0,
    )
    summary = cost.summarize(result)
    saving = next(
        x for x in summary["paired_intervals"] if x["contrast"] == "learned_energy/label_savings"
    )
    summary["label_savings_score"] = int(
        saving["lower_99"] > 0
        and inclusion == 0
        and all(x["served_risk"] is None or x["served_risk"] <= 0.05 for x in rows)
    )
    return dict(result, **summary)


def controls() -> list[Json]:
    """Boundary and mutation controls qualify the proof premises without a savings claim."""
    from scipy.stats import binomtest

    rows = []
    for alpha in [EARLY, FINAL]:
        floor = old.minimum(alpha)
        for n, errors in [(0, 0), (floor - 1, 0), (floor, 0), (960, 960)]:
            upper = base.old.upper(n, errors, alpha)
            check = (
                binomtest(errors, n, alternative="less").proportion_ci(1 - alpha).high if n else 1
            )
            rows.append(
                dict(
                    control=f"exact/{alpha}/{n}/{errors}",
                    passed=abs(upper - check) < 1e-10 and (upper <= 0.05) == (n == floor),
                )
            )
    rows.extend(
        [
            dict(
                control="altered_final_alpha_rejected",
                passed=not common_final(EARLY, FINAL, [0] * 960, [0] * 960),
            ),
            dict(
                control="altered_final_prefix_rejected",
                passed=not common_final(FINAL, FINAL, [0] * 960, [1] * 960),
            ),
            dict(
                control="unsafe_all_error_path_refuses",
                passed=not any(
                    e["certificate_issued"] for e in decision_path([(n, n) for n in LOOKS])
                ),
            ),
            dict(
                control="empty_path_refuses",
                passed=not any(e["certificate_issued"] for e in decision_path([(0, 0)] * 4)),
            ),
        ]
    )
    return rows
