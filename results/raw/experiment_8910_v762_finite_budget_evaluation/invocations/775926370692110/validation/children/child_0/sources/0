"""REQ-REPORT-8910: count fetched service labels independently of evaluator work.

The exact constructed truth measures this population. It cannot establish
independent natural-language generalization or a rare-event bound empirically.
"""

from collections.abc import Callable
from typing import Any

from carnot.reporting import finite_budget_methods_8906 as frozen
from carnot.reporting import sequential_label_cost_8885 as cost
from carnot.reporting import shared_terminal_methods_8893 as shared
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
Fetch = Callable[[str, int, str, str, list[Json]], Json]
__all__ = ["frozen", "shared"]
ROOT, SEED, base = frozen.ROOT, frozen.SEED, shared.base
NAME, TASK = "experiment_8910_v762_finite_budget_evaluation", "exp8910-finite-budget-evaluation"
CLI = f"scripts/experiments/{NAME}.py"
OWNED = [
    "python/carnot/reporting/finite_budget_evaluation_8910.py",
    "python/carnot/reporting/finite_budget_evaluation_runner_8910.py",
    CLI,
]
TESTS = ["tests/python/test_finite_budget_evaluation_8910.py"]
ARMS = ["fixed_final", "early_acceptance", "early_acceptance_refusal"]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report completed work without extending the fixed deadline."""
    print(f"[exp8910] phase={phase} completed={completed} pending={pending}", flush=True)


def protocol() -> Json:
    """Preserve the registered tests and predeclare the sole shared benefit contrast."""
    return dict(
        methods=frozen.protocol(),
        seed=SEED,
        arms=ARMS,
        primary_contrast="shared_union/label_savings",
        interval_contract=frozen.protocol()["interval_contract"],
        access_order=[
            "protocol",
            "operands",
            "selection_labels",
            "selection",
            "certificates",
            "census_labels",
            "evaluation",
        ],
        service_arm=ARMS[2],
        risk_ceiling=0.05,
        census_scope="evaluator only; after all service decisions",
        shared_benefit="primary lower99>0; zero decision or inclusion violations; valid controls; every served census risk<=.05",
        family_error="84/16800+28/5600=.01 per replicate; unchanged tests; forty replicates do not prove a rare-event bound",
    )


def certify(cohort: Json, scores: list[Json], trace: Json, selection: Json, fetch: Fetch) -> Json:
    """Fetch each group's next prefix only while at least one policy remains unresolved."""
    if canonical_hash(selection["choices"]) != selection["selection_hash"]:
        raise ValueError("selection_commitment_drift")
    data, indices = base.registry(cohort, scores)
    units = {u["unit_id"]: u for u in cohort["inputs"]}
    rows, ledger, shared = [], [], []
    for rep in trace["replicates"]:
        number = rep["replicate"]
        choices = [x for x in selection["choices"] if x["replicate"] == number]
        if [(x["arm"], x["seed"]) for x in choices] != base.POLICIES:
            raise ValueError("selection_policy_roster")
        for arm in ARMS:
            prefixes = {}
            for group in base.GROUPS:
                stream = rep["groups"][group]
                draws = stream["certification"]
                if len(draws) != 960 or any(
                    type(i) is not int or not 0 <= i < len(indices[group])
                    for i in draws + stream["selection"]
                ):
                    raise ValueError("candidate_draw_contract")
                statistics: list[list[tuple[int, int]]] = [[] for _ in choices]
                stops: dict[int, Json] = {}
                previous, evaluations = 0, 0
                for look in [960] if arm == ARMS[0] else frozen.LOOKS:
                    sample = [indices[group][i] for i in draws[previous:look]]
                    requested = [units[data[0][i]] for i in sample]
                    labels = fetch(arm, number, group, "certification", requested)
                    if set(labels) != {u["unit_id"] for u in requested}:
                        raise ValueError("label_fetch_contract")
                    for policy, choice in enumerate(choices):
                        if policy in stops:
                            continue
                        counts = base.count(data, labels, sample, policy, choice["cutoffs"][group])
                        n, e = statistics[policy][-1] if statistics[policy] else (0, 0)
                        current = (n + counts["accepted"], e + counts["errors"])
                        statistics[policy].append(current)
                        evaluations += len(sample)
                        # Future placeholders contain no new labels; only the present event is used.
                        complete = (
                            [(min(current[0], l), min(current[1], l)) for l in frozen.LOOKS]
                            if arm == ARMS[0]
                            else statistics[policy] + [current] * (4 - len(statistics[policy]))
                        )
                        path = frozen.decide(complete, fixed=arm == ARMS[0], refusal=arm == ARMS[2])
                        event = path["looks"][0 if arm == ARMS[0] else frozen.LOOKS.index(look)]
                        if event["reason"] != "pending":
                            stops[policy] = dict(
                                event,
                                looks=path["looks"][: len(statistics[policy])],
                                policy=policy,
                                cutoff=choice["cutoffs"][group],
                            )
                    previous = look
                    if len(stops) == 7:
                        break
                prefixes[group] = previous
                selection_scans = 2 * 7 * 240 * len(base.CUTOFFS)
                ledger.append(
                    dict(
                        replicate=number,
                        group=group,
                        protocol=arm,
                        selection_labels=240,
                        selection_fetch_requests=240 if arm == ARMS[2] else 0,
                        actual_label_requests=previous + (240 if arm == ARMS[2] else 0),
                        certification_labels=previous,
                        total_labels=240 + previous,
                        distinct_worlds=len(set(stream["selection"] + draws[:previous])),
                        selection_draw_indices=stream["selection"],
                        candidate_draw_indices=draws[:previous],
                        replacement_repeats_paid=True,
                        selection_policy_input_evaluations=selection_scans,
                        certification_policy_input_evaluations=evaluations,
                        policy_input_evaluations=selection_scans + evaluations,
                    )
                )
                for policy, stop in stops.items():
                    a, seed = base.POLICIES[policy]
                    rows.append(
                        dict(
                            stop,
                            unit_id=f"{number}/{arm}/{policy}/{group}",
                            cluster_id=f"rep/{number}",
                            replicate=number,
                            protocol=arm,
                            arm=a,
                            seed=seed,
                            group=group,
                            metric=240 + stop["candidate_draws"],
                            denominator=960,
                            selection_labels=240,
                            selection_policy_input_evaluations=2 * 240 * len(base.CUTOFFS),
                            certification_policy_input_evaluations=stop["candidate_draws"],
                            policy_input_evaluations=2 * 240 * len(base.CUTOFFS)
                            + stop["candidate_draws"],
                            total_labels=240 + stop["candidate_draws"],
                            final_prefix_hash=canonical_hash(draws),
                            status="completed",
                            missing_reason=None,
                        )
                    )
            shared.append(
                dict(
                    replicate=number,
                    protocol=arm,
                    group_prefixes=prefixes,
                    total_labels=960 + sum(prefixes.values()),
                )
            )
        progress("certification_replicates", number + 1, len(trace["replicates"]) - number - 1)
    return dict(
        rows=rows,
        shared_union_costs=shared,
        paid_label_ledger=[x for x in ledger if x["protocol"] == ARMS[2]],
        evaluator_label_ledger=[x for x in ledger if x["protocol"] != ARMS[2]],
        numerical_training_calls=0,
    )


def evaluate(cohort: Json, scores: list[Json], truth: Json, trace: Json, cert: Json) -> Json:
    """Check exact decisions before reducing service savings and post-decision census risk."""
    data, indices = base.registry(cohort, scores)
    rows, census, comparisons, standalone = [], [], [], []
    for stop in cert["rows"]:
        counts = base.count(data, truth, indices[stop["group"]], stop["policy"], stop["cutoff"])
        rows.append(
            dict(
                stop,
                census_denominator=counts["denominator"],
                served_accepted=counts["accepted"] if stop["certificate_issued"] else 0,
                served_errors=counts["errors"] if stop["certificate_issued"] else 0,
                served_risk=counts["risk"] if stop["certificate_issued"] else None,
            )
        )
    keyed = {(x["replicate"], x["group"], x["policy"], x["protocol"]): x for x in rows}
    inclusion, equality = 0, 0
    for early in [x for x in rows if x["protocol"] == ARMS[1]]:
        key = early["replicate"], early["group"], early["policy"]
        fixed, refusal = keyed[(*key, ARMS[0])], keyed[(*key, ARMS[2])]
        same = (
            early["certificate_issued"] == refusal["certificate_issued"]
            and early["served_accepted"] == refusal["served_accepted"]
        )
        included = not fixed["certificate_issued"] or early["certificate_issued"]
        passed = (
            same
            and included
            and refusal["candidate_draws"] <= early["candidate_draws"]
            and all(
                x["cutoff"] == early["cutoff"]
                and x["final_prefix_hash"] == early["final_prefix_hash"]
                for x in [fixed, refusal]
            )
            and (
                not refusal["certificate_issued"]
                or refusal["candidate_draws"] == early["candidate_draws"]
            )
        )
        inclusion += int(not included)
        equality += int(not same)
        comparisons.append(
            dict(
                unit_id=early["unit_id"] + "/comparison",
                cluster_id=early["cluster_id"],
                arm=early["arm"],
                seed=early["seed"],
                metric=int(passed),
                denominator=1,
                status="completed",
                missing_reason=None,
                passed=passed,
                fixed_certificate=fixed["certificate_issued"],
                early_certificate=early["certificate_issued"],
                refusal_certificate=refusal["certificate_issued"],
            )
        )
    if not all(x["passed"] for x in comparisons):
        raise ValueError("decision_equality_or_inclusion")
    for ledger in cert["paid_label_ledger"] + cert["evaluator_label_ledger"]:
        local = [
            x
            for x in rows
            if (x["replicate"], x["group"], x["protocol"])
            == (ledger["replicate"], ledger["group"], ledger["protocol"])
        ]
        if ledger["certification_labels"] != max(x["candidate_draws"] for x in local) or ledger[
            "certification_policy_input_evaluations"
        ] != sum(x["candidate_draws"] for x in local):
            raise ValueError("ledger_policy_cost_drift")
    for rep in trace["replicates"]:
        number = rep["replicate"]
        for arm, seed in base.POLICIES:
            standalone.append(
                dict(
                    replicate=number,
                    arm=arm,
                    seed=seed,
                    **{
                        name: sum(
                            x["total_labels"]
                            for x in rows
                            if (x["replicate"], x["arm"], x["seed"], x["protocol"])
                            == (number, arm, seed, name)
                        )
                        for name in ARMS
                    },
                )
            )
            for name in ARMS:
                cells = {
                    x["group"]: dict(
                        accepted=x["served_accepted"],
                        errors=x["served_errors"],
                        denominator=x["census_denominator"],
                    )
                    for x in rows
                    if (x["replicate"], x["arm"], x["seed"], x["protocol"])
                    == (number, arm, seed, name)
                }
                for mix, weights in base.MIXTURES.items():
                    observed = base.mixture(cells, weights)
                    census.append(
                        dict(
                            unit_id=f"{number}/{name}/{arm}/{seed}/{mix}",
                            cluster_id=f"rep/{number}",
                            replicate=number,
                            protocol=name,
                            arm=arm,
                            seed=seed,
                            mixture=mix,
                            metric=observed["coverage"],
                            served_risk=observed["risk"],
                            error_mass=observed["error_mass"],
                            denominator=len(data[0]),
                            status="completed",
                            missing_reason=None,
                        )
                    )
        progress("census_replicates", number + 1, len(trace["replicates"]) - number - 1)
    shared_pairs = []
    for rep in trace["replicates"]:
        local_costs = {
            x["protocol"]: x
            for x in cert["shared_union_costs"]
            if x["replicate"] == rep["replicate"]
        }
        shared_pairs.append(
            dict(
                replicate=rep["replicate"],
                fixed_total_labels=local_costs[ARMS[1]]["total_labels"],
                sequential_total_labels=local_costs[ARMS[2]]["total_labels"],
            )
        )
    # The established reducer pairs replicates and keeps five learned seeds inside each mean.
    summary = cost.summarize(
        dict(
            rows=rows,
            standalone_costs=[dict(x, fixed=x[ARMS[1]], sequential=x[ARMS[2]]) for x in standalone],
            shared_union_costs=shared_pairs,
            census_risk_rows=[
                dict(x, protocol="fixed" if x["protocol"] == ARMS[1] else "sequential")
                for x in census
                if x["protocol"] != ARMS[0]
            ],
        )
    )
    summary.pop("label_savings_score")
    intervals = {x["contrast"]: x for x in summary["paired_intervals"]}
    safe = all(x["served_risk"] is None or x["served_risk"] <= 0.05 for x in rows)
    return dict(
        cert,
        **summary,
        rows=rows,
        stopping_rows=cert["rows"],
        standalone_costs=standalone,
        census_risk_rows=census,
        decision_comparison_rows=comparisons,
        pathwise_inclusion_violations=inclusion,
        decision_equality_violations=equality,
        theoretical_family_error=frozen.protocol()["alpha_allocation"],
        shared_union_benefit_score=int(
            safe and intervals["shared_union/label_savings"]["lower_99"] > 0
        ),
        standalone_label_benefit_score=int(
            safe and intervals["learned_energy/label_savings"]["lower_99"] > 0
        ),
        evaluator_census_labels=len(data[0]),
        evaluator_label_ledger=cert["evaluator_label_ledger"]
        + [
            dict(
                protocol="post_decision_census",
                selection_labels=0,
                selection_fetch_requests=0,
                certification_labels=0,
                actual_label_requests=len(data[0]),
                distinct_worlds=len(data[0]),
                policy_input_evaluations=sum(x["census_denominator"] for x in rows),
                scope="evaluator only; shared truth reused across replicate risk reductions",
            )
        ],
    )
