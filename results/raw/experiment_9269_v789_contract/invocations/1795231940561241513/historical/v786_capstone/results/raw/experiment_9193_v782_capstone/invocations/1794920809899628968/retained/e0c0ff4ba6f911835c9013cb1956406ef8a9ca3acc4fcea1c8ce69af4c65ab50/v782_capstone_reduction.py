"""REQ-REPORT-9193-V782: derive closure from independent rows, not positive gates."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import yaml

from carnot.reporting import human_contrast_9141 as human
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v778_capstone_evidence import bounds

Json = dict[str, Any]


def reduce(work: Json) -> Json:
    """Unavailable scientific branches retain their original units and undefined bounds."""
    from carnot.reporting import v782_capstone_evidence as e

    require(e.canonical_hash(work["tasks"]) == e.TASK_PIN, "task_authority")
    prior = work["values"]["prior:9179"]["value"]
    source = work["values"][e.TASKS[0]]["value"]
    diagnosis = work["values"][e.TASKS[1]]["value"]
    arc = work["values"][e.TASKS[12]]["value"]
    labels = [
        dict(s, qid=r["qid"], role=r["role"], group_id=r["group_id"])
        for r in prior["human_label_rows"]
        for s in r["siblings"]
    ]
    human_rows = human.group_rows(labels)
    require(human_rows == prior["human_label_rows"], "human_rows")
    cost = deepcopy(prior["cost_summary"])
    counts = Counter(r["status"] for r in cost["rows"])
    cost_counts = dict(
        intended=len(cost["rows"]),
        attempted=len(cost["rows"]) - counts["unstarted"],
        **{k: counts[k] for k in ["completed", "failed", "censored", "unstarted"]},
    )
    require(cost_counts == cost["scientific_request_counts"], "cost_aggregate")
    donors = prior["donor_label_change_rows"]
    changed, support = sum(r["numerator"] for r in donors), sum(r["denominator"] for r in donors)
    require(
        support > 0 and changed / support == prior["donor_changed_label_fraction"],
        "donor_aggregate",
    )

    def retained(ref: Json) -> Json:
        return dict(json.loads(Path(work["operands"][ref["path"]]["path"]).read_bytes()))

    rosters = [retained(source[f"shard_{s}_roster_reference"]) for s in "abc"]
    units = [r for roster in rosters for r in roster["rows"]]
    original = retained(source["inference_view_reference"])["rows"]
    require(
        len(units) == len({r["unit_id"] for r in units}) == 900
        and sorted(units, key=lambda r: r["unit_id"])
        == sorted(original, key=lambda r: r["unit_id"]),
        "partition_union",
    )
    require([len(r["qids"]) for r in rosters] == [100] * 3, "partition_size")
    obligations = source["validation_obligations"]
    union = {r["node_id"]: deepcopy(r) for r in diagnosis["remaining_obligations"]}
    original_ids = obligations["original_node_ids"] + obligations["v779_addition_ids"]
    require(len(set(original_ids)) == 674 and set(original_ids) <= union.keys(), "obligation_union")
    executed = {r["unit_id"]: r for r in diagnosis["rows"]}
    require(len(executed) == len(diagnosis["rows"]), "obligation_once")
    arc_rows = arc["rows"]
    require(len(arc_rows) == len({r["unit_id"] for r in arc_rows}) == 124, "arc_units")
    require(
        all(r["numerator"] in {0, 1} and r["denominator"] == 1 for r in arc_rows),
        "arc_denominators",
    )
    before = {r["task_id"]: r for r in [*prior["historical_outcomes"], *prior["outcome_rows"]]}
    after = {r["task_id"]: r for r in e.outcomes(work)}
    manifest = yaml.safe_load(
        Path(work["snapshots"]["ops/exclusion_manifest.yaml"]["path"]).read_bytes()
    )
    retirements = []
    for task in work["tasks"]:
        for old in task.get("prior_failures", []):
            require(
                set(old) >= {"experiment_id", "verdict", "addressed_by", "retire_if_same_verdict"},
                "prior_failure_fields",
            )
            previous, current = before.get(old["experiment_id"], {}), after.get(task["id"], {})
            key = (
                "scheduler_honest_verdict"
                if old["verdict"].startswith("blocked_")
                else "accounting_honest_verdict"
            )
            actual, verdict = previous.get(key), current.get(key)
            same = bool(
                old["retire_if_same_verdict"] is True and actual == old["verdict"] == verdict
            )
            retirements.append(
                dict(
                    task_id=task["id"],
                    prior_task_id=old["experiment_id"],
                    prior_failure=deepcopy(old),
                    declared_prior_verdict=old["verdict"],
                    actual_prior_verdict=actual,
                    current_verdict=verdict,
                    prior_verdict_authenticated=actual == old["verdict"],
                    condition_met=same,
                    retire_if_same_verdict=old["retire_if_same_verdict"],
                    recommendation="retire_unchanged_execution_scope"
                    if same
                    else "no_exact_authenticated_retirement",
                    next_reopen_condition=old["addressed_by"],
                    hypothesis_retired=False,
                    manifest_mutated=False,
                    manifest_retirement_applied=same
                    and any(
                        r.get("experiment_id") == task["id"]
                        and old["experiment_id"]
                        in [r.get("prior_task_id"), *r.get("prior_task_ids", [])]
                        and r.get("exact_verdict") == verdict
                        for r in manifest["retired_extras"]
                    ),
                    prior_reference=previous.get("reference"),
                    current_reference=current.get("reference"),
                )
            )
    decisions = [
        dict(branch=b, disposition=d, evidence=proof, next_reopen_condition=condition)
        for b, d, proof, condition in [
            (
                "checker_correction",
                "retire_unchanged_failed_execution",
                "Exp9181 remains disqualified; structural observations do not qualify the complete obligation union",
                "Demonstrate a changed cause or prerequisite; preserve exact checker flags and original assertions; complete all affected controls, immutable receipts, sanitized-child observation and100 percent owned coverage",
            ),
            (
                "forward_validation",
                "retire_unchanged_failed_execution",
                "Exp9182 scheduler refusal repeats authenticated Exp9168 zero gate; retired task IDs never become dependencies",
                "A separately qualified changed correction with causal_fix_ready_score=1, followed by complete original674 plus all affected obligations in one current revision",
            ),
            (
                "collection",
                "terminal_blocked",
                "All three scientific shards and runtime primaries are absent",
                "Qualified forwards and measured complete100-qid/300-answer budgets for each of three shards; admit exactly900 original answers and original roles",
            ),
            (
                "static_energy",
                "terminal_blocked_unmeasured",
                "No static prediction rows; both original bounds undefined",
                "Complete admitted cohort and both unchanged static Brier bounds with defined nonworsening accepted risk; a complete null retires only that measured method",
            ),
            (
                "online_energy",
                "terminal_blocked_unmeasured",
                "No delayed predictions or recovery receipts; seven bounds undefined",
                "Original six arms, prediction-before-feedback streams, qid denominators, exact hard-exit recovery and all seven original learning bounds",
            ),
            (
                "CES",
                "terminal_blocked_secondary",
                "No TRAIN token reference or TEST scores; three ranking bounds unavailable",
                "Admitted original nonempty token statistics; frozen correct-TRAIN CDF; all three qid-bootstrap ranking contrasts; never rescue failed primary energy bounds",
            ),
            (
                "cost",
                "terminal_blocked_unmeasured",
                "No complete scientific service-cost primary",
                "Nonoverlapping measured full-service clocks including rejected updates, guards, storage and recovery; measured power before joule or hardware claims",
            ),
            (
                "ARC",
                "method_only_disqualified",
                "124 offline compatibility controls; current and original affected checks fail; no live path observed",
                "Pass all original ordering/mechanic regressions and demonstrate a changed reachable live selector before generalization; fixture actions grant no solve credit",
            ),
            (
                "independent_human_confirmation",
                "deferred_no_qualified_signal",
                "Original panel exposed; both verifier gaps remain open",
                "Useful qualified matched-control development signal first, then a later untouched independently labeled human cohort with frozen method and evaluation",
            ),
            (
                "hardware",
                "deferred_exclusions_preserved",
                "No physical-board measurement or operation-matched benefit",
                "Changed physical attestation, measured complete-service bottleneck, operation parity and energy before reopening any excluded route",
            ),
        ]
    ]
    science = deepcopy(prior["science_decisions"])
    science["validation"] = dict(
        verdict_class=diagnosis["verdict_class"],
        readiness=diagnosis["causal_fix_ready_score"],
        qualified_checker_correction=False,
        changed_semantics_and_parity=diagnosis["structural_scan_rows"],
        causal_evidence=diagnosis["causal_evidence"],
        affected_receipts=diagnosis["validation_receipts"],
        full_qualification_credit=0,
    )
    for branch in ["static", "learning"]:
        science[branch] = dict(
            status="unmeasured",
            verdict_class="blocked",
            complete_scientific_null=False,
            brier_change=None,
            ranking_change=None,
            censor_reason="missing_qualified_admitted_predictions",
        )
    ces_rows = [
        dict(
            bounds()["two_bound_rows"][0],
            unit_id=f"CES:{c}",
            arm="CES",
            condition=c,
            seed=782190,
            metric="AUROC_improvement",
            criterion="one_sided_lower>0;tail=0.05/3",
            status="unmeasured",
            censor_reason="missing_reference_and_scientific_predictions",
        )
        for c in source["entropy_reference_protocol"]["comparators"]
    ]
    previous_rows = prior["outcome_rows"]
    skipped = sum(r.get("primary_status") == "missing_file" for r in previous_rows)
    require(len(previous_rows) == 14 and skipped == 10, "previous_accounting")
    return dict(
        **bounds(),
        human_label_rows=human_rows,
        science_decisions=science,
        gauge_control_rows=deepcopy(prior["gauge_control_rows"]),
        donor_label_change_rows=donors,
        donor_changed_label_fraction=changed / support,
        hardware_limits=deepcopy(prior["hardware_limits"]),
        cost_summary=cost,
        historical_outcomes=previous_rows,
        continuation_decisions=decisions,
        retirement_recommendations=retirements,
        preserved_v781_retirement=[
            r
            for r in manifest["retired_extras"]
            if r.get("experiment_id") == "exp9168-forward-qualification"
        ],
        retired_qualification=source["retired_qualification"],
        retired_schedule=source["retired_schedule"],
        previous_milestone_accounting=dict(
            executed=14 - skipped, skipped=skipped, scientific_success_inferred=False
        ),
        validation_obligations=dict(
            original=len(obligations["original_node_ids"]),
            v779_additions=len(obligations["v779_addition_ids"]),
            union_count=len(union),
            observed_execution_rows=len(executed),
            current_qualification_credit=0,
            full_obligation_execution_complete=False,
            original_assertion_bindings=obligations["node_bindings"],
        ),
        validation_unit_rows=[
            dict(union[k], observed_execution=executed.get(k), current_qualification_credit=0)
            for k in sorted(union)
        ],
        collection_partition=dict(
            qid_counts=[len(r["qids"]) for r in rosters],
            answer_count=len(units),
            union_equals_original=True,
            scientific_roles=source["source_summary"]["roles"],
            execution_only=True,
            attempted_scientific_answers=0,
            execution_deficit_answers=len(units),
            execution_deficit_qids=len({r["qid"] for r in units}),
        ),
        arc_summary=dict(
            rows=arc_rows,
            completed=sum(r["status"] == "completed" for r in arc_rows),
            equality_numerator=sum(r["numerator"] for r in arc_rows),
            denominator=len(arc_rows),
            verdict_class=arc["verdict_class"],
            required_checks_passed=arc["required_checks_passed"],
            compatibility_matrix=arc["compatibility_matrix"],
            consumer_reachability=arc["consumer_reachability"],
            regression_result=arc["regression_result"],
            live_path_observed=arc["consumer_reachability"]["live_path_observed"],
            scope="Offline deterministic method controls; no independent solve evidence",
        ),
        ces_summary=dict(
            complete_score=0,
            secondary_signal_score=0,
            primary_gate_rescue=False,
            reference_status="missing_file",
            ranking_bound_rows=ces_rows,
            protocol=source["entropy_reference_protocol"],
            effective_independent_groups=0,
        ),
        independent_human_confirmation=dict(
            warranted_now=False,
            independent_generalization_score=0,
            reopen_condition=decisions[-2]["next_reopen_condition"],
        ),
        historical_code_resolutions=work["historical_code_resolutions"],
    )
