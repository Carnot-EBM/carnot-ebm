"""REQ-REPORT-9179-V781: recompute accounting from retained independent units."""

import ast
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
    """Incomplete predictions remain unmeasured even when every scheduling slot is known."""
    from carnot.reporting import v781_capstone_evidence as e

    require(e.canonical_hash(work["tasks"]) == e.TASK_PIN, "task_authority")
    prior = work["values"]["prior:9165"]["value"]
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
    cost_status = Counter(r["status"] for r in cost["rows"])
    cost_counts = dict(
        intended=len(cost["rows"]),
        attempted=len(cost["rows"]) - cost_status["unstarted"],
        **{k: cost_status[k] for k in ["completed", "failed", "censored", "unstarted"]},
    )
    require(cost_counts == cost["scientific_request_counts"], "cost_aggregate")
    cost["scientific_request_counts"] = cost_counts
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
    for ref in work["code"]:
        if Path(ref["original_path"]).name == "test_v781_capstone_9179.py":
            for node in ast.parse(Path(ref["path"]).read_text()).body:
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    identity = "tests/python/test_v781_capstone_9179.py::" + node.name
                    union[identity] = dict(
                        node_id=identity,
                        sha256=e.canonical_hash(ast.dump(node, include_attributes=False)),
                        status="pending",
                        current_qualification_credit=0,
                        reasons=["current_capstone_assertion"],
                    )
    arc_rows = arc["rows"]
    require(len(arc_rows) == len({r["unit_id"] for r in arc_rows}) == 124, "arc_units")
    equal = sum(
        r["after"] == (r["reference"] if r["recognized"] else r["before"]) for r in arc_rows
    )
    require(equal == sum(r["numerator"] for r in arc_rows), "arc_row_aggregate")
    arc_counts = Counter(r["status"] for r in arc_rows)
    before = {r["task_id"]: r for r in prior["task_outcome_rows"]}
    after = {r["task_id"]: r for r in e.outcomes(work)}
    manifest_ref = work["snapshots"]["ops/exclusion_manifest.yaml"]
    manifest = yaml.safe_load(Path(manifest_ref["path"]).read_bytes())
    retirements = []
    for task in work["tasks"]:
        for old in task.get("prior_failures", []):
            previous = before.get(old["experiment_id"], {})
            current = after.get(task["id"], {})
            key = (
                "scheduler_honest_verdict"
                if old["verdict"].startswith("blocked_")
                else "accounting_honest_verdict"
            )
            actual, verdict = previous.get(key), current.get(key)
            same = bool(old["retire_if_same_verdict"] and actual == old["verdict"] == verdict)
            retirements.append(
                dict(
                    task_id=task["id"],
                    prior_task_id=old["experiment_id"],
                    declared_prior_verdict=old["verdict"],
                    actual_prior_verdict=actual,
                    current_verdict=verdict,
                    condition_met=same,
                    prior_verdict_authenticated=actual == old["verdict"],
                    retire_if_same_verdict=old["retire_if_same_verdict"],
                    recommendation="retire_unchanged_scope"
                    if same
                    else "require_changed_prerequisite",
                    manifest_mutated=False,
                    manifest_retirement_applied=same
                    and any(
                        r.get("experiment_id") == task["id"]
                        and r.get("prior_task_id") == old["experiment_id"]
                        and r.get("exact_verdict") == verdict
                        for r in manifest["retired_extras"]
                    ),
                    workflow="existing exclusion manifest; exact scheduler scope only; producer reads retained manifest bytes",
                    next_reopen_condition=old["addressed_by"],
                )
            )
    decisions = [
        dict(branch=b, disposition=d, evidence=proof, next_reopen_condition=condition)
        for b, d, proof, condition in [
            (
                "forward_validation",
                "retire_unchanged_failed_path",
                "Exp9167 localized repeated structural traversal but corrective execution and tamper controls did not qualify; all obligations retained",
                "Demonstrate a new cause or changed external prerequisite; complete exact obligation union, original assertions and100 percent owned coverage; no renamed unchanged validation attempt",
            ),
            (
                "collection",
                "terminal_blocked",
                "No qualified forward/runtime or any of three scientific shard primaries",
                "Qualified forwards and measured budgets for all three complete100-qid shards; retain900 answers and exact execution deficit if over budget",
            ),
            (
                "static_energy",
                "terminal_blocked_unmeasured",
                "No admitted scientific predictions; two original bounds undefined",
                "Complete admitted cohort and both unchanged static bounds; complete scientific null retires unchanged method",
            ),
            (
                "online_energy",
                "terminal_blocked_unmeasured",
                "No causal learning predictions; all seven original bounds undefined",
                "Complete admitted cohort, original delayed updates and recovery with all seven bounds; complete scientific null retires unchanged method",
            ),
            (
                "independent_human_confirmation",
                "future_work",
                "Exposed development panel; both verifier gaps remain open",
                "Useful controlled development signal first, then untouched independently labeled human confirmation",
            ),
            (
                "hardware",
                "future_work",
                "Scientific service costs unmeasured; preserve all physical-board exclusions",
                "Operation-matched measured complete-service bottleneck and parity; no board retry without changed attestation",
            ),
            (
                "ARC",
                "method_only_disqualified",
                "124 deterministic fixture rows; affected regressions failed; live path not observed",
                "Repair affected regressions and demonstrate a reachable changed live method before live generalization; no solve credit from fixtures",
            ),
        ]
    ]
    science = deepcopy(prior["science_decisions"])
    science["human_labels"].update(
        group_counts=dict(Counter(r["slice"] for r in human_rows)),
        observed_pairs=sum(r["a2_observed_count"] for r in human_rows),
        missing_pairs=sum(r["a2_missing_count"] for r in human_rows),
        disagreements=sum(r["disagreement_count"] for r in human_rows),
        question_count=len(human_rows),
    )
    science["validation"] = dict(
        verdict_class=diagnosis["verdict_class"],
        cause_demonstrated=diagnosis["causal_evidence"]["cause_demonstrated"],
        causal_evidence=diagnosis["causal_evidence"],
        readiness=diagnosis["causal_fix_ready_score"],
        rows=diagnosis["rows"],
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
    science["cost"]["current_status"] = "blocked_missing_logit_runtime_and_complete_cost_primary"
    return dict(
        **bounds(),
        human_label_rows=human_rows,
        science_decisions=science,
        gauge_control_rows=deepcopy(prior["gauge_control_rows"]),
        donor_label_change_rows=deepcopy(prior["donor_label_change_rows"]),
        donor_changed_label_fraction=prior["donor_changed_label_fraction"],
        hardware_limits=deepcopy(prior["hardware_limits"]),
        cost_summary=cost,
        historical_outcomes=deepcopy(prior["task_outcome_rows"]),
        continuation_decisions=decisions,
        retirement_recommendations=retirements,
        validation_obligations=dict(
            original=len(obligations["original_node_ids"]),
            v779_additions=len(obligations["v779_addition_ids"]),
            union_count=len(union),
            current_qualification_credit=0,
            historical_completed=32,
            historical_censored=642,
            original_assertion_bindings=obligations["node_bindings"],
            affected_assertions=obligations["affected_node_assertions"],
            affected_v780_assertions=obligations["affected_v780_node_assertions"],
        ),
        validation_unit_rows=[union[k] for k in sorted(union)],
        measured_validation_costs=deepcopy(prior["measured_validation_costs"]),
        validation_budget_equation=deepcopy(prior["validation_budget_equation"]),
        retired_schedule=deepcopy(prior["retired_schedule"]),
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
            completed=arc_counts["completed"],
            recognized=sum(r["recognized"] for r in arc_rows),
            legacy_controls=sum(not r["recognized"] for r in arc_rows),
            equality_numerator=equal,
            denominator=len(arc_rows),
            verdict_class=arc["verdict_class"],
            required_checks_passed=arc["required_checks_passed"],
            live_path_observed=arc["consumer_reachability"]["live_path_observed"],
            scope="offline deterministic method controls; no game or independent solve evidence",
        ),
    )
