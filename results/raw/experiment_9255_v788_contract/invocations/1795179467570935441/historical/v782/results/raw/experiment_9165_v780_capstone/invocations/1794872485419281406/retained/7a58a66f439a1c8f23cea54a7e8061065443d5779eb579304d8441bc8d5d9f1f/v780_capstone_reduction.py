"""REQ-REPORT-9165-V780: recompute units while retaining all scientific limits."""

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import human_contrast_9141 as human
from carnot.reporting.v763_capstone_reduction import require
from carnot.reporting.v778_capstone_evidence import bounds

Json = dict[str, Any]


def reduce(work: Json) -> Json:
    """Only retained primitive rows can support accounting; no new science is inferred."""
    from carnot.reporting import v780_capstone_evidence as e

    require(e.canonical_hash(work["tasks"]) == e.TASK_PIN, "task_authority")
    prior = work["values"]["prior:9151"]["value"]
    source = work["values"][e.TASKS[0]]["value"]
    schedule = work["values"][e.TASKS[1]]["value"]
    labels = [
        dict(s, qid=r["qid"], role=r["role"], group_id=r["group_id"])
        for r in prior["human_label_rows"]
        for s in r["siblings"]
    ]
    rows = human.group_rows(labels)
    require(rows == prior["human_label_rows"], "human_rows")
    counts = dict(Counter(r["slice"] for r in rows))
    observed = sum(r["a2_observed_count"] for r in rows)
    missing = sum(r["a2_missing_count"] for r in rows)
    disagreements = sum(r["disagreement_count"] for r in rows)
    science = deepcopy(prior["science_decisions"])
    science["human_labels"].update(
        group_counts=counts,
        observed_pairs=observed,
        missing_pairs=missing,
        disagreements=disagreements,
        question_count=len(rows),
    )
    validation_rows = schedule["rows"]
    status = Counter(r["status"] for r in validation_rows)
    budget = dict(
        intended=len(validation_rows),
        attempted=status["completed"] + status["failed"],
        completed=status["completed"],
        failed=status["failed"],
        censored=status["censored"],
        effective_independent_groups=0,
    )
    require(budget == schedule["sample_size_budget"], "validation_budget")
    science["validation"] = dict(
        verdict_class=schedule["verdict_class"],
        readiness=schedule["schedule_ready_score"],
        sample_size_budget=budget,
        rows=validation_rows,
        computational_root_cause=schedule["computational_root_cause"],
        blocked_prerequisite=schedule["blocked_prerequisite"],
    )
    rosters = []
    for shard in "abc":
        ref = source[f"shard_{shard}_roster_reference"]
        rosters.append(json.loads(Path(work["operands"][ref["path"]]["path"]).read_bytes()))
    units = [r for roster in rosters for r in roster["rows"]]
    original = json.loads(
        Path(work["operands"][source["inference_view_reference"]["path"]]["path"]).read_bytes()
    )
    original_rows = original["rows"]
    require(
        len(units) == len({r["unit_id"] for r in units}) == 900
        and sorted(units, key=lambda r: r["unit_id"])
        == sorted(original_rows, key=lambda r: r["unit_id"]),
        "partition_union",
    )
    require([len(r["qids"]) for r in rosters] == [100] * 3, "partition_size")
    previous = {r["task_id"]: r for r in prior["task_outcome_rows"]}
    current = {r["task_id"]: r for r in e.outcomes(work)}
    current[e.TASKS[-1]] = dict(accounting_honest_verdict="complete_blocked_declared_gate")
    retirements = []
    for task in work["tasks"]:
        for old in task.get("prior_failures", []):
            before_row, after_row = previous.get(old["experiment_id"], {}), current[task["id"]]
            key = (
                "scheduler_honest_verdict"
                if old["verdict"].startswith("blocked_")
                else "accounting_honest_verdict"
            )
            before, after = before_row.get(key), after_row.get(key)
            same = bool(old["retire_if_same_verdict"] and before == old["verdict"] == after)
            retirements.append(
                dict(
                    task_id=task["id"],
                    prior_task_id=old["experiment_id"],
                    declared_prior_verdict=old["verdict"],
                    actual_prior_verdict=before,
                    current_verdict=after,
                    prior_verdict_authenticated=before == old["verdict"],
                    retire_if_same_verdict=old["retire_if_same_verdict"],
                    condition_met=same,
                    recommendation="retire_unchanged_scope"
                    if same
                    else "require_new_qualified_prerequisite",
                    changed_mechanism_context=old["addressed_by"],
                    manifest_mutated=False,
                    workflow="existing exclusion manifest; conductor reconciliation owns mutation",
                )
            )
    decisions = deepcopy(prior["continuation_decisions"])
    for decision in decisions:
        if decision["branch"] == "forward_validation":
            decision.update(
                decision="retire_exact_schedule_scope",
                next_prerequisite="Retire only the old sum-of-ceilings forecast. Measured execution remains blocked: identify a demonstrated new slow-child cause and preserve remaining642 of674 obligations, original assertions, full population and100 percent owned coverage. No blind rerun.",
            )
        if decision["branch"] == "collection":
            decision.update(
                next_prerequisite="Qualified complete forwards and runtime first; preserve all900 answers/300 qids across exactly three disjoint100-qid execution shards, retaining original siblings and scientific roles."
            )
        if decision["branch"] == "independent_human_confirmation":
            decision.update(
                next_prerequisite="Controlled development signal first, then new untouched questions and independent matched-control human confirmation. This exposed panel cannot close either verifier gap."
            )
    science["cost"]["current_status"] = "blocked_missing_logit_runtime_and_complete_cost_primary"
    return dict(
        **bounds(),
        human_label_rows=rows,
        science_decisions=science,
        gauge_control_rows=deepcopy(prior["gauge_control_rows"]),
        donor_label_change_rows=deepcopy(prior["donor_label_change_rows"]),
        donor_changed_label_fraction=prior["donor_changed_label_fraction"],
        hardware_limits=deepcopy(prior["hardware_limits"]),
        historical_outcomes=deepcopy(prior["task_outcome_rows"]),
        cost_summary=deepcopy(prior["cost_summary"]),
        validation_obligations=budget,
        validation_unit_rows=validation_rows,
        measured_validation_costs=deepcopy(schedule["cost_model_rows"]),
        validation_budget_equation=deepcopy(schedule["schedule_reference"]["budget_equation"]),
        retired_schedule=deepcopy(source["retired_schedule"]),
        collection_partition=dict(
            qid_counts=[len(r["qids"]) for r in rosters],
            answer_count=len(units),
            union_equals_original=True,
            scientific_roles=source["source_summary"]["roles"],
            execution_only=True,
        ),
        continuation_decisions=decisions,
        retirement_recommendations=retirements,
    )
