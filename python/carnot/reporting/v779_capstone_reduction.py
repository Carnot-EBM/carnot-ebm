"""REQ-REPORT-9151-V779: recompute descriptive support without granting missing science credit."""

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
    """Keep original question units, unfinished validation and all service-cost censor reasons."""
    from carnot.reporting import v779_capstone_evidence as e

    require(e.canonical_hash(work["tasks"]) == e.TASK_PIN, "task_authority")
    source = work["values"][e.TASKS[0]]["value"]
    prior = work["values"]["prior:9137"]["value"]
    audit = work["values"][e.TASKS[3]]["value"]
    cost = work["values"][e.TASKS[12]]["value"]
    ref = source["evaluator_label_reference"]
    labels = json.loads(Path(work["operands"][ref["path"]]["path"]).read_bytes())["rows"]
    rows = human.group_rows(labels)
    require(rows == audit["rows"], "human_rows")
    counts = dict(Counter(r["slice"] for r in rows))
    require(counts == audit["group_counts"], "human_groups")
    observed = sum(r["a2_observed_count"] for r in rows)
    disagreements = sum(r["disagreement_count"] for r in rows)
    missing = sum(r["a2_missing_count"] for r in rows)
    scientific = [r for r in cost["rows"] if r["scope"] == "science"]
    require(not any(r["status"] == "completed" for r in scientific), "unmeasured_cost")
    history = {r["task_id"]: r for r in prior["task_outcome_rows"]}
    retirement = []
    current = {r["task_id"]: r for r in e.outcomes(work)}
    current[e.TASKS[-1]] = dict(accounting_honest_verdict="complete_blocked_declared_gate")
    for task in work["tasks"]:
        for old in task.get("prior_failures", []):
            previous, now = history.get(old["experiment_id"], {}), current[task["id"]]
            before = (
                previous.get("scheduler_honest_verdict")
                if old["verdict"].startswith("blocked_")
                else previous.get("accounting_honest_verdict") or previous.get("honest_verdict")
            )
            after = (
                now.get("scheduler_honest_verdict")
                if old["verdict"].startswith("blocked_")
                else now.get("accounting_honest_verdict")
            )
            same = bool(old["retire_if_same_verdict"] and before == old["verdict"] == after)
            retirement.append(
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
    unfinished = work["values"][e.TASKS[2]]["value"]
    decisions = [
        (
            "forward_validation",
            "retire_exact_schedule_scope",
            "Second scheduling failure: Exp9125 then Exp9140. A different executable schedule must preserve all original assertions, full population, semantic controls and 100 percent owned coverage; no same-scope retry.",
        ),
        (
            "collection",
            "blocked",
            "After qualified forwards, split infeasible collection into smaller execution units with unchanged total900 answers/300 qids and original roles; no same-size retry.",
        ),
        (
            "static_energy",
            "unmeasured",
            "Qualified complete shards and both original lower>0 Brier bounds, with accepted-risk scope and gauge controls.",
        ),
        (
            "online_energy",
            "unmeasured",
            "Qualified causal predictions, actual changed donor labels, all seven original bounds and exact durable recovery.",
        ),
        (
            "independent_human_confirmation",
            "blocked",
            "A controlled development signal must precede genuinely independent matched-control human confirmation; this exposed panel cannot close either verifier gap.",
        ),
        (
            "gauge",
            "limited",
            "No gauge-sensitive score establishes a general theorem; normalized probability invariance supplies no independent verifier benefit.",
        ),
        (
            "hardware",
            "blocked",
            "Operation-matched parity and measured complete-service bottleneck are required; no new hardware speed, power or access claim.",
        ),
        (
            "ARC",
            "historical_only",
            "No new ARC solve or re-solving credit follows from this milestone.",
        ),
        (
            "publication",
            "operator_only",
            "Historical FoVer G1-G4 do not establish current science readiness or authorize external publication.",
        ),
    ]
    return dict(
        **bounds(),
        human_label_rows=rows,
        gauge_control_rows=deepcopy(prior["gauge_control_rows"]),
        donor_label_change_rows=deepcopy(prior["donor_label_change_rows"]),
        donor_changed_label_fraction=prior["donor_changed_label_fraction"],
        hardware_limits=deepcopy(cost["hardware_scope"]),
        historical_outcomes=deepcopy(prior["task_outcome_rows"]),
        cost_summary=dict(
            rows=scientific,
            scientific_request_counts=dict(
                intended=len(scientific),
                attempted=sum(r["status"] not in {"unstarted", "censored"} for r in scientific),
                completed=sum(r["status"] == "completed" for r in scientific),
                failed=sum(r["status"] == "failed" for r in scientific),
                censored=sum(r["status"] == "censored" for r in scientific),
                unstarted=sum(r["status"] == "unstarted" for r in scientific),
            ),
            measured_latency_s=None,
            energy_joules=None,
            amdahl_ceiling=None,
            complete_adaptation_rows=cost["complete_adaptation_rows"],
            hardware_operation_rows=cost["hardware_operation_rows"],
        ),
        science_decisions=dict(
            static=dict(status="unmeasured", bound_count=2, effective_independent_groups=0),
            online=dict(status="unmeasured", bound_count=7, effective_independent_groups=0),
            human_labels=dict(
                group_counts=counts,
                observed_pairs=observed,
                disagreements=disagreements,
                missing_pairs=missing,
                question_count=len(rows),
                agreement_intervals=audit["agreement_intervals"],
                panel_resolution=audit["panel_resolution"],
                truth_ceiling=False,
            ),
            contrast=deepcopy(audit["diagnostic_outcomes"]),
            cost=dict(
                scientific_request_counts=cost["scientific_request_counts"],
                measured_latency_s=None,
                energy_joules=None,
                amdahl_ceiling=None,
            ),
            prototype_correctness="qualified original static/online mechanics and human descriptive protocol",
            complete_null="administrative and fixture controls only; no measured scientific null",
            disqualification="Exp9125 remains disqualified; all original judge and forward quarantines preserved",
            validation=dict(
                verdict_class=unfinished["verdict_class"],
                readiness=unfinished["forward_validation_ready_score"],
                sample_size_budget=unfinished["sample_size_budget"],
                rows=unfinished["rows"],
            ),
        ),
        continuation_decisions=[
            dict(branch=b, decision=d, next_prerequisite=p) for b, d, p in decisions
        ],
        retirement_recommendations=retirement,
    )
