"""REQ-REPORT-9207-V783: recompute original units and keep new branch eligibility separate."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v782_capstone_evidence as prior
from carnot.reporting.v763_capstone_reduction import require

Json = dict[str, Any]


def reduce(work: Json) -> Json:
    """Use authenticated V782 primitives without crediting historical work as current science."""
    from carnot.reporting import v783_capstone_evidence as e

    require(e.canonical_hash(work["tasks"]) == e.TASK_PIN, "task_authority")
    old = work["values"]["prior:9193"]["value"]
    saved = work["operands"][old["work_reference"]["path"]]
    historical = json.loads(Path(saved["path"]).read_bytes())
    for key in ["snapshots", "operands"]:
        historical[key] = {k: work["operands"][r["path"]] for k, r in historical[key].items()}
    result = prior.reduce(historical)
    source = work["values"][e.TASKS[0]]["value"]
    rows = e.outcomes(work)
    require(
        len(rows) == 13 and all(r["accounting_evidence_reference"] for r in rows),
        "outcome_accounting",
    )
    require(sum(r["primary_status"] == "present" for r in rows) == 1, "frozen_execution_scope")
    previous = old["outcome_rows"]
    require(
        len(previous) == 14 and sum(r["primary_status"] == "missing_file" for r in previous) == 10,
        "previous_accounting",
    )
    before = {r["task_id"]: r for r in deepcopy([*old["historical_outcomes"], *previous])}
    for row in before.values():
        if (
            row.get("scheduler_events")
            and row.get("primary_status") == "missing_file"
            and not row.get("scheduler_alternative")
        ):
            row["scheduler_honest_verdict"] = "blocked_upstream_retired"
    after = {r["task_id"]: r for r in rows}
    retirements = []
    for task in work["tasks"]:
        for failure in task.get("prior_failures", []):
            require(
                set(failure)
                >= {"experiment_id", "verdict", "addressed_by", "retire_if_same_verdict"},
                "prior_failure_fields",
            )
            earlier, current = before.get(failure["experiment_id"], {}), after.get(task["id"], {})
            actual = earlier.get("scheduler_honest_verdict") or earlier.get("honest_verdict")
            verdict = current.get("scheduler_honest_verdict") or current.get("honest_verdict")
            same_scope = (
                task["id"].split("-", 1)[1] == failure["experiment_id"].split("-", 1)[1]
                and task["id"] != e.TASKS[-1]
            )
            same = (
                failure["retire_if_same_verdict"] is True
                and actual == failure["verdict"] == verdict
                and same_scope
            )
            retirements.append(
                dict(
                    task_id=task["id"],
                    prior_task_id=failure["experiment_id"],
                    prior_failure=deepcopy(failure),
                    declared_prior_verdict=failure["verdict"],
                    actual_prior_verdict=actual,
                    current_verdict=verdict,
                    prior_verdict_authenticated=actual == failure["verdict"],
                    scope_matches=same_scope,
                    condition_met=same,
                    recommendation="retire_unchanged_execution_scope"
                    if same
                    else "no_exact_authenticated_retirement",
                    next_reopen_condition=failure["addressed_by"],
                    hypothesis_retired=False,
                    manifest_mutated=False,
                    prior_reference=earlier.get(
                        "accounting_evidence_reference", earlier.get("reference")
                    )
                    or historical["snapshots"]["ops/conductor-log.md"],
                    current_reference=current.get("accounting_evidence_reference"),
                )
            )
    decisions = deepcopy(result["continuation_decisions"])
    decisions[0].update(
        disposition="terminal_blocked_unmeasured_changed_correction",
        evidence="Exp9195 scheduler refusal: contract_ready_score observed zero; no lossless prefilter measured",
        next_reopen_condition="Qualified contract, then every original recursive JSON path with lossless byte parity and unchanged decoder/assertions; corpus_scan_ready_score=1 after all affected checks",
    )
    decisions[1].update(
        evidence="Exp9196 log-only upstream-retired skip; original674 and all affected obligations never completed",
        next_reopen_condition="Separately qualified lossless corpus-input correction, then original621 plus53 and every affected obligation once with original fixtures/assertions and current receipts; never depend on Exp9181/9182",
    )
    decisions.extend(
        [
            dict(
                branch="label_ambiguity_method",
                disposition="terminal_blocked_independent_gate",
                evidence="Exp9197 source_ready_score observed zero; no method primary",
                next_reopen_condition="Authenticate original900 human responses and source_ready_score=1; independently prove exact affine endpoints, repeated-retention coefficients and missing-A2 controls without model readiness",
            ),
            dict(
                branch="label_ambiguity_sensitivity",
                disposition="terminal_blocked_unmeasured",
                evidence="No method authority or genuine frozen predictions for either sensitivity branch",
                next_reopen_condition="Qualified independent Exp9197 method plus independently eligible static or learning predictions on original supports; preserve A1 primary and separate observed disagreement from missing-A2 envelope",
            ),
        ]
    )
    result.update(
        historical_outcomes=deepcopy(previous),
        previous_milestone_accounting=dict(
            executed=4, skipped=10, scientific_success_inferred=False
        ),
        current_milestone_accounting=dict(
            executed=2,
            skipped=12,
            scheduler_artifact_skips=2,
            scheduler_log_skips=10,
            scientific_success_inferred=False,
        ),
        retirement_recommendations=retirements,
        continuation_decisions=decisions,
        preserved_v782_retirement=deepcopy(old["retirement_recommendations"]),
        corpus_scan_summary=dict(
            status="blocked",
            corpus_scan_ready_score="missing_field",
            qualified_checker_correction=False,
            lossless_input_parity_measured=False,
            checker_semantics_changed=False,
            full_corpus_traversal_complete=False,
            evidence=after[e.TASKS[1]],
        ),
        forward_qualification_summary=dict(
            status="blocked",
            forward_validation_ready_score="missing_file",
            complete_obligation_execution=False,
            current_qualification_credit=0,
            evidence=after[e.TASKS[2]],
        ),
        label_ambiguity_summary=dict(
            method_status="blocked",
            method_evidence=after[e.TASKS[3]],
            method_requires_model_readiness=False,
            protocol=source["label_ambiguity_protocol"],
            primary_gate_rescue=False,
            branches={
                b: dict(
                    status="blocked",
                    absolute_metric=None,
                    numerator=None,
                    denominator=0,
                    censor_reason="missing_method_and_frozen_predictions",
                )
                for b in ["observed_disagreement", "missing_A2_envelope"]
            },
        ),
        original_method_protocols=deepcopy(source["protocol_references"]),
    )
    result["science_decisions"]["validation"].update(
        qualified_checker_correction=False,
        current_corpus_scan_status="blocked",
        current_full_qualification_status="blocked",
    )
    result["validation_obligations"]["v783_affected_obligations"] = source["validation_obligations"]
    result["arc_summary"].update(
        current_ARC_experiment_scheduled=False, historical_experiment_id=9192
    )
    return dict(result)
