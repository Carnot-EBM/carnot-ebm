"""REQ-REPORT-9221-V784: recompute original units and keep new branch eligibility separate."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v782_capstone_evidence as prior
from carnot.reporting.v763_capstone_reduction import require

Json = dict[str, Any]


def reduce(work: Json) -> Json:
    """Use authenticated V782/V783 primitives and Exp9211 label evidence without inventing model science."""
    from carnot.reporting import v784_capstone_evidence as e

    require(e.canonical_hash(work["tasks"]) == e.TASK_PIN, "task_authority")
    old_v782 = work["values"]["prior:9193"]["value"]
    old_v783 = work["values"]["prior:9207"]["value"]
    saved_v782 = work["operands"][old_v782["work_reference"]["path"]]
    historical = json.loads(Path(saved_v782["path"]).read_bytes())
    for key in ["snapshots", "operands"]:
        historical[key] = {k: work["operands"][r["path"]] for k, r in historical[key].items()}
    result = prior.reduce(historical)
    saved_v783 = work["operands"][old_v783["work_reference"]["path"]]
    historical_v783 = json.loads(Path(saved_v783["path"]).read_bytes())
    source = work["values"][e.TASKS[0]]["value"]
    ambiguity = work["values"][e.TASKS[3]]["value"]
    rows = e.outcomes(work)
    require(
        len(rows) == 13 and all(r["accounting_evidence_reference"] for r in rows),
        "outcome_accounting",
    )
    require(sum(r["primary_status"] == "present" for r in rows) == 2, "frozen_execution_scope")
    require(sum(r["primary_status"] == "missing_file" for r in rows) == 11, "frozen_skip_scope")
    previous = old_v783["outcome_rows"]
    require(
        len(previous) == 14 and sum(r["primary_status"] == "missing_file" for r in previous) == 12,
        "previous_accounting",
    )
    require(
        old_v783["previous_milestone_accounting"]
        == dict(executed=4, skipped=10, scientific_success_inferred=False),
        "preserved_v782_accounting",
    )
    require(
        source["replay_authority_ready_score"] == 0
        and source["contract_ready_score"] == 0
        and source["preserved_v783"]["readiness_scores"] == [0] * 4,
        "replay_contract_state",
    )
    masks = json.loads(
        Path(work["operands"][ambiguity["ambiguity_masks_reference"]["path"]]["path"]).read_bytes()
    )
    ambiguity_rows = ambiguity["rows"]
    require(
        len(ambiguity_rows) == len(masks["rows"]) == 900
        and len(ambiguity["target_identifiability_rows"]) == 2700,
        "ambiguity_units",
    )
    require(
        sum(r["factual_error_status"] == "present" for r in ambiguity_rows) == 900
        and sum(
            r["expressed_uncertainty_status"] == "absent"
            and r["reelicitation_status"] == "absent"
            and not r["identifiable_si_cf"]
            for r in ambiguity_rows
        )
        == 900,
        "target_identifiability_counts",
    )
    require(
        sum(len(r["observed_disagreement_admissible"]) == 2 for r in masks["rows"]) == 29
        and sum(len(r["missing_annotation_envelope_admissible"]) == 2 for r in masks["rows"]) == 629
        and ambiguity["target_identifiability_ready_score"] == 1
        and ambiguity["label_ambiguity_ready_score"] == 1
        and ambiguity["required_checks_passed"] is True,
        "ambiguity_method_qualified",
    )
    before = {
        r["task_id"]: r
        for r in deepcopy([*old_v782["historical_outcomes"], *old_v782["outcome_rows"], *previous])
    }
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
                    or work["operands"][
                        historical_v783["snapshots"]["ops/conductor-log.md"]["path"]
                    ],
                    current_reference=current.get("accounting_evidence_reference"),
                )
            )
    decisions = deepcopy(result["continuation_decisions"])
    decisions[0].update(
        disposition="terminal_blocked_unmeasured_changed_correction",
        evidence="Exp9208 direct_consumers timed out (-9); Exp9209 scheduler refusal recorded contract_ready_score=0 and replay_authority_ready_score=0; no lossless prefilter measured",
        next_reopen_condition="Qualify Exp9208 protocol-owner replay and isolated historical inputs within deadline, then stream every recursive JSON path with lossless byte parity and corpus_scan_ready_score=1",
    )
    decisions[1].update(
        evidence="Exp9210 log-only upstream-retired skip; original674 and all affected obligations never completed",
        next_reopen_condition="Separately qualified lossless corpus-input correction, then original621 plus53 and every affected V780-V784 obligation once with original assertions and current receipts; never depend on Exp9181/9182",
    )
    decisions.extend(
        [
            dict(
                branch="label_ambiguity_method",
                disposition="qualified_independent_method",
                evidence="Exp9211 independently authenticated Exp9097 custody and Exp9126/9132 roles, proved factual_error present on 900 responses with zero identifiable SI/CF rows, and qualified exact affine Brier endpoints",
                next_reopen_condition="Retain immutable Exp9211 masks, rubric and reducer; apply only after genuine frozen static or online predictions exist on original supports",
            ),
            dict(
                branch="label_ambiguity_sensitivity",
                disposition="terminal_blocked_unmeasured",
                evidence="Exp9211 method is qualified, but model tasks skipped and no genuine frozen predictions exist for either sensitivity branch",
                next_reopen_condition="Qualified Exp9211 method plus independently eligible static or learning predictions on original supports; preserve A1 primary and separate observed disagreement from missing-A2 envelope",
            ),
        ]
    )
    result.update(
        historical_outcomes=deepcopy(previous),
        preserved_v782_accounting=deepcopy(old_v783["previous_milestone_accounting"]),
        previous_milestone_accounting=deepcopy(old_v783["current_milestone_accounting"]),
        current_milestone_accounting=dict(
            executed=3,
            skipped=11,
            scheduler_artifact_skips=1,
            scheduler_log_skips=10,
            scientific_success_inferred=False,
        ),
        retirement_recommendations=retirements,
        continuation_decisions=decisions,
        preserved_v782_retirement=deepcopy(old_v782["retirement_recommendations"]),
        preserved_v783_retirement=deepcopy(old_v783["retirement_recommendations"]),
        replay_authority_summary=dict(
            status="disqualified",
            replay_authority_ready_score=source["replay_authority_ready_score"],
            contract_ready_score=source["contract_ready_score"],
            preserved_v783_readiness_scores=source["preserved_v783"]["readiness_scores"],
            protocol_owner_rows=deepcopy(source["protocol_owner_rows"]),
            evidence=after[e.TASKS[0]],
        ),
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
            method_status="qualified",
            label_ambiguity_ready_score=ambiguity["label_ambiguity_ready_score"],
            target_identifiability_ready_score=ambiguity["target_identifiability_ready_score"],
            target_identifiability_summary=deepcopy(ambiguity["target_identifiability_summary"]),
            ambiguity_masks_reference=deepcopy(ambiguity["ambiguity_masks_reference"]),
            target_rubric_reference=deepcopy(ambiguity["target_rubric_reference"]),
            reducer_reference=deepcopy(ambiguity["reducer_reference"]),
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
                    censor_reason="missing_frozen_scientific_predictions",
                )
                for b in ["observed_disagreement", "missing_A2_envelope"]
            },
        ),
        original_method_protocols=deepcopy(source["protocol_references"]),
    )
    result["science_decisions"]["validation"].update(
        qualified_checker_correction=False,
        current_replay_authority_status="disqualified",
        current_corpus_scan_status="blocked",
        current_full_qualification_status="blocked",
    )
    result["validation_obligations"]["v784_affected_obligations"] = source["validation_obligations"]
    result["arc_summary"].update(
        current_ARC_experiment_scheduled=False, historical_experiment_id=9192
    )
    return dict(result)
