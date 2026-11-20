"""REQ-REPORT-9240-V786: recompute original units and preserve V785 and V786 branch eligibility."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v782_capstone_evidence as prior
from carnot.reporting.v763_capstone_reduction import require

Json = dict[str, Any]


def reduce(work: Json) -> Json:
    """Use authenticated V782-V785 primitives and Exp9211/Exp9225 method evidence without inventing model science."""
    from carnot.reporting import v786_capstone_evidence as e

    require(e.canonical_hash(work["tasks"]) == e.TASK_PIN, "task_authority")
    old_v782 = work["values"]["prior:9193"]["value"]
    old_v783 = work["values"]["prior:9207"]["value"]
    old_v784 = work["values"]["prior:9221"]["value"]
    saved_v782 = work["operands"][old_v782["work_reference"]["path"]]
    historical = json.loads(Path(saved_v782["path"]).read_bytes())
    for key in ["snapshots", "operands"]:
        historical[key] = {k: work["operands"][r["path"]] for k, r in historical[key].items()}
    result = prior.reduce(historical)
    source_v785 = work["values"]["prior:9222"]["value"]
    ambiguity = work["values"]["prior:9211"]["value"]
    deferral = work["values"]["prior:9225"]["value"]
    rows = e.outcomes(work)
    v785_rows = e.v785_outcomes(work)
    require(
        len(rows) == 13 and all(r["accounting_evidence_reference"] for r in rows),
        "outcome_accounting",
    )
    require(sum(r["primary_status"] == "present" for r in rows) == 0, "frozen_execution_scope")
    require(sum(r["primary_status"] == "missing_file" for r in rows) == 13, "frozen_skip_scope")
    require(
        sum(r["disposition"] == "authenticated_scheduler_pretest_skip" for r in rows) == 2
        and sum(r["disposition"] == "authenticated_scheduler_log_skip" for r in rows) == 11,
        "v786_scheduler_skip_partition",
    )
    previous_v784 = old_v784["outcome_rows"]
    require(
        len(previous_v784) == 14
        and sum(r["primary_status"] == "missing_file" for r in previous_v784) == 11,
        "previous_v784_accounting",
    )
    require(
        old_v784["preserved_v782_accounting"]
        == dict(executed=4, skipped=10, scientific_success_inferred=False),
        "preserved_v782_accounting",
    )
    require(
        old_v784["previous_milestone_accounting"]
        == dict(
            executed=2,
            skipped=12,
            scheduler_artifact_skips=2,
            scheduler_log_skips=10,
            scientific_success_inferred=False,
        ),
        "preserved_v783_accounting",
    )
    require(
        old_v784["current_milestone_accounting"]
        == dict(
            executed=3,
            skipped=11,
            scheduler_artifact_skips=1,
            scheduler_log_skips=10,
            scientific_success_inferred=False,
        ),
        "preserved_v784_accounting",
    )
    require(
        source_v785["replay_authority_ready_score"] == 1
        and source_v785["contract_ready_score"] == 0
        and source_v785["source_ready_score"] == 1
        and source_v785["static_methods_ready_score"] == 1
        and source_v785["online_methods_ready_score"] == 1
        and source_v785["required_checks_passed"] is True
        and len(source_v785["consumer_shard_rows"]) == 3
        and all(r["passed"] and r["exit_code"] == 0 for r in source_v785["consumer_shard_rows"])
        and len(source_v785["full_task_objects"]) == 5
        and source_v785["censored_count"] == 9
        and source_v785["preserved_v784"]["readiness_scores"] == [0] * 5,
        "v785_replay_contract_state",
    )
    require(
        len(v785_rows) == 14
        and sum(r["primary_status"] == "present" for r in v785_rows) == 2
        and sum(r["primary_status"] == "missing_file" for r in v785_rows) == 3
        and sum(r["primary_status"] == "missing_field" for r in v785_rows) == 9,
        "v785_outcome_accounting",
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
    deferral_masks = json.loads(
        Path(
            work["operands"][deferral["selective_deferral_masks_reference"]["path"]]["path"]
        ).read_bytes()
    )
    require(
        len(deferral["rows"]) == len(deferral_masks["rows"]) == 150
        and deferral["effective_independent_groups"] == 50
        and deferral["selective_deferral_ready_score"] == 1
        and deferral["spline_support_diagnostic_ready_score"] == 1
        and deferral["required_checks_passed"] is True
        and deferral["scientific_effect_measured"] is False
        and len(deferral["estimator_control_rows"]) == 5
        and all(r["passed"] for r in deferral["estimator_control_rows"])
        and all(r["passed"] and not r["labels_read"] for r in deferral["spline_control_rows"]),
        "selective_deferral_method_qualified",
    )
    before = {
        r["task_id"]: r
        for r in deepcopy(
            [
                *old_v782["historical_outcomes"],
                *old_v782["outcome_rows"],
                *old_v783["outcome_rows"],
                *previous_v784,
                *v785_rows[:5],
            ]
        )
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
                    or work["snapshots"]["ops/conductor-log.md"],
                    current_reference=current.get("accounting_evidence_reference"),
                )
            )
    decisions = deepcopy(result["continuation_decisions"])
    decisions[0].update(
        disposition="terminal_blocked_unmeasured_changed_correction",
        evidence="Exp9222 qualified replay_authority_ready_score=1 across all three consumer shards, but blocked contract_ready_score=0 on stale V784 markdown and a 5-task queue; Exp9223 refused at conductor pre-gate; Exp9227/Exp9228 skipped in V786; no lossless prefilter measured",
        next_reopen_condition="Qualify Exp9227 14-task V786 contract parity and split consumer shards within deadline, then stream every recursive JSON path in Exp9228 with lossless byte parity and corpus_scan_ready_score=1",
    )
    decisions[1].update(
        evidence="Exp9224 and Exp9229 log-only upstream-retired skips; original674 and all affected obligations never completed in one normal pass",
        next_reopen_condition="Separately qualified lossless corpus-input correction, then original621 plus53 and every affected V780-V786 obligation once with original assertions and current receipts; never depend on Exp9181/9182",
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
            dict(
                branch="selective_deferral_and_spline_support_method",
                disposition="qualified_independent_method",
                evidence="Exp9225 independently qualified 10-stream selective-deferral feedback masks, Horvitz-Thompson IPW risk controls (alpha>0 unbiased, alpha=0 unidentifiable), and label-free 57-column cubic B-spline support-density strata",
                next_reopen_condition="Retain immutable Exp9225 masks, protocols and classifier; consume only after eligible frozen online or static predictions exist on original supports",
            ),
            dict(
                branch="conformal_risk_and_mycelium_protocol",
                disposition="terminal_blocked_pretest_skip",
                evidence="Exp9230 had three scheduler pre-test skips in ops/conductor-log.md and no canonical primary artifact",
                next_reopen_condition="Resolve pre-test failure, independently authenticate Exp9097/9126/9132/9211/9225 hashes, and qualify qid-cluster-valid conformal risk bounds and Mycelium provenance slices without model work",
            ),
        ]
    )
    v785_accounting = dict(
        executed=2,
        skipped=3,
        scheduler_artifact_skips=1,
        scheduler_log_skips=2,
        omitted_slots=9,
        scientific_success_inferred=False,
    )
    result.update(
        historical_outcomes=deepcopy(v785_rows),
        preserved_v782_accounting=deepcopy(old_v784["preserved_v782_accounting"]),
        preserved_v783_accounting=deepcopy(old_v784["previous_milestone_accounting"]),
        preserved_v784_accounting=deepcopy(old_v784["current_milestone_accounting"]),
        preserved_v785_accounting=deepcopy(v785_accounting),
        previous_milestone_accounting=deepcopy(v785_accounting),
        preserved_v785_outcomes=deepcopy(v785_rows),
        current_milestone_accounting=dict(
            executed=1,
            skipped=13,
            scheduler_artifact_skips=0,
            scheduler_log_skips=13,
            scheduler_pretest_log_skips=2,
            scheduler_upstream_retired_log_skips=11,
            scientific_success_inferred=False,
        ),
        retirement_recommendations=retirements,
        continuation_decisions=decisions,
        preserved_v782_retirement=deepcopy(old_v782["retirement_recommendations"]),
        preserved_v783_retirement=deepcopy(old_v783["retirement_recommendations"]),
        preserved_v784_retirement=deepcopy(old_v784["retirement_recommendations"]),
        replay_authority_summary=dict(
            status="blocked",
            replay_authority_ready_score="missing_file",
            contract_ready_score="missing_file",
            v785_replay_authority_ready_score=source_v785["replay_authority_ready_score"],
            v785_contract_ready_score=source_v785["contract_ready_score"],
            v785_source_ready_score=source_v785["source_ready_score"],
            v785_static_methods_ready_score=source_v785["static_methods_ready_score"],
            v785_online_methods_ready_score=source_v785["online_methods_ready_score"],
            v785_consumer_shard_rows=deepcopy(source_v785["consumer_shard_rows"]),
            v785_protocol_owner_rows=deepcopy(source_v785["protocol_owner_rows"]),
            preserved_v784_readiness_scores=source_v785["preserved_v784"]["readiness_scores"],
            evidence=after[e.TASKS[0]],
        ),
        corpus_scan_summary=dict(
            status="blocked",
            corpus_scan_ready_score="missing_file",
            v785_corpus_scan_ready_score="missing_field",
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
            method_evidence=previous_v784[3],
            method_requires_model_readiness=False,
            protocol=source_v785["label_ambiguity_protocol"],
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
        selective_deferral_summary=dict(
            method_status="qualified",
            selective_deferral_ready_score=deferral["selective_deferral_ready_score"],
            selective_deferral_masks_reference=deepcopy(
                deferral["selective_deferral_masks_reference"]
            ),
            selective_deferral_protocol_reference=deepcopy(
                deferral["selective_deferral_protocol_reference"]
            ),
            classifier_reference=deepcopy(deferral["classifier_reference"]),
            method_evidence=v785_rows[3],
            method_requires_model_readiness=False,
            primary_gate_rescue=False,
            branches={
                b: dict(
                    status="blocked",
                    absolute_metric=None,
                    numerator=None,
                    denominator=0,
                    censor_reason="missing_frozen_scientific_predictions",
                )
                for b in [
                    "full_release",
                    "escalation_only",
                    "audited_deferral_alpha_0.10",
                    "audited_deferral_alpha_0.25",
                ]
            },
        ),
        spline_support_summary=dict(
            method_status="qualified",
            spline_support_diagnostic_ready_score=deferral["spline_support_diagnostic_ready_score"],
            spline_support_protocol_reference=deepcopy(
                deferral["spline_support_protocol_reference"]
            ),
            method_evidence=v785_rows[3],
            method_requires_model_readiness=False,
            primary_gate_rescue=False,
            branches={
                b: dict(
                    status="blocked",
                    absolute_metric=None,
                    numerator=None,
                    denominator=0,
                    censor_reason="missing_frozen_scientific_predictions",
                )
                for b in ["interior_dense", "boundary_sparse_or_clipped"]
            },
        ),
        conformal_risk_summary=dict(
            method_status="blocked",
            conformal_risk_ready_score="missing_file",
            mycelium_provenance_ready_score="missing_file",
            method_evidence=after[e.TASKS[3]],
            method_requires_model_readiness=False,
            primary_gate_rescue=False,
            branches={
                b: dict(
                    status="blocked",
                    absolute_metric=None,
                    numerator=None,
                    denominator=0,
                    censor_reason="missing_conformal_protocol_and_frozen_predictions",
                )
                for b in [
                    "clopper_pearson_cluster_risk",
                    "betting_e_process_cluster_risk",
                    "mycelium_provenance_slices",
                ]
            },
        ),
        original_method_protocols=deepcopy(source_v785["protocol_references"]),
    )
    result["science_decisions"]["validation"].update(
        qualified_checker_correction=False,
        v785_replay_authority_status="qualified",
        v785_contract_status="blocked",
        current_replay_authority_status="blocked",
        current_corpus_scan_status="blocked",
        current_full_qualification_status="blocked",
    )
    result["validation_obligations"]["v785_affected_obligations"] = source_v785[
        "validation_obligations"
    ]
    result["arc_summary"].update(
        current_ARC_experiment_scheduled=False, historical_experiment_id=9192
    )
    return dict(result)
