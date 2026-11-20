"""REQ-REPORT-9268-V788: recompute original units and preserve V787 and V788 branch eligibility."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v782_capstone_evidence as prior
from carnot.reporting.v763_capstone_reduction import require

Json = dict[str, Any]


def reduce(work: Json) -> Json:
    """Use authenticated V782-V788 primitives and Exp9211/Exp9225/Exp9244/Exp9258 evidence without inventing model science."""
    from carnot.reporting import v788_capstone_evidence as e

    require(e.canonical_hash(work["tasks"]) == e.TASK_PIN, "task_authority")
    old_v782 = work["values"]["prior:9193"]["value"]
    old_v783 = work["values"]["prior:9207"]["value"]
    old_v784 = work["values"]["prior:9221"]["value"]
    old_v786 = work["values"]["prior:9240"]["value"]
    saved_v782 = work["operands"][old_v782["work_reference"]["path"]]
    historical = json.loads(Path(saved_v782["path"]).read_bytes())
    for key in ["snapshots", "operands"]:
        historical[key] = {k: work["operands"][r["path"]] for k, r in historical[key].items()}
    result = prior.reduce(historical)
    source_v785 = work["values"]["prior:9222"]["value"]
    source_v787 = work["values"]["prior:9241"]["value"]
    ambiguity = work["values"]["prior:9211"]["value"]
    deferral = work["values"]["prior:9225"]["value"]
    conformal = work["values"]["prior:9244"]["value"]
    source = work["values"][e.TASKS[0]]["value"]
    prefilter = work["values"][e.TASKS[1]]["value"]
    victris = work["values"][e.TASKS[3]]["value"]
    rows = e.outcomes(work)
    v787_rows = e.v787_outcomes(work)
    require(
        len(rows) == 13 and all(r["accounting_evidence_reference"] for r in rows),
        "outcome_accounting",
    )
    require(sum(r["primary_status"] == "present" for r in rows) == 3, "frozen_execution_scope")
    require(sum(r["primary_status"] == "missing_file" for r in rows) == 10, "frozen_skip_scope")
    require(
        sum(r["disposition"] == "authenticated_scheduler_skip" for r in rows) == 1
        and sum(r["disposition"] == "authenticated_scheduler_log_skip" for r in rows) == 9,
        "v788_scheduler_skip_partition",
    )
    previous_v784 = old_v784["outcome_rows"]
    previous_v786 = old_v786["outcome_rows"]
    v785_rows = old_v786["preserved_v785_outcomes"]
    require(
        len(previous_v786) == 14
        and sum(r["primary_status"] == "missing_file" for r in previous_v786) == 13,
        "previous_v786_accounting",
    )
    require(
        old_v786["preserved_v782_accounting"]
        == dict(executed=4, skipped=10, scientific_success_inferred=False),
        "preserved_v782_accounting",
    )
    require(
        old_v786["preserved_v783_accounting"]
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
        old_v786["preserved_v784_accounting"]
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
        old_v786["preserved_v785_accounting"]
        == dict(
            executed=2,
            skipped=3,
            scheduler_artifact_skips=1,
            scheduler_log_skips=2,
            omitted_slots=9,
            scientific_success_inferred=False,
        ),
        "preserved_v785_accounting",
    )
    require(
        old_v786["current_milestone_accounting"]
        == dict(
            executed=1,
            skipped=13,
            scheduler_artifact_skips=0,
            scheduler_log_skips=13,
            scheduler_pretest_log_skips=2,
            scheduler_upstream_retired_log_skips=11,
            scientific_success_inferred=False,
        ),
        "preserved_v786_accounting",
    )
    require(
        source_v787["replay_authority_ready_score"] == 0
        and source_v787["contract_ready_score"] == 0
        and source_v787["source_ready_score"] == 0
        and source_v787["static_methods_ready_score"] == 0
        and source_v787["online_methods_ready_score"] == 0
        and source_v787["required_checks_passed"] is False
        and len(source_v787["consumer_shard_rows"]) == 6
        and all(r["passed"] and r["exit_code"] == 0 for r in source_v787["consumer_shard_rows"])
        and len(source_v787["full_task_objects"]) == 7
        and source_v787["censored_count"] == 7,
        "v787_replay_contract_state",
    )
    require(
        len(v787_rows) == 14
        and sum(r["primary_status"] == "present" for r in v787_rows) == 2
        and sum(r["primary_status"] == "missing_file" for r in v787_rows) == 5
        and sum(r["primary_status"] == "missing_field" for r in v787_rows) == 7,
        "v787_outcome_accounting",
    )
    require(
        source["replay_authority_ready_score"] == 1
        and source["contract_ready_score"] == 1
        and source["source_ready_score"] == 1
        and source["static_methods_ready_score"] == 1
        and source["online_methods_ready_score"] == 1
        and source["required_checks_passed"] is True
        and len(source["consumer_shard_rows"]) == 7
        and all(r["passed"] and r["exit_code"] == 0 for r in source["consumer_shard_rows"])
        and source["censored_count"] == 0
        and source["preserved_v787"]["exp9241_owned_statement_coverage"]
        == dict(
            complete=True,
            percent=100.0,
            measured_file_count=5,
            owned_changed_path_count=6,
        ),
        "v788_replay_contract_state",
    )
    require(
        prefilter["corpus_scan_ready_score"] == 0
        and prefilter["verdict_class"] == "disqualified"
        and prefilter["required_checks_passed"] is False
        and prefilter["candidate_counts"]["visited"] == 764150
        and prefilter["candidate_counts"]["unfinished"] == 0
        and prefilter["candidate_counts"]["changed"] == 0
        and prefilter["corpus_manifest_reference"]["candidate_bytes_stable"] is True,
        "v788_corpus_prefilter_state",
    )
    require(
        victris["victris_harness_ready_score"] == 1
        and victris["verdict_class"] == "blocked"
        and victris["honest_verdict"] == "complete_blocked_arena_sim_wheel_missing"
        and victris["required_checks_passed"] is True
        and victris["arena_sim_installed"] is False
        and victris["heldout_matches_played"] == 0
        and victris["scientific_effect_measured"] is False
        and len(victris["archive_versions"]) == 6
        and all(r["max_tick_ms"] < 10.0 for r in victris["archive_versions"]),
        "v788_victris_state",
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
    provenance_ledger = json.loads(
        Path(
            work["operands"][conformal["provenance_ledger_reference"]["path"]]["path"]
        ).read_bytes()
    )
    require(
        len(conformal["rows"]) == len(provenance_ledger["rows"]) == 900
        and conformal["effective_independent_groups"] == 300
        and conformal["conformal_risk_ready_score"] == 1
        and conformal["ca_ocrc_ready_score"] == 1
        and conformal["provenance_audit_ready_score"] == 1
        and conformal["required_checks_passed"] is True
        and conformal["scientific_effect_measured"] is False
        and len(conformal["coverage_control_rows"]) == 8
        and all(r["passed"] for r in conformal["coverage_control_rows"])
        and len(conformal["edge_control_rows"]) == 3
        and all(r["passed"] for r in conformal["edge_control_rows"])
        and len(conformal["ca_ocrc_control_rows"]) == 12
        and all(r["passed"] for r in conformal["ca_ocrc_control_rows"]),
        "conformal_risk_method_qualified",
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
                *previous_v786,
                *v787_rows[:7],
            ]
        )
    }
    for row in before.values():
        if (
            row.get("scheduler_events")
            and row.get("primary_status") == "missing_file"
            and not row.get("scheduler_alternative")
            and not row.get("scheduler_honest_verdict")
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
        disposition="disqualified_full_suite_timeout",
        evidence="Exp9255 qualified all five readiness scores (contract_ready_score=1, replay_authority_ready_score=1) across seven consumer shards and 14-task parity; Exp9256 traversed all 764,150 recursive JSON files with zero parity discrepancies and passed the target module (1544.2s), owned tests, V782 controls, and E2E-018/021, but disqualified corpus_scan_ready_score=0 when full_python_suite timed out at 599.87s (-9)",
        next_reopen_condition="Keep repository-wide pytest health separate from scoped validation receipts (repository_health_separate=true) as in contract/capstone producers so passing lossless corpus traversal, target module, V782 controls, and 100 percent owned coverage qualify corpus_scan_ready_score=1",
    )
    decisions[1].update(
        evidence="Exp9257 refused at conductor pre-gate on exp9256-corpus-prefilter.corpus_scan_ready_score==0; original 674 and all affected obligations never completed in one normal pass",
        next_reopen_condition="Qualify corpus_scan_ready_score=1 first, then execute original 621 plus 53 and every affected V780-V788 obligation once with original assertions and current receipts; never depend on Exp9181/9182",
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
                branch="conformal_risk_ca_ocrc_and_mycelium_protocol",
                disposition="qualified_independent_method",
                evidence="Exp9244 independently qualified qid-cluster conformal risk bounds (conformal_risk_ready_score=1), CA-OCRC capacity-constrained delayed-feedback IPW tracking (ca_ocrc_ready_score=1), and the 900-response Mycelium provenance ledger (provenance_audit_ready_score=1)",
                next_reopen_condition="Retain immutable Exp9244 protocols, ledger and reducer; consume only as non-interfering shadow and descriptive branches after eligible frozen predictions exist",
            ),
            dict(
                branch="victris_arena_offline_energy_rsi",
                disposition="terminal_blocked_missing_simulator_wheel",
                evidence="Exp9258 qualified victris_harness_ready_score=1 across six sub-10ms Gibbs state-energy archive versions with strict untrusted-shout isolation and DEV/HELDOUT disjointness, and blocked honestly on absent arena_sim wheel with heldout_matches_played=0",
                next_reopen_condition="Supply authenticated arena_sim-0.83.0.whl (sha256:5393811a...) in .venv before running dev RSI selection and one-shot n=100 held-out Wilson-CI evaluation",
            ),
        ]
    )
    v787_accounting = dict(
        executed=2,
        skipped=5,
        scheduler_artifact_skips=1,
        scheduler_log_skips=4,
        omitted_slots=7,
        scientific_success_inferred=False,
    )
    result.update(
        historical_outcomes=deepcopy(v787_rows),
        preserved_v782_accounting=deepcopy(old_v786["preserved_v782_accounting"]),
        preserved_v783_accounting=deepcopy(old_v786["preserved_v783_accounting"]),
        preserved_v784_accounting=deepcopy(old_v786["preserved_v784_accounting"]),
        preserved_v785_accounting=deepcopy(old_v786["preserved_v785_accounting"]),
        preserved_v786_accounting=deepcopy(old_v786["current_milestone_accounting"]),
        preserved_v787_accounting=deepcopy(v787_accounting),
        previous_milestone_accounting=deepcopy(v787_accounting),
        preserved_v785_outcomes=deepcopy(v785_rows),
        preserved_v786_outcomes=deepcopy(previous_v786),
        preserved_v787_outcomes=deepcopy(v787_rows),
        current_milestone_accounting=dict(
            executed=4,
            skipped=10,
            scheduler_artifact_skips=1,
            scheduler_log_skips=9,
            scientific_success_inferred=False,
        ),
        retirement_recommendations=retirements,
        continuation_decisions=decisions,
        preserved_v782_retirement=deepcopy(old_v782["retirement_recommendations"]),
        preserved_v783_retirement=deepcopy(old_v783["retirement_recommendations"]),
        preserved_v784_retirement=deepcopy(old_v784["retirement_recommendations"]),
        preserved_v786_retirement=deepcopy(old_v786["retirement_recommendations"]),
        replay_authority_summary=dict(
            status="qualified",
            replay_authority_ready_score=source["replay_authority_ready_score"],
            contract_ready_score=source["contract_ready_score"],
            source_ready_score=source["source_ready_score"],
            static_methods_ready_score=source["static_methods_ready_score"],
            online_methods_ready_score=source["online_methods_ready_score"],
            consumer_shard_rows=deepcopy(source["consumer_shard_rows"]),
            protocol_owner_rows=deepcopy(source["protocol_owner_rows"]),
            v787_replay_authority_ready_score=source_v787["replay_authority_ready_score"],
            v787_contract_ready_score=source_v787["contract_ready_score"],
            v787_source_ready_score=source_v787["source_ready_score"],
            v787_static_methods_ready_score=source_v787["static_methods_ready_score"],
            v787_online_methods_ready_score=source_v787["online_methods_ready_score"],
            v787_consumer_shard_rows=deepcopy(source_v787["consumer_shard_rows"]),
            v787_owned_statement_coverage=deepcopy(
                source["preserved_v787"]["exp9241_owned_statement_coverage"]
            ),
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
            status="disqualified",
            corpus_scan_ready_score=prefilter["corpus_scan_ready_score"],
            v787_corpus_scan_ready_score="missing_field",
            v785_corpus_scan_ready_score="missing_field",
            qualified_checker_correction=False,
            lossless_input_parity_measured=True,
            checker_semantics_changed=False,
            full_corpus_traversal_complete=True,
            candidate_counts=deepcopy(prefilter["candidate_counts"]),
            disqualification_reason="full_python_suite_timed_out_at_599.87s_exit_minus_9",
            evidence=after[e.TASKS[1]],
        ),
        forward_qualification_summary=dict(
            status="blocked",
            forward_validation_ready_score="missing_field",
            failed_upstream_gate_observed=0,
            complete_obligation_execution=False,
            current_qualification_credit=0,
            evidence=after[e.TASKS[2]],
        ),
        victris_arena_summary=dict(
            status="blocked",
            harness_status="qualified",
            victris_harness_ready_score=victris["victris_harness_ready_score"],
            arena_sim_installed=victris["arena_sim_installed"],
            heldout_matches_played=victris["heldout_matches_played"],
            scientific_effect_measured=victris["scientific_effect_measured"],
            archive_versions=deepcopy(victris["archive_versions"]),
            heldout_summary=deepcopy(victris["heldout_summary"]),
            censor_reason="arena_sim_wheel_missing",
            evidence=after[e.TASKS[3]],
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
            method_status="qualified",
            conformal_risk_ready_score=conformal["conformal_risk_ready_score"],
            ca_ocrc_ready_score=conformal["ca_ocrc_ready_score"],
            provenance_audit_ready_score=conformal["provenance_audit_ready_score"],
            mycelium_provenance_ready_score=conformal["provenance_audit_ready_score"],
            conformal_risk_protocol_reference=deepcopy(
                conformal["conformal_risk_protocol_reference"]
            ),
            ca_ocrc_protocol_reference=deepcopy(conformal["ca_ocrc_protocol_reference"]),
            provenance_ledger_reference=deepcopy(conformal["provenance_ledger_reference"]),
            reducer_reference=deepcopy(conformal["reducer_reference"]),
            provenance_summary=deepcopy(conformal["provenance_summary"]),
            method_evidence=v787_rows[3],
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
                    "clopper_pearson_cluster_risk",
                    "betting_e_process_cluster_risk",
                    "ca_ocrc_ipw_shadow_tracking",
                    "mycelium_provenance_slices",
                ]
            },
        ),
        ca_ocrc_summary=dict(
            method_status="qualified",
            ca_ocrc_ready_score=conformal["ca_ocrc_ready_score"],
            ca_ocrc_protocol_reference=deepcopy(conformal["ca_ocrc_protocol_reference"]),
            method_evidence=v787_rows[3],
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
                for b in ["capacity_1", "capacity_2", "capacity_5"]
            },
        ),
        original_method_protocols=deepcopy(source["protocol_references"]),
    )
    result["science_decisions"]["validation"].update(
        qualified_checker_correction=False,
        v785_replay_authority_status="qualified",
        v785_contract_status="blocked",
        v787_replay_authority_status="blocked",
        v787_contract_status="blocked",
        current_replay_authority_status="qualified",
        current_contract_status="qualified",
        current_corpus_scan_status="disqualified",
        current_full_qualification_status="blocked",
    )
    result["validation_obligations"]["v785_affected_obligations"] = source_v785[
        "validation_obligations"
    ]
    result["arc_summary"].update(
        current_ARC_experiment_scheduled=False, historical_experiment_id=9192
    )
    return dict(result)
