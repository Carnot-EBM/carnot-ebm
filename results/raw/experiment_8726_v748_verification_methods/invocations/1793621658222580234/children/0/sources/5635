"""REQ-REPORT-8694: bookkeeping readiness preserves scientific blocks and nulls."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import v745_capstone_inputs as e
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.v744_capstone_report import REOPEN as ORIGINAL_REOPEN

Json = dict[str, Any]
REOPEN = dict(
    ORIGINAL_REOPEN,
    methods="Qualify unchanged original Exp8681 nodes, assertions, coverage retention and source custody in a new producer; preserve Exp8667/8681 disqualification.",
    arc="V744 horizon diagnostic stays closed; Exp8691 is quarantined. New causal mechanism, reset-cause separation and uncensored cross-game support are required for policy efficacy.",
    hardware="Keep Exp8692 CPU32/24 feasibility separate. Implement and authenticate a target-board spline kernel, measured complete-service clocks, transfer and durable updates; preserve all three board restrictions.",
)


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Passing owned checks grants reconciliation only; missing science stays blocked."""
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    ready = owned and work["contract"]["ready"] and work["preconditions_passed"]
    verdict = "blocked" if owned else "disqualified"
    rows = deepcopy(work["slots"])
    rows.append(
        dict(
            experiment_id=8694,
            task_id=e.TASK,
            unit_id=e.TASK,
            arm="scheduled_slot",
            status="completed",
            censored=False,
            excluded=False,
            absolute_metric=int(ready),
            effective_independent_groups=0,
            producer_executed=True,
            producer_qualified=ready,
            eligible=False,
            scientific_result_measured=False,
            disposition="capstone_reconciliation",
            honest_verdict=f"complete_{verdict}_v745_capstone",
            verdict_class=verdict,
            flagged_adversarial=False,
            adversarial_findings=[],
            missing_reason=None,
        )
    )
    decisions = []
    for task, row in zip(work["protocol"]["task_map_v745"], rows, strict=True):
        history = []
        for registration in task.get("prior_failures", []):
            old = work["prior"].get(registration["experiment_id"], {})
            literal = old.get("honest_verdict")
            repeat = bool(
                row["producer_executed"]
                and old.get("terminal_sealed")
                and literal is not None
                and literal == registration["verdict"] == row["honest_verdict"]
            )
            history.append(
                dict(
                    registration,
                    literal_prior_verdict=literal,
                    exact_authenticated_repeat=repeat,
                    retire_scope=repeat and registration["retire_if_same_verdict"],
                )
            )
        decisions.append(
            dict(
                task_id=task["id"],
                scope=task["title"],
                decision="stop_registered_scope",
                reason="Registered diagnostic or reconciliation closes once"
                if row["producer_executed"]
                else "Prerequisite execution absent; hypothesis remains untested",
                producer_executed=row["producer_executed"],
                actual_prior_outcomes=history,
                retire_exact_scope=any(r["retire_scope"] for r in history),
                retired_scientific_hypothesis=False,
                automatically_queue_another_version=False,
                changed_operands_needed_to_reopen=REOPEN,
            )
        )
    scores = dict(
        reconciliation_ready_score=int(ready),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        methods_ready_score=0,
        learning_execution_ready_score=0,
        learning_rows_ready_score=0,
        corpus_ready_score=0,
        batch_service_ready_score=0,
        batch_cost_ready_score=0,
        runtime_repair_ready_score=0,
        arc_decision_ready_score=0,
        precision_mapping_ready_score=int(ready and work["reductions"]["precision"][-1]["passed"]),
        hardware_mapping_ready_score=int(ready),
        physical_hardware_ready_score=0,
        promotion_ready_score=0,
        nfr01_score=0,
    )
    reduced, publication = work["reductions"], work["publication"]
    gaps = [
        dict(gap=scope, status="blocked", reopening_operands=REOPEN[operand])
        for scope, operand in [
            ("verification_calibration_continuous_self_learning", "learning"),
            ("efficient_deployable_inference", "deployment"),
            ("live_agent_cross_game_outcomes", "arc"),
        ]
    ]
    value: Json = dict(
        experiment_id=8694,
        task_id=e.TASK,
        milestone="2026.11.745",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v745_capstone",
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        substrate_by_phase=dict(
            readers="aggregation_from_upstream_artifacts",
            constructed_checks="deterministic_verifier",
            numerical_head="verifier_ensemble_against_cached_candidates; upstream rows only",
        ),
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        current_language_model_calls=0,
        numerical_inference_this_invocation=False,
        target_model=dict(
            kind="authenticated_numerical_energy_head",
            parameter_count=109,
            checkpoint=work["protocol"]["head_checkpoint_reference"],
            physical_fpga_run=False,
        ),
        historical_model_provenance=work["historical_models"],
        rows=rows,
        task_outcome_rows=rows,
        intended_count=14,
        completed_count=14,
        failed_count=0 if owned else 1,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            intended_scheduling_slots=14,
            new_independent_scientific_units=0,
            producer_denominators="Original questions, constructed worlds and games; seeds/views/timings do not multiply N",
        ),
        scheduling_completed_count=14,
        producer_executed_count=sum(r["producer_executed"] for r in rows),
        scientific_measurement_count=sum(r["scientific_result_measured"] for r in rows),
        verifier_is_oracle=False,
        exposure_scope="Exposed cached observations and constructed integer controls; no unexposed natural generalization",
        required_checks_passed=owned,
        flagged_adversarial=False,
        adversarial_findings=[],
        acceptance_gates=dict(
            owned_reconciliation=ready,
            essential_learning=False,
            independent_semantics=False,
            same_batch_10x=False,
            physical_execution=False,
        ),
        validation_receipts=deepcopy(receipts),
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        started_at_utc=work["started_at_utc"],
        ended_at_utc=work["ended_at_utc"],
        phase_spans=work["phase_spans"],
        random_seed=7458694,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["operand_refs"],
        cited_upstream_artifacts=[r["primary_reference"] for r in rows[:-1]],
        full_task_objects=work["protocol"]["task_map_v745"],
        canonical_tasks_sha256=e.methods.TASKS_PIN,
        task_contract_rows=work["contract"]["rows"],
        gate_edge_rows=work["gate_rows"],
        gate_check_summary=work["gate_rows"] + work["external_checks"],
        prd_gap_rows=gaps,
        learning_decision_rows=reduced["learning"],
        semantic_decision_rows=reduced["semantics"],
        deployment_decision_rows=reduced["deployment"],
        arc_decision_rows=reduced["arc"],
        hardware_decision_rows=reduced["hardware"],
        precision_decision_rows=reduced["precision"],
        retirement_decisions=decisions,
        reopening_operands=REOPEN,
        literature_adopt_defer_ledger=reduced["literature"],
        calibration_self_learning_floor=dict(
            status="blocked_unexecuted", GAP_ORACLE_DISTINCT="open"
        ),
        arc_standing_floor=dict(
            status="closed_censored_diagnostic",
            hidden_generalization_established=False,
            headline_solve_credit=0,
            policy_change_authorized=False,
        ),
        current_quarantines=[r for r in rows if r.get("flagged_adversarial")],
        publication_gates=publication,
        unmet_gates=publication["unmet_gates"],
        paper_ready=all(publication["gates"][g]["pass"] for g in ("G1", "G2", "G3", "G4")),
        publication_authorized=False,
        branch_gate_values=scores,
        primitive_reference=e.reference(Path(work["raw"]) / "measurement.json"),
        validation_reference=e.reference(Path(work["raw"]) / "validation.json"),
        invocation_id=work["invocation_id"],
        publication_output=str(output),
        methodology_note="Deterministic read-only reconciliation. All fourteen scheduling outcomes retained. Historical stamps persist; absent prerequisite execution is unmeasured. CPU precision is constructed feasibility, not a physical result or independent detector win.",
    )
    value.update(deepcopy(reduced["history"]))
    value.update(scores)
    value.update({g: publication["gates"][g] for g in ("G1", "G2", "G3", "G4")})
    value["field_principles"] = {
        k: "Bind original bytes, invocation and scope; scheduling, scientific benefit and execution qualification remain distinct."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = e.checksum(value)
    return value
