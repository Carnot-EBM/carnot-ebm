"""REQ-REPORT-8708: deterministic terminal decisions retain each scientific boundary."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import v746_capstone_inputs as e
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS

Json = dict[str, Any]
REOPEN = {
    "learning": "Qualify the unchanged union nodes and assertions in a new sealed producer; require prediction-before-feedback, future proper-loss gain and simultaneous retention before claiming benefit.",
    "relational": "Supply authenticated480 numerical head probabilities with original candidate/label joins; source-to-record fidelity and natural independent truth remain separate.",
    "deployment": "Repair and freshly qualify the original installed envelope assertions, then measure complete matched same-batch Python/Rust cost and batching separately, including launch, transfer, parsing and durable cost.",
    "qwen": "Require a dated causal runtime repair and qualified symbolic controls before the registered bounded36 Qwen calls; no unchanged probe or model substitution.",
    "arc": "Closed censored diagnostic. Reopen only for a distinct intervention, separated reset causes and uncensored cross-game support; keep curated policy unchanged.",
    "numeric": "Retain constructed32/24 CPU lifecycle limits; a changed domain or kernel requires new range, fallback, dense-renormalization, restart and full-cost evidence.",
    "hardware": "Require a compatible operation-specific kernel, source/build hashes, actual device parity and complete clocks. KV260 k<=5 Ising, PolarFire Linux CPU and GateMate physical/JTAG restrictions persist.",
    "authority": "Require changed scientific operands or a new causal mechanism; administrative readiness alone authorizes no rerun, activation or publication.",
}


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Owned validation can qualify bookkeeping while incomplete external science remains blocked."""
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    ready = bool(
        owned
        and not work["private_control"]
        and work["contract"]["ready"]
        and work["preconditions_passed"]
    )
    verdict = "blocked" if owned else "disqualified"
    rows = deepcopy(work["slots"])
    rows.append(
        dict(
            experiment_id=8708,
            task_id=e.TASK,
            unit_id=e.TASK,
            arm="scheduled_slot",
            status="completed",
            absolute_metric=int(ready),
            censored=False,
            excluded=False,
            effective_independent_groups=0,
            producer_executed=True,
            producer_qualified=ready,
            scientific_result_measured=False,
            scientific_verdict=None,
            honest_verdict=f"complete_{verdict}_v746_capstone",
            verdict_class=verdict,
            flagged_adversarial=False,
            disposition="capstone_reconciliation",
            owned_validation_passed=owned,
            validation_receipt_count=len(receipts),
            owned_validation_duration_s=sum(r.get("duration_s", 0.0) for r in receipts),
        )
    )
    values = work["values"]
    arc, numeric, board = (values.get(str(n), {}) for n in (8705, 8706, 8707))
    scores = dict(
        reconciliation_ready_score=int(ready),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        physical_hardware_ready_score=0,
        learning_execution_ready_score=0,
        learning_rows_ready_score=0,
        batch_service_ready_score=0,
        batch_cost_ready_score=0,
        runtime_repair_ready_score=0,
        symbolic_control_ready_score=0,
        promotion_ready_score=0,
        nfr01_score=0,
    )
    for field, number in [
        ("contract_ready_score", 8695),
        ("learning_authority_ready_score", 8695),
        ("learning_core_ready_score", 8696),
        ("corpus_ready_score", 8698),
        ("arc_decision_ready_score", 8705),
        ("lifecycle_rows_ready_score", 8706),
        ("numerical_lifecycle_ready_score", 8706),
        ("hardware_mapping_ready_score", 8707),
    ]:
        row = next(r for r in rows if r["experiment_id"] == number)
        scores[field] = int(
            ready and row["producer_qualified"] and values[str(number)].get(field) == 1
        )
    retirement = []
    for prior in work["prior_rows"]:
        current = next(r for r in rows if r["task_id"] == prior["task_id"])
        exact = bool(
            prior["terminal_authenticated"]
            and prior["registered_verdict_matches"]
            and current["scientific_verdict"] == prior["actual_terminal_verdict"]
            and prior["prior_scope"] == prior["current_scope"]
        )
        retirement.append(
            dict(
                prior,
                exact_authenticated_repeat=exact,
                exact_scope_match=prior["prior_scope"] == prior["current_scope"],
                retirement_candidate=exact and prior["registration"]["retire_if_same_verdict"],
                changed_method_reopen_conditions=REOPEN,
            )
        )
    decisions = [
        dict(
            task_id=t["id"],
            scope=t["title"],
            decision="stop_unchanged_scope",
            producer_executed=r["producer_executed"],
            scientific_hypothesis_retired=False,
            automatic_rerun_authorized=False,
            actual_causal_difference_required=True,
            reopening_conditions=REOPEN,
        )
        for t, r in zip(work["contract"]["tasks"], rows, strict=True)
    ]
    history = work["history"]
    value: Json = dict(
        experiment_id=8708,
        task_id=e.TASK,
        milestone=e.MILESTONE,
        run_date=work["run_date"],
        invocation_id=work["invocation_id"],
        honest_verdict=f"complete_{verdict}_v746_capstone",
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        current_language_model_calls=0,
        target_model=dict(
            kind="authenticated_upstream_numerical_head",
            parameter_count=109,
            checkpoint=work["checkpoint_reference"],
            physical_fpga_run=False,
        ),
        historical_model_provenance={
            k: v.get("historical_model_provenance", []) for k, v in values.items()
        },
        rows=rows,
        task_outcome_rows=rows,
        intended_count=14,
        completed_count=14,
        failed_count=int(not owned),
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        producer_executed_count=sum(r["producer_executed"] for r in rows),
        skipped_study_count=sum(not r["producer_executed"] for r in rows),
        sample_size_budget=dict(
            intended_scheduling_slots=14,
            new_independent_scientific_units=0,
            denominator="Original questions/worlds/games; seeds, views, heads and repeats do not multiply N",
        ),
        verifier_is_oracle=False,
        exposure_scope="Cached exposed and constructed upstream evidence; no independent natural generalization",
        required_checks_passed=owned,
        flagged_adversarial=False,
        adversarial_findings=[],
        acceptance_gates=dict(
            owned_reconciliation=ready,
            future_loss_and_retention=False,
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
        random_seed=7468708,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["operand_refs"],
        cited_upstream_artifacts=[r["primary_reference"] for r in rows[:-1]],
        full_task_objects=work["contract"]["tasks"],
        visible_task_table=work["contract"]["visible_table"],
        canonical_tasks_sha256=work["contract"]["canonical_tasks_sha256"],
        task_contract_rows=work["contract"]["comparison"]["rows"],
        gate_edge_rows=work["gate_rows"],
        gate_check_summary=work["gate_rows"] + work["external_checks"],
        retirement_candidate_rows=retirement,
        next_decision_rows=decisions,
        learning_decision_rows=dict(
            measured=False,
            future_brier=None,
            joint_retention=None,
            reason="Exp8697 owned union failed; Exp8699/8700 did not execute",
        ),
        semantic_decision_rows=dict(
            measured=False,
            missing_numerical_probabilities=480,
            source_to_record_independent_generalization=False,
            reason="Exp8701 has no authenticated480 P2 probabilities; constructed relational controls do not establish natural truth",
        ),
        deployment_decision_rows=dict(
            measured=False,
            same_batch_rust_ratio=None,
            batching_ratio=None,
            qwen_calls=0,
            bounded_qwen_call_budget=36,
            reason="Exp8702 owned validation failed; Exp8703/8704 skipped",
        ),
        arc_decision_rows={
            k: arc.get(k)
            for k in ("complete_horizon_count", "policy_decision", "generalization_floor_decision")
        },
        numerical_decision_rows={
            k: numeric.get(k)
            for k in (
                "rows",
                "complete_operation_cost_rows",
                "accumulator_bound_rows",
                "boundary_rows",
                "decay_error_rows",
                "restart_rows",
            )
        },
        hardware_decision_rows={
            k: board.get(k)
            for k in (
                "board_status_rows",
                "operation_support_rows",
                "complete_cost_rows",
                "conditional_acceleration_bounds",
            )
        },
        historical_quarantines=history["historical_quarantines"],
        preserved_nulls=history["preserved_nulls"],
        preserved_v737_supervisor_null=history["preserved_v737_supervisor_null"],
        preserved_v743_batch_one_null=history["preserved_v743_batch_one_null"],
        historical_stamped_outcomes=work["historical_stamps"],
        historical_outcome_archive=history["historical_stamped_outcomes"],
        exp8681_failure_evidence={
            k: values["8695"]["historical_failure_reproduction"][k]
            for k in (
                "original_node_count",
                "retained_child_sqlite_count",
                "final_retained_reference_count",
                "original_rejection_line_executed",
            )
        },
        current_quarantines=[r for r in rows if r["verdict_class"] == "disqualified"],
        publication_gates=work["publication"],
        unmet_gates=work["publication"]["unmet_gates"],
        paper_ready=all(work["publication"]["gates"][g]["pass"] for g in ("G1", "G2", "G3", "G4")),
        publication_authorized=False,
        activation_authorized=False,
        branch_gate_values=scores,
        primitive_reference=e.reference(Path(work["raw"]) / "measurement.json"),
        validation_reference=e.reference(Path(work["raw"]) / "validation.json"),
        publication_output=str(output),
        methodology_note="Read-only terminal reconciliation of fourteen frozen full task objects. Upstream terminal receipts and primitives retain original failures. Missing science stays blocked; constructed CPU arithmetic supplies no independent learning or physical acceleration.",
    )
    value.update(scores)
    value.update({g: work["publication"]["gates"][g] for g in ("G1", "G2", "G3", "G4")})
    value["field_principles"] = {
        k: "Bind actual task, invocation, byte hash and claim scope; absence is not zero and administrative readiness is not scientific benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = e.digest(value)
    return value
