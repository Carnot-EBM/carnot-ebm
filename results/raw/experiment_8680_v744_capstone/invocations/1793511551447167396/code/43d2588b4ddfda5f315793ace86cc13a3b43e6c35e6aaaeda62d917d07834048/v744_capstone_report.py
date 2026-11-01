"""REQ-REPORT-8680: successful reconciliation can coexist with blocked science."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import v744_capstone_inputs as e
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS

Json = dict[str, Any]
REOPEN = {
    "methods": "Fix original owned_tests, exact_node_union and newly owned coverage failures under frozen sources; do not clear the Exp8667 flag retroactively.",
    "learning": "Fresh unchanged-source39-node core plus25-node deadline cohort, original64/27 union and83 statements, then registered causal updates and independent question-cluster future Brier/joint retention. Historical39 passes with changed bytes remain disqualified.",
    "semantics": "Qualified120 worlds/four byte-bound views, original independent evaluator, collision witnesses and equivalent-wording controls; unexposed independent natural truth and oracle-distinct controls are separate requirements.",
    "deployment": "Qualified installed vectorized batch mechanism with request/decision parity;60 equal-work batch cells and61440 evaluations; same-batch native/Python lower95>=10 separately from transport gain. Preserve V735 cold-cost and V743 batch-one nulls.",
    "qwen": "Changed CUDA environment/library hashes, actual context/copy success, qualified claim corpus and real tokens/device receipts before any optional Qwen call.",
    "arc": "Ownership repair qualifies only the frozen source. The four-game diagnostic closes with censored horizons; a new causal supervisor mechanism with uncensored cross-game support is needed for efficacy. Preserve V737 null and Exp8664 quarantine.",
    "hardware": "Compatible implemented numerical kernel, fixed-point range/error and typed-decision parity, disjoint arithmetic/transfer/update/store clocks, durable updates and authenticated physical execution. KV260 stays quadratic Ising k_max<=5; PolarFire Linux CPU; GateMate dated change then IDCODE0x20000001 then ordered n16 smoke. NPU/TSU unqualified.",
    "generalization": "Unexposed independent questions and an oracle-distinct matched control; constructed or exposed wins cannot clear GAP-ORACLE-DISTINCT.",
}


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Owned qualification grants bookkeeping readiness, never absent scientific benefit."""
    owned = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and work["contract"]["ready"]
        and work["preconditions_passed"]
    )
    verdict = "blocked" if owned else "disqualified"
    rows = deepcopy(work["slots"])
    rows.append(
        dict(
            experiment_id=8680,
            task_id=e.TASK,
            unit_id=e.TASK,
            arm="scheduled_slot",
            status="completed",
            censored=False,
            excluded=False,
            absolute_metric=int(owned),
            effective_independent_groups=0,
            eligible=owned,
            producer_executed=True,
            scientific_result_measured=False,
            disposition="capstone_reconciliation",
            honest_verdict=f"complete_{verdict}_v744_capstone",
            verdict_class=verdict,
            literal_verdict_class=verdict,
            flagged_adversarial=False,
            adversarial_findings=[],
            failed_or_missing_gate_operands=[],
            missing_reason=None,
        )
    )
    decisions = e.decisions(work["protocol"], rows, work["prior_values"])
    for row in decisions:
        row["changed_operands_needed_to_reopen"] = REOPEN
        row["stop_condition"] = (
            "Stop this exact scope after its registered result; do not rerun unchanged external absence or treat unexecuted science as null."
        )
    scores = dict(
        reconciliation_ready_score=int(owned),
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        methods_ready_score=0,
        learning_operands_ready_score=0,
        learning_core_ready_score=0,
        learning_execution_ready_score=0,
        learning_rows_ready_score=0,
        corpus_ready_score=0,
        batch_service_ready_score=0,
        batch_cost_ready_score=0,
        runtime_repair_ready_score=0,
        observer_delivery_ready_score=int(owned),
        arc_rows_ready_score=int(owned),
        promotion_ready_score=0,
        physical_hardware_ready_score=0,
        hardware_mapping_ready_score=int(owned),
        nfr01_score=0,
    )
    reduced, publication = work["reductions"], work["publication"]
    gaps = [
        dict(
            gap="verification_and_calibrated_continuous_self_learning",
            status="blocked",
            requirements=["FR-06", "FR-11", "FR-12", "GAP-ORACLE-DISTINCT"],
            evidence="No current learning union, causal updates or retention audit; constructed registration is not a collision experiment. Natural source coverage retains55/190 in128 exposed groups.",
            reopening_operands=[REOPEN["learning"], REOPEN["semantics"], REOPEN["generalization"]],
        ),
        dict(
            gap="efficient_deployable_inference",
            status="blocked",
            requirements=["FR-05", "FR-08", "NFR-01"],
            evidence="No V744 batch prototype or timing; preserve qualified V743 compatibility, packaging noninferiority and complete10x null. Physical spline support absent.",
            reopening_operands=[REOPEN["deployment"], REOPEN["hardware"]],
        ),
        dict(
            gap="live_agent_cross_game_outcomes",
            status="closed_censored_diagnostic_efficacy_blocked",
            requirements=["ARC-STANDING-FLOOR"],
            evidence="Fresh ownership/parity qualifies;16 censored horizons and zero complete horizons cannot estimate supervisor benefit.",
            reopening_operands=[REOPEN["arc"]],
        ),
    ]
    gate_rows = work["gate_rows"]
    value: Json = dict(
        experiment_id=8680,
        task_id=e.TASK,
        milestone="2026.11.744",
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v744_capstone",
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        substrate_by_phase=dict(
            readers="aggregation_from_upstream_artifacts",
            constructed_checks="deterministic_verifier",
        ),
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        current_language_model_calls=0,
        numerical_inference_this_invocation=False,
        historical_model_provenance=work["historical_models"],
        target_model=dict(
            kind="authenticated_numerical_energy_head",
            checkpoint=work["protocol"]["head_checkpoint_reference"],
            parameter_count=109,
            numerical_inference_this_invocation=False,
        ),
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
            producer_denominators="Original question clusters, worlds and games; seeds, siblings, views and repeats do not multiply N",
        ),
        verifier_is_oracle=False,
        exposure_scope="Exposed upstream evidence and deterministic constructed registration; no independent generalization",
        required_checks_passed=owned,
        flagged_adversarial=False,
        acceptance_gates=dict(
            owned_reconciliation=owned,
            essential_learning=False,
            independent_semantics=False,
            batch_10x=False,
            physical_execution=False,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=[],
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=8680,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["operand_refs"],
        cited_upstream_artifacts=[r["primary_reference"] for r in rows[:-1]],
        full_task_objects=work["protocol"]["task_map_v744"],
        canonical_tasks_sha256=e.methods.TASKS_PIN,
        task_contract_rows=work["contract"]["rows"],
        gate_edge_rows=gate_rows,
        gate_check_summary=gate_rows + work["external_checks"],
        prd_gap_rows=gaps,
        learning_decision_rows=reduced["learning"],
        semantic_witness_rows=reduced["semantics"],
        constructed_registration=reduced["constructed_registration"],
        deployment_decision_rows=reduced["deployment"],
        arc_decision_rows=reduced["arc"],
        hardware_decision_rows=reduced["hardware"],
        qwen_decision_rows=reduced["qwen"],
        retirement_decisions=decisions,
        reopening_operands=REOPEN,
        preserved_nulls=work["history"]["preserved_nulls"],
        historical_quarantines=work["history"]["historical_quarantines"],
        historical_stamped_outcomes=reduced["historical_stamped_outcomes"],
        preserved_v737_supervisor_null=work["history"]["preserved_v737_supervisor_null"],
        preserved_v743_batch_one_null=reduced["historical"]["installed"],
        current_quarantines=[
            dict(
                task_id=r["task_id"],
                flagged_adversarial=r["flagged_adversarial"],
                verdict_class=r["verdict_class"],
                adversarial_findings=r["adversarial_findings"],
                failed_owned_checks=r.get("failed_owned_checks", []),
            )
            for r in rows
            if r["flagged_adversarial"]
        ],
        original_union_audit=reduced["historical_unions"],
        calibration_self_learning_floor=dict(
            status="blocked_unexecuted",
            GAP_ORACLE_DISTINCT="open",
            required_operands=REOPEN["learning"],
        ),
        arc_standing_floor=dict(
            status="diagnostic_completed_efficacy_unmeasured",
            hidden_generalization_established=False,
            headline_solve_credit=0,
        ),
        literature_adopt_defer_ledger=work["protocol"]["literature_method_rows"],
        publication_gates=publication,
        unmet_gates=publication["unmet_gates"],
        paper_ready=all(publication["gates"][g]["pass"] for g in ("G1", "G2", "G3", "G4")),
        publication_authorized=False,
        branch_gate_values=scores,
        primitive_reference=e.reference(Path(work["raw"]) / "measurement.json"),
        validation_reference=e.reference(Path(work["raw"]) / "validation.json"),
        invocation_id=work["invocation_id"],
        publication_output=str(output),
        methodology_note="Read-only deterministic reduction. V744 pre-gated hypotheses are unmeasured. Historical qualification/nulls/quarantines remain original; current clean readers cannot promote old flagged claims.",
    )
    value.update(scores)
    value.update({g: publication["gates"][g] for g in ("G1", "G2", "G3", "G4")})
    value["field_principles"] = {
        k: "Bind this invocation to original evidence and scope; qualification is separate from benefit and external absence remains blocked."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = e.checksum(value)
    return value
