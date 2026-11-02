"""REQ-REPORT-8722: keep qualified local accounting separate from blocked science."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import v747_capstone_inputs as e
from carnot.reporting.v747_capstone_reduce import reduce
from carnot.reporting.v746_capstone_report import REOPEN
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS

Json = dict[str, Any]


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Preserve valid nulls and terminal blocks while reporting this invocation's actual checks."""
    owned = bool(receipts) and all(r["passed"] for r in receipts)
    ready = int(
        owned
        and not work["private_control"]
        and work["contract"]["ready"]
        and work["preconditions_passed"]
    )
    verdict = "blocked" if owned else "disqualified"
    rows = deepcopy(work["slots"])
    rows.append(
        dict(
            experiment_id=8722,
            task_id=e.TASK,
            unit_id=e.TASK,
            arm="scheduled_slot",
            status="completed",
            absolute_metric=ready,
            censored=False,
            excluded=False,
            effective_independent_groups=0,
            producer_executed=True,
            producer_qualified=bool(ready),
            scientific_result_measured=False,
            scientific_verdict=None,
            honest_verdict=f"complete_{verdict}_v747_capstone",
            verdict_class=verdict,
            flagged_adversarial=False,
            disposition="capstone_reconciliation",
            owned_validation_passed=owned,
            validation_receipt_count=len(receipts),
            owned_validation_duration_s=sum(r.get("duration_s", 0.0) for r in receipts),
        )
    )
    science = reduce(work)
    scores = dict(
        reconciliation_ready_score=ready,
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        physical_hardware_ready_score=0,
        learning_rows_ready_score=0,
        learning_execution_ready_score=0,
        learning_cost_ready_score=0,
        learning_benefit_ready_score=0,
        runtime_repair_ready_score=0,
        same_batch_10x_score=0,
        nfr01_score=science.get("batch", {}).get("nfr01_score", 0),
        batching_benefit_score=science.get("batch", {}).get("batching_benefit_score", 0),
    )
    for row in rows[:-1]:
        if row["producer_qualified"]:
            upstream = e.load(row["primary_reference"])
            scores.update(
                {
                    k: int(v)
                    for k, v in upstream.get("branch_gate_values", {}).items()
                    if k not in scores and type(v) is int and v in (0, 1)
                }
            )
    retirement = []
    for prior in work["prior_rows"]:
        current = next(r for r in rows if r["task_id"] == prior["task_id"])
        scope_match = prior["prior_scope"] == prior["current_scope"]
        exact = bool(
            prior["terminal_authenticated"]
            and prior["registered_verdict_matches"]
            and current["scientific_verdict"] == prior["actual_terminal_verdict"]
            and scope_match
        )
        retirement.append(
            dict(
                prior,
                exact_scope_match=scope_match,
                exact_authenticated_repeat=exact,
                retirement_candidate=exact and prior["registration"]["retire_if_same_verdict"],
                changed_method_reopen_conditions=REOPEN,
                scope_rule="Literal registered scope and authenticated terminal verdict; skips are not scientific repeats",
            )
        )
    decisions = [
        dict(
            task_id=task["id"],
            scope=task["title"],
            decision="close_valid_null"
            if row["verdict_class"] == "null"
            else "stop_unchanged_scope",
            producer_executed=row["producer_executed"],
            automatic_rerun_authorized=False,
            scientific_hypothesis_retired=False,
            priority=i + 1,
            reopening_conditions=task.get("prior_failures", []),
            changed_mechanism_conditions=REOPEN,
        )
        for i, (task, row) in enumerate(zip(work["contract"]["tasks"], rows, strict=True))
    ]
    history = e.load(work["history_reference"])
    value: Json = dict(
        experiment_id=8722,
        task_id=e.TASK,
        milestone=e.MILESTONE,
        run_date=work["run_date"],
        invocation_id=work["invocation_id"],
        honest_verdict=f"complete_{verdict}_v747_capstone",
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        current_language_model_calls=0,
        target_model=None,
        historical_model_provenance=history["historical_model_provenance"],
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
            denominator="Original worlds, questions and games; views, seeds and repeats do not multiply N",
        ),
        verifier_is_oracle=False,
        exposure_scope="Constructed and exposed upstream evidence; no natural independent generalization",
        required_checks_passed=owned,
        flagged_adversarial=False,
        adversarial_findings=[],
        acceptance_gates=dict(
            owned_reconciliation=bool(ready),
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
        random_seed=7478722,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["operand_refs"],
        cited_upstream_artifacts=[r["primary_reference"] for r in rows[:-1]],
        full_task_objects=work["contract"]["tasks"],
        visible_task_table=work["contract"]["visible_table"],
        canonical_tasks_sha256=work["contract"]["canonical_tasks_sha256"],
        task_contract_rows=work["contract"]["rows"],
        gate_edge_rows=work["gate_rows"],
        gate_check_summary=work["gate_rows"] + work["external_checks"],
        retirement_candidate_rows=retirement,
        next_decision_rows=decisions,
        scientific_reconstruction=science,
        learning_decision_rows=science["learning"],
        semantic_decision_rows=science.get("relational", {}),
        deployment_decision_rows=science.get("batch", {}),
        numerical_decision_rows=science.get("compiled", {}),
        hardware_decision_rows=science.get("hardware", {}),
        arc_decision_rows=dict(
            closed_panel=history["arc_decision_rows"], new_outcomes=science.get("arc", {})
        ),
        historical_quarantines=history["historical_quarantines"],
        preserved_nulls=history["preserved_nulls"],
        preserved_v737_supervisor_null=history["preserved_v737_supervisor_null"],
        preserved_v743_batch_one_null=history["preserved_v743_batch_one_null"],
        historical_stamped_outcomes=history["historical_stamped_outcomes"],
        historical_outcome_archive=history["historical_outcome_archive"],
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
        next_measurement_priorities=[
            "Repair and qualify current learning execution, then measure future Brier and retention",
            "Measure independent semantics with changed representation; keep constructed oracle scope explicit",
            "Retain batching gain and same-batch null; measure a changed kernel only against matched full costs",
            "Require actual operation-specific board parity and complete clocks before physical speed claims",
        ],
        methodology_note="Read-only reconciliation of fourteen complete V747 objects. Original rows reconstruct lexical collisions, verifier detections, complete batch intervals and CPU integer parity. Missing learning, Qwen execution and physical acceleration remain terminal blocked. Historical verdicts remain unchanged.",
    )
    value.update(scores)
    value.update({g: work["publication"]["gates"][g] for g in ("G1", "G2", "G3", "G4")})
    value["field_principles"] = {
        k: "Bind actual invocation, original bytes and scientific scope; missing evidence differs from zero, readiness differs from benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = e.digest(value)
    return value
