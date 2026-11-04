"""REQ-REPORT-8800: distinguish complete accounting from qualified scientific claims."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from carnot.reporting import v753_capstone as c
from carnot.reporting import v752_capstone_report as previous
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]


def build(work: Json, validation: Json, output: Path) -> Json:
    """Exact failed operands close comparisons; scheduling completion grants no benefit."""
    owned = (
        bool(validation["receipts"])
        and all(r["passed"] for r in validation["receipts"])
        and validation["coverage"]["complete"]
        and validation["union"]["passed"]
    )
    checked = owned and work["preconditions"]["projection_fits"]
    verdict = "disqualified" if not owned else "blocked"
    honest = (
        "complete_blocked_resource"
        if owned and not checked
        else "complete_" + verdict + "_v753_capstone"
    )
    rows = deepcopy(work["slots"])
    gates, admissions = [], []
    by_id = {r["task_id"]: r for r in rows}
    for task, row in zip(work["tasks"], rows):
        admissions.append(
            dict(
                task_id=row["task_id"],
                reference=row["primary_reference"],
                admitted=row["producer_qualified"] and not row["excluded"],
                headline_admitted=False,
                disposition=row["disposition"],
                authentication_errors=row["authentication_errors"],
                imported_fields=[
                    "honest_verdict",
                    "verdict_class",
                    "required_checks_passed",
                    "flagged_adversarial",
                    "duration_s",
                    "exposure_scope",
                ],
            )
        )
        for gate in task["gated_on"]:
            producer = by_id[gate["upstream"]]
            actual = c.observe(
                producer["primary_reference"],
                gate["upstream"],
                gate["artifact_field"],
                gate["value"],
            )
            gates.append(
                dict(
                    actual,
                    consumer_task_id=task["id"],
                    producer_qualified=producer["producer_qualified"],
                    passed=actual["passed"] and producer["producer_qualified"],
                )
            )
        for recorded in row["conductor_gate_receipts"]:
            ref = dict(path=recorded["artifact_path"], sha256=recorded["artifact_sha256"])
            actual = c.observe(
                ref, recorded["upstream"], recorded["artifact_field"], recorded["expected"]
            )
            gates.append(
                dict(
                    actual,
                    consumer_task_id=task["id"],
                    check="conductor_receipt_matches_operand",
                    expected=[recorded["actual"], recorded["passed"]],
                    observed=[actual["observed"], actual["passed"]],
                    passed=[actual["observed"], actual["passed"]]
                    == [recorded["actual"], recorded["passed"]],
                )
            )
        if row["authentication_errors"] or row["excluded"]:
            gates.append(
                dict(
                    row["primary_reference"],
                    upstream=task["id"],
                    check="producer_admission",
                    field="exact_identity_raw_validation_and_quarantine",
                    operator="==",
                    expected=True,
                    observed="quarantined_evidence" if row["excluded"] else "rejected_producer",
                    passed=False,
                )
            )
    self_row: Json = dict(
        task_id=c.TASK,
        experiment_id=8800,
        unit_id=c.TASK,
        arm="scheduled_task",
        status="completed",
        absolute_metric=int(checked),
        raw_numerator=int(checked),
        raw_denominator=1,
        censored=False,
        excluded=False,
        effective_independent_groups=0,
        producer_executed=True,
        producer_qualified=checked,
        honest_verdict=honest,
        verdict_class=verdict,
        scheduler_outcome="self_reconciliation",
        primary_reference=dict(path=str(output.absolute()), sha256=None),
        conductor_reference=None,
        conductor_gate_receipts=[],
        conductor_log_reference=None,
        conductor_disposition_lines=[],
        authentication_errors=[],
        evidence_references=[],
        cost_s=None,
        independent_count=0,
        disposition="validated_reconciliation" if checked else verdict,
        scientific_measurement_admitted=False,
    )
    rows.append(self_row)
    decisions, retirements = [], []
    for task, row in zip(work["tasks"], rows, strict=True):
        gate_operands = deepcopy(task["gated_on"])
        change = {
            8789: "Correct original failure-publication and archive validation with unchanged assertions; memory_protocol_ready_score must equal1 with all owned checks passing.",
            8791: "Correct quarantined intervention controls and strict types; evidence_interventions_ready_score must equal1 on fresh disjoint held-out worlds.",
            8796: "Supply exact authenticated changed learning contract; observer qualification must pass overhead<=0.05 and reconciliation error<=0.02 on its complete workload.",
            8800: "Supply the missing qualified causal-learning, natural-extraction, ARC and physical-operation operands; each producer must pass its frozen acceptance gates.",
        }.get(
            row["experiment_id"],
            "Change the exact upstream primary bytes and satisfy every registered producer field in falsifiable_gates before executing this closed comparison.",
        )
        if row["experiment_id"] == 8793:
            change += " The overflow positive control must pass; otherwise pending-feedback usefulness remains unmeasured."
        decisions.append(
            dict(
                task_id=task["id"],
                comparison_closed=True,
                automatic_rerun_authorized=False,
                changed_operand=change,
                falsifiable_gates=gate_operands,
                verdict_class=row["verdict_class"],
            )
        )
        for prior in task["prior_failures"]:
            evidence = next(
                p
                for p in work["prior_outcomes"]
                if p["task_id"] == task["id"] and p["prior_task_id"] == prior["experiment_id"]
            )
            same = bool(
                row["honest_verdict"] is not None
                and row["honest_verdict"] == evidence["actual_honest_verdict"]
                and evidence["registered_verdict_matches"]
                and evidence["exact_identity_matches"]
            )
            skip = row["scheduler_outcome"] in {"conductor_pre_gate", "conductor_cascade_skip"}
            retirements.append(
                dict(
                    task_id=task["id"],
                    registration=prior,
                    prior_evidence=evidence,
                    observed_producer_verdict=row["honest_verdict"],
                    scheduler_outcome=row["scheduler_outcome"],
                    exact_repeat=same,
                    explicit_conductor_skip=skip,
                    recommendation="retire_unchanged_scope"
                    if prior["retire_if_same_verdict"] and (same or skip)
                    else "close_prior_comparison",
                    retire_if_same_verdict=prior["retire_if_same_verdict"],
                    reopening_conditions=[change],
                    falsifiable_gates=gate_operands,
                    manifest_edited=False,
                )
            )
    end = validation["ended_at_utc"]
    duration = (
        datetime.fromisoformat(end) - datetime.fromisoformat(work["started_at_utc"])
    ).total_seconds()
    publication = work["publication_gates"]
    primitive = c.reference(Path(work["raw"]) / "work.json")
    validation_ref = c.reference(Path(work["raw"]) / "validation.json")
    for field, observed in work["preconditions"].items():
        if observed is False:
            gates.append(
                dict(
                    primitive,
                    upstream=c.TASK,
                    check="precondition",
                    field=field,
                    operator="==",
                    expected=True,
                    observed=False,
                    passed=False,
                )
            )
    if not owned:
        gates.append(
            dict(
                validation_ref,
                upstream=c.TASK,
                check="owned_validation",
                field="required_checks_passed",
                operator="==",
                expected=True,
                observed=False,
                passed=False,
            )
        )
    value = dict(
        experiment_id=8800,
        task_id=c.TASK,
        milestone=c.MILESTONE,
        run_date=work["run_date"],
        invocation_id=work["invocation_id"],
        honest_verdict=honest,
        verdict_class=verdict,
        gate_check_summary=[g for g in gates if not g["passed"]],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_language_model_calls=0,
        numerical_training_calls=0,
        rows=rows,
        intended_count=12,
        completed_count=12,
        failed_count=sum(r["verdict_class"] == "disqualified" for r in rows),
        censored_count=0,
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=0,
        n_samples=0,
        verifier_is_oracle=False,
        exposure_scope="administrative twelve-slot reconciliation; no scientific measurement or independent reproduction",
        independent_generalization_score=0,
        acceptance_gates=dict(
            scoped_validation=owned,
            exact_task_custody=len(rows) == 12,
            resource_projection=work["preconditions"]["projection_fits"],
        ),
        required_checks_passed=owned,
        flagged_adversarial=False,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        preconditions_checked=work["preconditions"],
        duration_s=duration,
        started_at_utc=work["started_at_utc"],
        ended_at_utc=end,
        phase_spans=[
            dict(
                phase="freeze_and_validation",
                start_utc=work["started_at_utc"],
                end_utc=end,
                duration_s=duration,
            )
        ],
        random_seed=7538800,
        source_artifact_hashes=work["references"],
        code_config_hashes=work["sources"],
        raw_shard_hashes=[primitive, validation_ref, *validation.get("raw_references", [])],
        cited_upstream_artifacts=work["references"],
        repository_health_reference=work.get("repository_health_reference", {}),
        scoped_validation_scope=dict(
            test_files=c.TESTS,
            consumers=c.CONSUMERS,
            e2e=["E2E-018", "E2E-021"],
            native_round_trip_owned=False,
            repository_health_is_dependency=False,
        ),
        reconciliation_ready_score=int(checked),
        task_outcome_rows=rows,
        gate_edge_rows=gates,
        full_task_objects=work["tasks"],
        canonical_tasks_sha256=c.PIN,
        visible_task_table=work["visible_table"],
        source_admission_rows=admissions,
        retirement_recommendations=retirements,
        next_measurement_priorities=decisions,
        component_cost_ready_score=0,
        learning_benefit_ready_score=0,
        pending_feedback_useful_score=0,
        scientific_measurement_count=0,
        publication_gates=publication,
        paper_ready=publication["paper_ready"],
        unmet_gates=publication["unmet_gates"],
        activation_authorized=False,
        publication_authorized=False,
        primitive_reference=primitive,
        validation_reference=validation_ref,
        publication_output=str(output.absolute()),
        original_node_union=validation["union"],
        methodology_note="Account for all twelve exact V753 tasks, including four primary conductor skips, four cascade skips, disqualified qualification, quarantined methods, blocked observer work and this validated capstone. No task completion, parameter movement or constructed feedback establishes learning benefit. No literature-inspired held-out comparison executed; prior constructed evidence is history only. No model load, numerical training, hidden-game transfer, physical acceleration or independent reproduction is claimed.",
        **publication["gates"],
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Bind exact producer fields and immutable operands; distinguish missing science from zero, rejected work, quarantine and scheduler accounting.",
    )
    value["reproducibility_checksum"] = c.digest(value)
    return value


# Reuse the sealed-byte replay, with isolated globals so V752 remains unchanged.
_replay = namespace(previous, "v753_capstone_replay")
_replay.c, _replay.build = c, build
replay = _replay.replay
