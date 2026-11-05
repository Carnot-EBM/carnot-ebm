"""REQ-REPORT-8826: reconciliation readiness never substitutes for scientific benefit."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v755_capstone as c
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash

Json = dict[str, Any]


def build(work: Json, validation: Json) -> Json:
    """Retain each branch result while missing natural and device operands block overall science."""
    owned = (
        bool(validation["receipts"])
        and all(r["passed"] for r in validation["receipts"])
        and validation["coverage"]["complete"]
    )
    ready = bool(owned and work["contract"]["ready"] and all(r["accounted"] for r in work["slots"]))
    verdict = "blocked" if owned else "disqualified"
    rows = [
        {k: deepcopy(v) for k, v in r.items() if k not in {"producer", "snapshots", "reduction"}}
        for r in work["slots"]
    ]
    rows.append(
        dict(
            task_id=c.TASK,
            experiment_id=8826,
            unit_id=c.TASK,
            cluster_id="administrative_task_slots",
            arm="scheduled_task",
            seed=None,
            metric=int(ready),
            denominator=1,
            status="completed",
            missing_reason=None,
            accounted=owned,
            disposition="validated_reconciliation" if owned else "disqualified",
            verdict_class=verdict,
            primary_honest_verdict="complete_" + verdict + "_v755_capstone",
            admitted=owned,
        )
    )
    branches, rejected, gates = [], [], deepcopy(work["edges"])
    for slot in work["slots"]:
        producer = slot["producer"]
        branches.append(
            dict(
                task_id=slot["task_id"],
                verdict_class=slot["verdict_class"],
                disposition=slot["disposition"],
                admitted=slot["admitted"],
                exposure_scope=producer.get("exposure_scope", "no admitted measurement"),
                reduction=slot["reduction"],
                acceptance_gates=producer.get("acceptance_gates", {}),
                scores={k: v for k, v in producer.items() if k.endswith("_score")},
            )
        )
        gates += producer.get("gate_check_summary", [])
        if not slot["admitted"]:
            rejected.append(
                dict(
                    task_id=slot["task_id"],
                    reference=slot["primary_reference"],
                    errors=slot["admission_errors"],
                    reported_completed_rows=producer["reported_completed_rows"],
                    rejected_constructed_measurement_count=producer["reported_completed_rows"]
                    if slot["experiment_id"] in {8815, 8818, 8819}
                    else 0,
                    disposition=slot["disposition"],
                )
            )
            gates.append(
                dict(
                    slot["primary_reference"],
                    check="producer_admission",
                    upstream=slot["task_id"],
                    artifact_field="qualified_primary",
                    expected=True,
                    observed=False,
                    missing_reason="rejected_producer"
                    if producer.get("task_id")
                    else "missing_file",
                    passed=False,
                )
            )
    retirements = []
    by_id = {r["task_id"]: r for r in rows}
    for prior in work["priors"]:
        row = by_id[prior["task_id"]]
        registration = prior["registration"]
        same = bool(
            prior["actual_honest_verdict"]
            and prior["actual_honest_verdict"] == registration["verdict"]
            and prior["actual_verdict_class"] == row["verdict_class"]
            and row["verdict_class"] in {"null", "blocked", "disqualified"}
        )
        retirements.append(
            dict(
                task_id=row["task_id"],
                prior_verdict=prior["actual_honest_verdict"],
                registered_prior_verdict=registration["verdict"],
                prior_reference=prior["reference"],
                new_verdict=row["primary_honest_verdict"],
                new_disposition=row["disposition"],
                changed_mechanism=registration["addressed_by"],
                acceptance_gate=next(
                    t["gated_on"] for t in work["contract"]["tasks"] if t["id"] == row["task_id"]
                ),
                same_terminal_class=same,
                same_mechanism_observed=same
                and row["disposition"] in {"conductor_pre_gate", "conductor_cascade_skip"},
                recommendation="retire_exact_scope_pending_changed_evidence"
                if same and registration["retire_if_same_verdict"]
                else "continue_only_with_changed_evidence",
                evidence_needed_to_reopen=registration["addressed_by"],
                scope="this task and registered mechanism only",
                manifest_edited=False,
            )
        )
    reductions = [s["reduction"] for s in work["slots"] if s["admitted"]]
    counts = {
        key: sum(r[key] for r in reductions)
        for key in [
            "constructed_measurement_count",
            "independent_constructed_unit_count",
            "natural_generalization_count",
        ]
    }
    refs = work["snapshots"] + [r for slot in work["slots"] for r in slot["snapshots"]]
    refs += [r["reference"] for r in work["priors"] if r["reference"]["sha256"]]
    refs += c.references(dict(validation_receipts=validation["receipts"]))
    refs += [
        validation["coverage"][k]
        for k in ["report_reference", "data_reference"]
        if k in validation["coverage"]
    ]
    if not owned:
        gates.append(
            dict(
                c.reference(Path(work["raw"]) / "validation.json"),
                check="owned_validation",
                upstream=c.TASK,
                artifact_field="required_checks_passed",
                expected=True,
                observed=False,
                missing_reason="rejected_producer",
                passed=False,
            )
        )
    receipt = validation.get("current_work_receipt", {})
    value = dict(
        experiment_id=8826,
        task_id=c.TASK,
        milestone=c.MILESTONE,
        run_date=work["run_date"],
        honest_verdict="complete_" + verdict + "_v755_capstone",
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        verifier_is_oracle=False,
        duration_s=receipt.get("duration_s", 0.001),
        random_seed=7558826,
        phase_spans=receipt.get("phase_spans", []),
        current_work_receipt=receipt,
        source_artifact_hashes=work["snapshots"],
        code_config_hashes=work["code"],
        raw_shard_hashes=refs,
        preconditions_checked=dict(
            full_task_authority=work["contract"]["ready"], no_external_activation=True
        ),
        rows=rows,
        task_outcome_rows=rows,
        intended_count=13,
        completed_count=13,
        failed_count=sum(r["verdict_class"] == "disqualified" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=sum(r["disposition"] == "quarantined_evidence" for r in rows),
        independent_count=0,
        gate_check_summary=[g for g in gates if not g.get("passed", False)],
        gate_edge_rows=work["edges"],
        acceptance_gates=dict(
            owned_validation=owned, complete_accounting=ready, natural_generalization=False
        ),
        required_checks_passed=owned,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        exposure_scope="thirteen-task custody; admitted constructed evidence and CPU nulls only",
        independent_generalization_score=0,
        reconciliation_ready_score=int(ready),
        branch_conclusions=branches,
        rejected_evidence=rejected,
        retirement_recommendations=retirements,
        full_task_objects=work["contract"]["tasks"],
        canonical_tasks_sha256=c.authority.PIN,
        visible_task_table=work["contract"]["table"],
        historical_publication_scope="historical FoVer paper only",
        historical_publication_gates=work["historical_publication_gates"],
        publication_authorized=False,
        activation_authorized=False,
        cited_upstream_artifacts=[
            dict(
                experiment_id=s["experiment_id"],
                fields_imported=["rows", "honest_verdict", "acceptance_gates"],
                **s["primary_reference"],
            )
            for s in work["slots"]
        ],
        global_test_debt=validation.get("global_test_debt", {}),
        flagged_adversarial=False,
        methodology_note="Independent registered row reductions retain constructed nulls. Exact oracle success remains circular. Missing natural extraction, ARC outcomes and qualified hardware costs block overall science.",
        work_reference=c.reference(Path(work["raw"]) / "work.json"),
        validation_state=validation,
        **counts,
    )
    value["field_principles"] = dict.fromkeys(
        [*value, "field_principles", "reproducibility_checksum"],
        "Bind exact evidence; count constructed measurements separately from natural generalization and administrative completion.",
    )
    value["reproducibility_checksum"] = c.checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild derived fields from frozen rows so a repaired checksum cannot hide a false count."""
    try:
        value = json.loads(path.read_bytes())
        for ref in (
            value["raw_shard_hashes"] + value["code_config_hashes"] + [value["work_reference"]]
        ):
            if ref["sha256"] and c.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("bound_operand_drift")
        work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
        if canonical_hash(work["contract"]["tasks"]) != "sha256:" + c.authority.PIN:
            raise ValueError("task_authority_drift")
        for slot in work["slots"]:
            if slot["admitted"]:
                producer = json.loads(Path(slot["primary_reference"]["path"]).read_bytes())
                if (
                    c.producer_summary(producer) != slot["producer"]
                    or c.reduce(producer) != slot["reduction"]
                ):
                    raise ValueError("producer_reduction_drift")
        passed = value == build(work, value["validation_state"])
        return dict(passed=passed, reason="replay_passed" if passed else "semantic_drift")
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="invalid_or_changed_evidence")
