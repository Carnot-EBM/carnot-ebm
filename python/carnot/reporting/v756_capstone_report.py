"""REQ-REPORT-8839: evidence readiness and scientific benefit use separate denominators."""

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v756_capstone as c
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS, canonical_hash

Json = dict[str, Any]


def build(work: Json, validation: Json) -> Json:
    """Keep independent branch conclusions even while missing external science blocks the total."""
    owned = (
        bool(validation["receipts"])
        and all(
            r["passed"] for r in validation["receipts"] if r.get("scope") != "repository_health"
        )
        and validation["coverage"]["complete"]
        and validation["coverage"]["percent"] == 100
    )
    ready = bool(owned and work["contract"]["ready"] and all(r["accounted"] for r in work["slots"]))
    verdict = "blocked" if owned else "disqualified"
    honest = "complete_" + verdict + "_v756_capstone"
    rows = [
        {k: deepcopy(v) for k, v in r.items() if k not in {"producer", "snapshots", "reduction"}}
        for r in work["slots"]
    ]
    rows.append(
        dict(
            task_id=c.TASK,
            experiment_id=8839,
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
            primary_honest_verdict=honest,
            admitted=owned,
        )
    )
    branches, rejected, retirements = [], [], []
    gates = deepcopy(work["edges"])
    if not owned:
        gates.append(
            dict(
                path=str(Path(work["raw"]) / "validation.json"),
                sha256=None,
                check="owned_validation",
                upstream=c.TASK,
                artifact_field="required_checks_passed",
                expected=True,
                observed=False,
                passed=False,
                missing_reason="rejected_producer",
            )
        )
    historical = {r["task_id"]: r for r in work["historical"]["rows"]}
    by_id = {r["task_id"]: r for r in rows}
    for slot in work["slots"]:
        producer = slot["producer"]
        branches.append(
            dict(
                task_id=slot["task_id"],
                disposition=slot["disposition"],
                verdict_class=slot["verdict_class"],
                admitted=slot["admitted"],
                exposure_scope=producer.get("exposure_scope", "no admitted measurement"),
                reduction=slot["reduction"],
                acceptance_gates=producer.get("acceptance_gates", {}),
                scores={k: v for k, v in producer.items() if k.endswith("_score")},
            )
        )
        gates.extend(producer.get("gate_check_summary", []))
        if not slot["admitted"]:
            rejected.append(
                dict(
                    task_id=slot["task_id"],
                    reference=slot["primary_reference"],
                    errors=slot["admission_errors"],
                    disposition=slot["disposition"],
                    reported_row_count=producer["reported_completed_rows"],
                    science_admitted=False,
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
                    passed=False,
                    missing_reason="rejected_producer"
                    if producer.get("task_id")
                    else "missing_file",
                )
            )
    for task in work["contract"]["tasks"]:
        for prior in task["prior_failures"]:
            old = historical.get(prior["experiment_id"], {})
            actual = old.get("primary_honest_verdict")
            row = by_id[task["id"]]
            retirements.append(
                dict(
                    c.retirement(prior, row, actual),
                    task_id=task["id"],
                    prior_reference=old.get(
                        "primary_reference", c.reference(c.ROOT / c.authority.HISTORY)
                    ),
                    current_reference=row.get(
                        "primary_reference", dict(path="current_capstone", sha256=None)
                    ),
                    observed_gate=[g for g in work["edges"] if g["downstream"] == task["id"]],
                    prior_scheduler_receipts=old.get("scheduler_receipts", []),
                    scope=task["title"],
                )
            )
    reductions = [r["reduction"] for r in work["slots"] if r["admitted"]]
    units = sorted({u for r in reductions for u in r["constructed_unit_ids"]})
    refs = work["snapshots"] + [r for s in work["slots"] for r in s["snapshots"]]
    refs.extend(c.shared.references(dict(validation_receipts=validation["receipts"])))
    refs.extend(
        validation["coverage"][k]
        for k in ["report_reference", "data_reference"]
        if k in validation["coverage"]
    )
    health_ref = validation.get("repository_health", {}).get("log_reference")
    if health_ref:
        refs.append(health_ref)
    receipt = validation.get("current_work_receipt", {})
    value = dict(
        experiment_id=8839,
        task_id=c.TASK,
        milestone=c.MILESTONE,
        run_date=work["run_date"],
        honest_verdict=honest,
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        numerical_training_calls=0,
        verifier_is_oracle=False,
        exposure_scope="thirteen current task slots; admitted constructed worlds and complete CPU costs",
        independent_generalization_score=0,
        duration_s=receipt.get("duration_s", 0),
        random_seed=7568839,
        current_work_receipt=receipt,
        phase_spans=receipt.get("phase_spans", []),
        preconditions_checked=work["preconditions"],
        source_artifact_hashes=work["snapshots"],
        code_config_hashes=work["code"],
        raw_shard_hashes=refs,
        rows=rows,
        task_outcome_rows=rows,
        intended_count=13,
        completed_count=sum(r["accounted"] for r in rows),
        failed_count=sum(r["verdict_class"] == "disqualified" for r in rows),
        censored_count=sum(r["status"] == "censored" for r in rows),
        excluded_count=sum(r["disposition"] == "quarantined_evidence" for r in rows),
        independent_count=0,
        gate_check_summary=[g for g in gates if not g.get("passed", False)],
        gate_edge_rows=work["edges"],
        acceptance_gates=dict(
            owned_validation=dict(
                passed=owned,
                threshold=100,
                principle="All owned checks and statements must qualify this producer",
            ),
            complete_accounting=dict(
                passed=ready,
                threshold=13,
                principle="Every scheduled slot needs primary or skip custody",
            ),
            natural_generalization=dict(
                passed=False,
                threshold=1,
                principle="Constructed or oracle checks do not establish natural generalization",
            ),
        ),
        required_checks_passed=owned,
        validation_receipts=validation["receipts"],
        owned_statement_coverage=validation["coverage"],
        reconciliation_ready_score=int(ready),
        branch_conclusions=branches,
        rejected_evidence=rejected,
        retirement_recommendations=retirements,
        constructed_measurement_count=sum(r["constructed_measurement_count"] for r in reductions),
        independent_constructed_unit_count=len(units),
        constructed_unit_ids=units,
        natural_generalization_count=0,
        cpu_cost_window_count=sum(r["cpu_cost_window_count"] for r in reductions),
        historical_counts={
            k: work["historical"][k]
            for k in [
                "constructed_measurement_count",
                "independent_constructed_unit_count",
                "natural_generalization_count",
            ]
        },
        historical_publication_scope="historical FoVer paper only",
        historical_publication_gates=validation.get(
            "historical_publication_gates", work["historical"]["historical_publication_gates"]
        ),
        publication_authorized=False,
        activation_authorized=False,
        full_task_objects=work["contract"]["tasks"],
        canonical_tasks_sha256=c.authority.PIN,
        visible_task_table=work["contract"]["table"],
        work_reference=c.reference(Path(work["raw"]) / "work.json"),
        validation_state=validation,
        repository_health=validation.get("repository_health", {}),
        flagged_adversarial=False,
        methodology_note="Independent frozen row reductions preserve admitted nulls and circular checks. Nested variants and seeds do not create worlds. Hardware mappings reuse CPU operands. External natural extraction and live outcomes remain missing; completed accounting grants no publication or activation authority.",
    )
    value["field_principles"] = {
        k: "Bind current evidence and keep administrative completion separate from scientific benefit: "
        + k
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = c.checksum(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild reductions and the report so rehashed false totals cannot pass a fresh reader."""
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
            admitted = c.admit(Path(slot["primary_reference"]["path"]), slot["task_id"])
            if admitted["admitted"] != slot["admitted"]:
                raise ValueError("producer_admission_drift")
            if slot["admitted"]:
                producer = json.loads(Path(slot["primary_reference"]["path"]).read_bytes())
                if (
                    c.shared.producer_summary(producer) != slot["producer"]
                    or c.reduce(producer) != slot["reduction"]
                ):
                    raise ValueError("producer_reduction_drift")
        passed = value == build(work, value["validation_state"])
        return dict(passed=passed, reason="replay_passed" if passed else "semantic_drift")
    except (OSError, ValueError, KeyError, TypeError):
        return dict(passed=False, reason="invalid_or_changed_evidence")
