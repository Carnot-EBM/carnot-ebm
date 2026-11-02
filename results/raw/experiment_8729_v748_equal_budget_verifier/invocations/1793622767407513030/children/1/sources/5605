"""REQ-REPORT-8639: administrative closure preserves scientific claim boundaries."""

from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v739_capstone_report as previous
from carnot.reporting import v741_capstone_evidence as e
from carnot.reporting import v741_capstone_inputs as inputs
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from scripts.recurring_blocker_ledger import normalise

Json = dict[str, Any]
SYNTHETIC_RECEIPTS = {
    "exact_disjoint_node_union",
    "new_owned_statement_coverage_100",
    "private_publication_readers",
}

REOPEN = dict(
    learning="Retain immutable original child SQLite coverage shards before combining; qualify83/83 statements, the27-node family and deadline3599/3600/3601 controls. Then seal fresh causal predictions, released feedback and simultaneous retention under unchanged gates.",
    streaming="Pass owned streaming validation and original common-domain parity; then measure complete windows and recovered support on the same190 responses in128 groups. Completeness cannot establish semantic correctness.",
    packaging="Pass the immutable original installed compatibility node union and cleanup controls; then measure all120 registered complete installed/private cost cells for noninferiority. Packaging does not calibrate a verifier.",
    nfr01="Change the registered native mechanism and measure complete matched native/Python lower95>=10; the V740 lean lifecycle benefit is a different contrast.",
    qwen="Supply causal CUDA environment/library repair hashes and genuine context/copy success before a leased mandated-model canary; no unchanged probe.",
    arc="A changed causal supervisor and authenticated cross-game outcomes meeting original three-game/five-firing support; real observer horizons alone cannot justify policy benefit.",
    hardware="Supply qualified new learning or installed costs with disjoint host/lifecycle/transport spans and compatible measured physical execution; retain existing board restrictions.",
)


def decisions(work: Json, rows: list[Json], prior: Json) -> list[Json]:
    """Compare actual prior outcomes as well as literal registrations and normalized causes."""
    result = []
    history = prior["rows"] + prior["quarantined_inputs"] + prior["qualified_scientific_results"]
    for task, row in zip(work["protocol"]["task_map_v741"], rows, strict=True):
        old = []
        for registration in task.get("prior_failures", []):
            number = int(registration["experiment_id"][3:7])
            recorded = next((r for r in history if r["experiment_id"] == number), None)
            if recorded is None:
                path = inputs.locate(
                    dict(
                        id=registration["experiment_id"],
                        deliverable=f"results/experiment_{number}_missing.json",
                    )
                )
                recorded = (
                    e.read(e.reference(path))
                    if path.is_file()
                    else dict(
                        honest_verdict="complete_blocked_missing_primary",
                        missing_reason="producer_primary_absent",
                    )
                )
            literal = recorded.get(
                "original_honest_verdict",
                recorded.get("honest_verdict", "complete_blocked_unrecorded_prior_verdict"),
            )
            old.append(
                dict(
                    experiment_id=number,
                    literal_verdict=literal,
                    normalized_verdict=normalise(literal),
                    normalized_failure_cause=inputs.cause(recorded),
                    failed_checks=recorded.get("checks", recorded.get("gate_check_summary", [])),
                    primary_reference=recorded.get("primary_reference"),
                )
            )
        literal = row.get("original_honest_verdict", row["honest_verdict"])
        result.append(
            dict(
                task_id=task["id"],
                prior_failures=task.get("prior_failures", []),
                actual_prior_outcomes=old,
                observed_verdict=literal,
                normalized_verdict=normalise(literal),
                normalized_failure_cause=inputs.cause(row),
                exact_verdict_repeat=any(r["literal_verdict"] == literal for r in old),
                normalized_cause_repeat=any(
                    r["normalized_failure_cause"] == inputs.cause(row) for r in old
                ),
                decision="close_qualified_null"
                if row["eligible"] and row["verdict_class"] == "null"
                else "retain_qualified_mechanics"
                if row["eligible"]
                else "retire_unchanged_failed_scope",
                hypothesis_retired=False,
                exclusion_manifest_owner="conductor",
                reopen=REOPEN,
            )
        )
    return result


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Failed owned validation disqualifies; completed external blocks stay blocked."""
    sample = work["samples"]
    sample_ok = all(r["matches_original"] for r in sample["streaming"] + sample["installed_clocks"])
    sample_ok = (
        sample_ok
        and sample["arc"]["matches_original"]
        and sample["v740"]["released_feedback"]["matches_original"]
        and sample["v740"]["retention_maximum"]["matches_original"]
        and all(
            r["matches_original"]
            for r in sample["v740"]["causal_losses"] + sample["v740"]["complete_lean_cost"]["rows"]
        )
        and sample["v740"]["triviaplus_coverage"]["matches_original"]
        and sample["v740"]["current_arc_horizons"]["matches_original"]
    )
    owned = (
        bool(receipts)
        and all("stdout_path" in r or r.get("name") in SYNTHETIC_RECEIPTS for r in receipts)
        and all(
            r["passed"] for r in receipts + work["branch_receipts"] + [work["publication_receipt"]]
        )
        and sample_ok
        and not work["precondition_failures"]
        and work["contract"]["ready"]
    )
    verdict = "blocked" if owned else "disqualified"
    prior = e.read(e.reference(e.ROOT / "results/experiment_8626_v740_capstone.json"))
    rows = deepcopy(work["slots"]) + [
        dict(
            experiment_id=8639,
            task_id=e.TASK,
            unit_id=e.TASK,
            status="completed",
            eligible=owned,
            producer_executed=True,
            disposition="capstone_reconciliation",
            verdict_class=verdict,
            honest_verdict=f"complete_{verdict}_v741_capstone",
            censored=False,
            excluded=False,
            absolute_metric=0,
            effective_independent_groups=0,
            checks=[],
            missing_reason=None,
        )
    ]
    checks = (
        work["precondition_failures"]
        + work["contract"]["gate_check_summary"]
        + [g for r in rows for g in r["checks"]]
        + [g for g in work["declared_gates"] if not g["passed"]]
    )
    if not owned:
        checks += [e.methods.gate(Path(work["raw"]), "owned_validation", True, False, e.TASK)]
    validation_path = Path(work["raw"]) / "validation.json"
    value: Json = dict(
        experiment_id=8639,
        task_id=e.TASK,
        milestone=e.MILESTONE,
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v741_capstone",
        verdict_class=verdict,
        gate_check_summary=checks,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        current_language_model_calls=0,
        historical_model_provenance=prior["historical_model_provenance"],
        target_model=None,
        small_head_numerical_inference=dict(
            performed=False,
            historical_checkpoint=work["protocol"]["head_checkpoint_reference"],
            arithmetic="Recomputed retained scalar losses and clocks; no head prediction or model load",
        ),
        rows=rows,
        intended_count=13,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["verdict_class"] == "disqualified" for r in rows),
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(task_slots=13, independent_questions=0, fixed_diagnostic_only=True),
        verifier_is_oracle=True,
        exposure_scope="Previously exposed development responses and engineering fixtures; prior exposure remains unknown",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        acceptance_gates=dict(
            owned_validation=owned,
            original_authority=work["contract"]["ready"],
            all_slots_accounted=len(rows) == 13,
            qualified_learning=False,
            streaming_coverage=False,
            packaging_noninferiority=False,
            complete_native_cost=False,
            nfr01=False,
        ),
        validation_receipts=receipts,
        validation_reference=e.reference(validation_path),
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=[r for r in receipts if not r["passed"]],
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"] + sum(r.get("duration_s", 0) for r in receipts),
        phase_spans=work["phase_spans"],
        random_seed=7418639,
        source_artifact_hashes=work["source_refs"],
        cited_upstream_artifacts=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[
            e.reference(Path(work["raw"]) / "measurement.json"),
            e.reference(validation_path),
        ],
        primitive_reference=e.reference(Path(work["raw"]) / "measurement.json"),
        publication_output=str(output),
        administrative_complete=all(r["status"] == "completed" for r in rows),
        task_count=13,
        scheduling_completion_count=sum(r["status"] == "completed" for r in rows),
        executed_producer_count=sum(r["producer_executed"] for r in rows),
        task_contract_rows=work["contract"]["rows"],
        canonical_tasks_sha256=e.methods.TASKS_PIN,
        full_task_objects=work["protocol"]["task_map_v741"],
        declared_upstream_gate_rows=work["declared_gates"],
        qualified_scientific_results=deepcopy(prior["qualified_scientific_results"])
        + [r for r in rows[:-1] if r["eligible"]],
        blocked_inputs=[r for r in rows[:-1] if r["verdict_class"] == "blocked"],
        quarantined_inputs=deepcopy(prior["quarantined_inputs"])
        + [r for r in rows[:-1] if r["verdict_class"] == "disqualified"],
        preserved_nulls=deepcopy(prior["preserved_nulls"]),
        preserved_v740_findings=prior["deterministic_claim_samples"],
        deterministic_claim_samples=sample,
        next_evidence_requirements=REOPEN,
        branch_outcomes=dict(
            learning="blocked_no_current_execution",
            streaming="disqualified_prototype_and_pregated_comparison",
            packaging="qualified_prototype_disqualified_compatibility_unmeasured_noninferiority",
            native_cost="blocked_no_current_complete_cost",
            arc="qualified_recording_null_no_policy_benefit",
            hardware="quarantined_owned_validation",
        ),
        g_gate_rows=[dict(gate=k, result=v) for k, v in work["publication"]["gates"].items()],
        paper_ready=work["publication"]["paper_ready"],
        unmet_gates=work["publication"]["unmet_gates"],
        publication_authorized=False,
        publication_gate_receipt=work["publication_receipt"],
        future_dependencies=work["protocol"]["future_dependencies"],
        methodology_note="Thirteen terminal scheduling dispositions do not imply executed science. Preserve V740 lost shards and55/190 extraction limit, V733 source, V735 cold-cost and V737 ARC nulls. V740 lean lifecycle CI95[2.6021614,2.7140500] differs from native/Python CI95[1.1799076,1.2584954]; NFR01 remains unmet. Streaming is not semantic correctness; installed mechanics are not calibration; observer support is not causal policy benefit. G1-G4 concern the existing FoVer claim only.",
    )
    value["continuation_decisions"] = decisions(work, rows, prior)
    value["retirement_recommendations"] = deepcopy(value["continuation_decisions"])
    value["branch_gate_values"] = {
        k: 0
        for k in [
            "learning_execution_ready_score",
            "conditional_exposed_learning_benefit_score",
            "streaming_coverage_ready_score",
            "packaging_noninferiority_score",
            "complete_native_cost_ready_score",
            "nfr01_score",
            "independent_generalization_score",
            "generalized_learning_benefit_score",
        ]
    }
    value.update(value["branch_gate_values"])
    value["field_principles"] = {
        k: "Bind original operands to this invocation; scheduling, qualification and scientific benefit remain separate."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    for field in value["branch_gate_values"]:
        value["field_principles"][field] = (
            "Numeric0/1 requires qualified evidence for this exact branch. Missing data do not represent an observed zero outcome."
        )
    value["reproducibility_checksum"] = e.checksum(value)
    return value


def replay(path: Path) -> None:
    """Reconstruct every claim; repaired candidate hashes cannot change original evidence."""
    value = __import__("json").loads(path.read_bytes())
    work = e.read(value["primitive_reference"])
    if any(
        "stdout_path" not in r and r.get("name") not in SYNTHETIC_RECEIPTS
        for r in value["validation_receipts"]
    ):
        raise ValueError("unbound_validation_receipt")
    if work["declared_gates"] != inputs.bindings(work["protocol"]["task_map_v741"]):
        raise ValueError("declared_gate_drift")
    with patch.multiple(previous, e=e, build=build):
        previous.replay(path)
