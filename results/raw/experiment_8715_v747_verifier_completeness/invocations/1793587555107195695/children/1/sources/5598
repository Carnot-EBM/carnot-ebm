"""REQ-REPORT-8652: administrative completion cannot promote failed science."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from carnot.reporting import v742_capstone_evidence as e
from carnot.reporting.v741_capstone_inputs import cause
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from scripts.recurring_blocker_ledger import normalise

Json = dict[str, Any]
SYNTHETIC = {
    "exact_disjoint_node_union",
    "new_owned_statement_coverage_100",
    "private_publication_readers",
    "preserved_child_coverage",
}
REOPEN = {
    "learning": "Qualify Exp8641 original158-node union and original deadline semantics with no killed child. Preserve each original coverage SQLite before combining copies. Then execute unchanged Exp8643 causal updates and Exp8644 later Brier plus simultaneous retention gates; stop if qualification fails.",
    "coverage": "Qualify Exp8642 original child coverage custody and unchanged common-domain parity. Then measure Exp8645 complete windows on original190 responses in128 groups; stop on missing shards or truncated source. Source support cannot establish semantic truth.",
    "installed": "Repair the original checkout cleanup that triggered TrackedResultWriteError in Exp8646 nodes-1; retain the distinct Exp8634 resident_protocol.py input_hash failure. Pass the original union in separate immutable/current roots. Then measure all120 Exp8647 complete cost cells; stop on any failed assertion or cleanup error.",
    "runtime": "Supply changed CUDA environment/library hashes and genuine context/copy success before a bounded mandated-Qwen canary; stop on the first runtime failure.",
    "arc": "Repair Exp8650 recording TypeError and duplicate_observation issues while retaining action parity. Bind live reset request sites to post-observation hashes and unambiguous requesters. Stop on any failed qualification. Supervisor promotion separately needs a changed causal policy and original cross-game outcome support.",
    "hardware": "Supply qualified complete disjoint host/lifecycle/transport costs plus compatible physical kernel, source/build/bitstream hashes, executed parity and host/device clocks. No cost-removal ceiling or hardware speed claim before those operands exist.",
    "generalization": "Use unexposed independent questions and an oracle-distinct matched control with original confidence gates; stop if exposure or verifier circularity remains.",
    "native": "Change the registered native mechanism and measure matched complete native/Python lower95>=10. Packaging noninferiority and lean lifecycle gains cannot satisfy this gate.",
}


def decisions(work: Json, rows: list[Json], prior: Json) -> list[Json]:
    """Compare literal registrations with actual historical outcomes before recommending closure."""
    result = []
    history = prior["rows"] + prior["quarantined_inputs"] + prior["qualified_scientific_results"]
    for task, row in zip(work["protocol"]["task_map_v742"], rows, strict=True):
        previous = []
        for registration in task.get("prior_failures", []):
            number = int(registration["experiment_id"][3:7])
            recorded = next((r for r in history if r["experiment_id"] == number), None)
            if recorded is None:
                ref = work["protocol"]["original_producers"].get(str(number))
                recorded = (
                    e.read(ref)
                    if ref
                    else dict(
                        honest_verdict="complete_blocked_missing_historical_producer",
                        missing_reason="producer_primary_absent",
                    )
                )
            literal = recorded.get("original_honest_verdict", recorded["honest_verdict"])
            ref = recorded.get("primary_reference")
            failed_children = (
                e.failure_evidence(e.read(ref))
                if ref and ref.get("sha256")
                else e.failure_evidence(recorded)
            )
            for failure in failed_children:
                failure["logs"] = [
                    e.freeze(Path(log["path"]), Path(work["raw"]) / "prior_failed_children")
                    for log in failure["logs"]
                ]
            previous.append(
                dict(
                    experiment_id=number,
                    registered_verdict=registration["verdict"],
                    literal_verdict=literal,
                    normalized_verdict=normalise(literal),
                    normalized_failure_cause=cause(recorded),
                    failed_checks=recorded.get("checks", recorded.get("gate_check_summary", [])),
                    failed_owned_checks=failed_children,
                    underlying_cause_shapes=sorted(
                        {r["normalized_underlying_cause"] for r in failed_children}
                    ),
                )
            )
        literal = row.get("original_honest_verdict", row["honest_verdict"])
        current_shapes = sorted(
            {r["normalized_underlying_cause"] for r in row.get("failed_owned_checks", [])}
        )
        decision = (
            "close_qualified_null_protocol"
            if row["eligible"] and row["verdict_class"] == "null"
            else "retain_qualified_mechanics"
            if row["eligible"]
            else "require_changed_prerequisite"
        )
        result.append(
            dict(
                task_id=task["id"],
                observed_verdict=literal,
                actual_prior_outcomes=previous,
                normalized_failure_cause=cause(row),
                underlying_cause_shapes=current_shapes,
                exact_verdict_repeat=any(r["literal_verdict"] == literal for r in previous),
                normalized_cause_repeat=any(
                    r["underlying_cause_shapes"] == current_shapes
                    if current_shapes
                    else r["normalized_failure_cause"] == cause(row)
                    for r in previous
                ),
                decision=decision,
                hypothesis_retired=decision == "close_qualified_null_protocol",
                conductor_owner_action="recommendation_only",
                reopening_operands=REOPEN,
            )
        )
    return result


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Failed owned checks disqualify; external unchanged incompleteness stays blocked."""
    sample = work["samples"]
    samples_ok = all(
        r["matches_original"]
        for r in sample["causal_losses"]
        + sample["released_feedback"]
        + sample["streaming_windows"]
        + sample["installed_clocks"]
    )
    samples_ok &= all(
        sample[k]["matches_original"]
        for k in ["retention", "arc_reset_transfer", "reset_provenance"]
    )
    owned = (
        bool(receipts)
        and all(
            r["passed"] and ("stdout_path" in r or r.get("name") in SYNTHETIC)
            for r in receipts + work["branch_receipts"] + [work["publication_receipt"]]
        )
        and samples_ok
        and work["contract"]["ready"]
        and not work["precondition_failures"]
    )
    verdict = "blocked" if owned else "disqualified"
    rows = deepcopy(work["slots"]) + [
        dict(
            experiment_id=8652,
            task_id=e.TASK,
            unit_id=e.TASK,
            status="completed",
            eligible=owned,
            producer_executed=True,
            disposition="capstone_reconciliation",
            verdict_class=verdict,
            honest_verdict=f"complete_{verdict}_v742_capstone",
            censored=False,
            excluded=False,
            absolute_metric=0,
            effective_independent_groups=0,
            checks=[],
            missing_reason=None,
        )
    ]
    prior = e.read(e.reference(e.ROOT / "results/experiment_8639_v741_capstone.json"))
    checks = (
        work["precondition_failures"]
        + work["contract"]["gate_check_summary"]
        + [g for r in rows for g in r["checks"]]
        + [g for g in work["declared_gates"] if not g["passed"]]
    )
    if not owned:
        checks.append(e.methods.gate(Path(work["raw"]), "owned_validation", True, False, e.TASK))
    validation = (
        Path(work["raw"])
        / "validation_snapshots"
        / (e.checksum(dict(receipts=receipts)).split(":")[-1] + ".json")
    )
    e.atomic_json(validation, dict(receipts=receipts))
    value: Json = dict(
        experiment_id=8652,
        task_id=e.TASK,
        milestone=e.MILESTONE,
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v742_capstone",
        verdict_class=verdict,
        gate_check_summary=checks,
        inference_substrate="immutable evidence reduction and frozen small energy head inference",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_language_model_calls=0,
        historical_model_provenance=prior["historical_model_provenance"],
        target_model=sample["numerical_head"],
        small_head_numerical_inference=sample["numerical_head"],
        rows=rows,
        intended_count=13,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["verdict_class"] == "disqualified" for r in rows),
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            task_slots=13, independent_questions=0, repetitions_are_not_independent=True
        ),
        verifier_is_oracle=True,
        exposure_scope="Previously exposed development responses and engineering fixtures; historical exposure remains unknown",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        acceptance_gates=dict(
            owned_reconciliation=owned,
            original_authority=work["contract"]["ready"],
            all_slots_accounted=len(rows) == 13,
        ),
        validation_receipts=receipts,
        validation_reference=e.reference(validation),
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=[r for r in receipts if not r["passed"]],
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=20261018,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[e.reference(Path(work["raw"]) / "measurement.json")],
        primitive_reference=e.reference(Path(work["raw"]) / "measurement.json"),
        cited_upstream_artifacts=work["source_refs"],
        administrative_complete=owned and len(rows) == 13,
        executed_producer_count=sum(r["producer_executed"] for r in rows[:-1]),
        task_dispositions=rows,
        gate_edge_rows=work["declared_gates"],
        full_task_objects=work["protocol"]["task_map_v742"],
        canonical_tasks_sha256="sha256:" + e.methods.TASKS_PIN,
        task_contract_rows=work["contract"]["rows"],
        deterministic_claim_samples=sample,
        publication_gate_receipt=work["publication_receipt"],
        paper_ready=work["publication"]["paper_ready"],
        unmet_gates=work["publication"]["unmet_gates"],
        publication_output=str(output),
        publication_authorized=False,
        future_dependencies=work["protocol"]["future_dependencies"],
        preserved_nulls=prior["preserved_nulls"],
        preserved_v737_supervisor_null=work["protocol"]["closed_comparisons"],
        historical_quarantines=work["protocol"]["historical_quarantines"],
        quarantined_inputs=prior["quarantined_inputs"]
        + [
            r
            for r in rows[:-1]
            if r["producer_executed"]
            and (r["verdict_class"] == "disqualified" or r.get("upstream_flagged_adversarial"))
        ],
        qualified_science_rows=[
            r for r in rows[:-1] if r["eligible"] and r["verdict_class"] == "null"
        ],
        reconciliation_success=owned,
    )
    value.update({k: work["publication"]["gates"][k] for k in ["G1", "G2", "G3", "G4"]})
    value["continuation_decisions"] = decisions(work, rows, prior)
    value["literal_prior_verdicts"] = [
        dict(task_id=r["task_id"], priors=r["actual_prior_outcomes"])
        for r in value["continuation_decisions"]
    ]
    value["normalized_failure_causes"] = [
        dict(task_id=r["task_id"], cause=r["normalized_failure_cause"])
        for r in value["continuation_decisions"]
    ]
    value["retirement_recommendations"] = deepcopy(value["continuation_decisions"])
    value["reopening_conditions"] = REOPEN
    value["prd_gap_decisions"] = [
        dict(gap=gap, decision="remain_open", reopening_operands=REOPEN[branch])
        for gap, branch in [
            ("FR-06: qualified continual updates", "learning"),
            ("FR-11: later benefit and simultaneous retention", "learning"),
            ("FR-12: complete source support versus semantic verification", "coverage"),
            ("FR-08: installed interoperability and complete cost", "installed"),
            ("NFR-01", "native"),
            ("GAP-ORACLE-DISTINCT", "generalization"),
            ("GAP-DETECTOR-AUROC-4208", "generalization"),
            ("FR-11: independent generalization", "generalization"),
            ("FR-12: live supervisor causal benefit", "arc"),
            ("FR-04/NFR-01: compatible physical execution", "hardware"),
        ]
    ]
    gates = {
        k: 0
        for k in [
            "learning_execution_ready_score",
            "learning_rows_ready_score",
            "learning_cost_ready_score",
            "learning_benefit_ready_score",
            "conditional_exposed_learning_benefit_score",
            "streaming_extractor_ready_score",
            "streaming_coverage_ready_score",
            "source_comparison_ready_score",
            "installed_measurement_ready_score",
            "installed_cost_ready_score",
            "packaging_noninferiority_score",
            "complete_native_cost_ready_score",
            "nfr01_score",
            "reset_transfer_rows_ready_score",
            "promotion_ready_score",
            "physical_hardware_ready_score",
            "independent_generalization_score",
            "generalized_learning_benefit_score",
        ]
    }
    gates.update(
        capstone_ready_score=int(owned),
        reset_provenance_ready_score=int(
            any(r["experiment_id"] == 8649 and r["eligible"] for r in rows)
        ),
    )
    value["branch_gate_values"] = gates
    value.update(gates)
    value["field_principles"] = {
        k: "Bind exact original evidence to this invocation. Qualification, science, independence and publication authority are separate."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = e.checksum(value)
    return value


def replay(path: Path) -> None:
    """Rebuild claims and authenticate original logs; repaired hashes confer no authority."""
    value = json.loads(path.read_bytes())
    work = e.read(value["primitive_reference"])
    if (
        e.checksum(value) != value["reproducibility_checksum"]
        or work["protocol"] != e.methods.protocol()
        or work["contract"] != e.methods.authority(e.ROOT)
        or work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9
        or not 0 <= work["duration_s"] < 3600
    ):
        raise ValueError("invocation_drift")
    for ref in work["source_refs"] + work["code_refs"]:
        if ref.get("exists", True):
            for source in {ref["path"], ref.get("snapshot_path", ref["path"])}:
                if e.methods.sha256_file(Path(source)) != ref["sha256"]:
                    raise ValueError("source_drift")
        elif Path(ref["path"]).exists():
            raise ValueError("missing_input_changed")
    if e.read(value["validation_reference"])["receipts"] != value["validation_receipts"]:
        raise ValueError("validation_receipts_drift")
    with TemporaryDirectory(prefix="exp8652-replay-", dir="/var/tmp") as directory:
        recomputed, _ = e.sample(Path(directory))
    if (
        work["samples"] != recomputed
        or work["slots"] != e.slots(Path(work["raw"]) / "primaries", work["conductor"])
        or work["declared_gates"] != e.bindings()
    ):
        raise ValueError("reduction_drift")
    if work["publication"] != json.loads(
        Path(work["publication_receipt"]["stdout_path"]).read_bytes()
    ):
        raise ValueError("publication_drift")
    for receipt in (
        value["validation_receipts"] + work["branch_receipts"] + [work["publication_receipt"]]
    ):
        if "stdout_path" in receipt:
            original = Path(receipt["stdout_path"]).with_name(receipt["name"] + ".receipt.json")
            if json.loads(original.read_bytes()) != receipt:
                raise ValueError("receipt_drift")
            for stream in ["stdout", "stderr"]:
                if (
                    e.methods.sha256_file(Path(receipt[stream + "_path"]))
                    != receipt[stream + "_sha256"]
                ):
                    raise ValueError("receipt_stream_drift")
        elif receipt.get("name") not in SYNTHETIC:
            raise ValueError("unbound_validation_receipt")
    if value != build(work, value["validation_receipts"], Path(value["publication_output"])):
        raise ValueError("claim_drift")
