"""REQ-REPORT-8578: scheduling completion cannot erase nulls or historical quarantine."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from carnot.reporting import v736_capstone_evidence as e
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v736_historical_inputs import checked

Json = dict[str, Any]
DECISIONS = dict(
    fixed_brier_learning=dict(
        decision="close_unchanged_retry_retain_unmeasured_hypothesis",
        qualified_benefit=None,
        reopen="Diagnose and bind changed validator revision hashes and repair complete authority before the same fixed delayed-feedback Brier protocol; Exp8569/8570 never measured benefit.",
    ),
    resident_native_boundary=dict(
        decision="retain_qualified_resident_parity_close_unchanged_cost_retry",
        qualified_benefit=None,
        reopen="Repair the diagnosed stale fourteen-object authority, then measure Exp8573 complete resident session costs once; parity and lifecycle clocks grant no serving gain and never revise Exp8559 cold null.",
    ),
    fixed_arc_panel=dict(
        decision="close_unchanged_panel_retry_retain_unmeasured_hypothesis",
        qualified_benefit=None,
        reopen="Repair diagnosed Exp8561 publication closure at line282 and qualify actual closure under corrected authority before fixed mask panel measurement; a version increment alone cannot reopen.",
    ),
)


def build(work: Json, validation: Json, raw: Path, output: Path) -> Json:
    """Only required owned receipts qualify this aggregation; missing science remains blocked."""
    receipts = e.read(validation)["receipts"]
    planned = e.read(work["execution_manifest_reference"])["commands"]
    matched = len(receipts) >= len(planned) and all(
        r["argv"] == p["argv"] and r["name"] == p["name"] and r["expected_exit"] == p["expected"]
        for p, r in zip(planned, receipts)
    )
    sample_ok = all(work["samples"][k]["matches_original"] for k in ["learning", "arc"]) and all(
        r["matches_original"] for r in work["samples"]["resident_clock_sample"]
    )
    owned = (
        bool(receipts)
        and matched
        and sample_ok
        and all(r["passed"] for r in [*receipts, *work["branch_receipts"]])
        and work["publication_receipt"]["passed"]
    )
    verdict = "blocked" if owned else "disqualified"
    prior = e.read(work["protocol"]["original_producers"]["8564"])
    originals = {n: e.read(work["protocol"]["original_producers"][str(n)]) for n in [8553, 8559]}
    preserved = deepcopy(prior["preserved_nulls"]) + [
        dict(
            experiment_id=8559,
            verdict_class="null",
            eligible=True,
            primary_reference=work["protocol"]["original_producers"]["8559"],
            complete_throughput_ratio=originals[8559]["complete_throughput_ratio"],
            throughput_ratio_ci95=originals[8559]["throughput_ratio_ci95"],
        )
    ]
    checks = work["contract"]["gate_check_summary"] + work["precondition_failures"]
    checks += [g for r in work["slots"] for g in [*r["checks"], *r.get("source_gates", [])]]
    checks += [
        e.gate(Path(r["stdout_path"]), r["name"], True, False) for r in receipts if not r["passed"]
    ]
    if not matched or not sample_ok:
        checks.append(e.gate(Path(validation["path"]), "owned_manifest_and_samples", True, False))
    own = dict(
        experiment_id=8578,
        task_id=e.TASK,
        unit_id=e.TASK,
        status="completed",
        censored=False,
        excluded=False,
        absolute_metric=1,
        effective_independent_groups=0,
        eligible=owned,
        producer_executed=True,
        disposition="capstone_reconciliation",
        honest_verdict=f"complete_{verdict}_v736_capstone",
        verdict_class=verdict,
        missing_reason=None,
        terminal_seal="exact_final_primary_bytes_in_external_bound_sidecar",
    )
    rows = work["slots"] + [own]
    value: Json = dict(
        experiment_id=8578,
        task_id=e.TASK,
        milestone=e.MILESTONE,
        run_date="20261016",
        honest_verdict=own["honest_verdict"],
        verdict_class=verdict,
        gate_check_summary=checks,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=work["history"]["historical_model_provenance"],
        rows=rows,
        intended_count=14,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["verdict_class"] == "disqualified" for r in rows),
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            intended_scheduling_slots=14,
            independent_questions=0,
            diagnostic_only=True,
            repeats_multiply_evidence=False,
        ),
        verifier_is_oracle=True,
        exposure_scope="Exposed original questions and public ARC diagnostic; no independent labels or learning credit",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        acceptance_gates=dict(
            owned_validation=owned,
            all_slots_accounted=len(rows) == 14,
            full_authority=work["contract"]["ready"],
            qualified_learning_benefit=False,
            qualified_resident_cost=False,
            qualified_arc_panel=False,
        ),
        validation_receipts=receipts,
        validation_reference=validation,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=[f for r in receipts for f in r.get("adversarial_findings", [])],
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"] + sum(r["duration_s"] for r in receipts),
        phase_spans=work["phase_spans"],
        random_seed=7368578,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[e.reference(raw / "measurement.json"), validation],
        cited_upstream_artifacts=list(work["protocol"]["original_producers"].values()),
        primitive_reference=e.reference(raw / "measurement.json"),
        publication_output=str(output),
        milestone_status="fourteen_scheduling_slots_accounted_scientific_milestone_blocked",
        task_count=14,
        administrative_complete=all(r["status"] == "completed" for r in rows),
        task_contract_rows=work["contract"]["rows"],
        canonical_tasks_sha256=e.methods.TASKS_PIN,
        qualified_scientific_results=preserved
        + [r for r in work["slots"] if r["eligible"] and r["verdict_class"] == "null"],
        blocked_inputs=[r for r in work["slots"] if r["verdict_class"] == "blocked"],
        quarantined_inputs=prior["quarantined_inputs"]
        + [r for r in work["slots"] if r["verdict_class"] == "disqualified"],
        preserved_nulls=preserved,
        hallmark_label_rejection=dict(
            source=work["protocol"]["original_producers"]["8553"],
            admission_decision=originals[8553]["admission_decision"],
        ),
        conditional_exposed_benefits=dict(
            status="no_qualified_learning_or_arc_benefit_no_resident_speed_measurement",
            generalization_credit=0,
        ),
        publication_gate_receipt=work["publication_receipt"],
        g_gate_rows=[dict(gate=k, result=v) for k, v in work["publication"]["gates"].items()],
        paper_ready=work["publication"]["paper_ready"],
        unmet_gates=work["publication"]["unmet_gates"],
        publication_authorized=False,
        retirement_recommendations=[dict(scope=k, **v) for k, v in DECISIONS.items()],
        continuation_decisions=DECISIONS,
        next_evidence_requirements=dict(
            fresh_data="New disjoint independently adjudicated human labels; HALLMARK constructed labels and exposed FaithBench cannot substitute.",
            causal_cuda="A diagnosed changed runtime mechanism, authenticated causal repair, and bounded context/copy evidence; no model call in this invocation.",
            physical_operations="Compatible installed operations, full transport clocks, physical source/cable/IDCODE/flash/smoke evidence; CPU ceilings grant no device performance.",
        ),
        architecture_boundary_notes=prior["architecture_boundary_notes"],
        deterministic_claim_samples=work["samples"],
        archive_authority=work["protocol"]["archive_authority"],
        methodology_note="Immutable original V735/V736 producer and conductor evidence; thirteen predecessor slots plus this administrative capstone. Small original learning, resident-clock and ARC arithmetic is diagnostic only. No model, training, hardware benchmark or complete historical experiment rerun. Count correction preserves failed producers; missing cost and learning remain unmeasured; cold cost null remains unchanged. G1-G4 confer no external publication permission.",
    )
    value["field_principles"] = {
        k: "Bind original invocation operands and qualification; missing differs from zero and repeated diagnostics add no independent scientific evidence."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = e.checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild original claims and rows so repaired envelope hashes cannot invent evidence."""
    try:
        value = json.loads(path.read_bytes())
        work = e.read(value["primitive_reference"])
        if (
            e.checksum(value) != value["reproducibility_checksum"]
            or e.methods.protocol() != work["protocol"]
        ):
            return False
        for ref in work["source_refs"] + work["code_refs"]:
            if ref.get("exists", True) and not checked(ref):
                return False
        if e.methods.authority(Path(work["root"])) != work["contract"]:
            return False
        if (
            json.loads(Path(work["publication_receipt"]["stdout_path"]).read_bytes())
            != work["publication"]
        ):
            return False
        if value != build(
            work,
            value["validation_reference"],
            Path(value["primitive_reference"]["path"]).parent,
            Path(value["publication_output"]),
        ):
            return False
        tasks = work["protocol"]["task_map_v736"]
        log = next(
            r
            for r in work["source_refs"]
            if r["path"].endswith("ops/conductor-log.md") and "snapshot_path" in r
        )
        lines = Path(log["snapshot_path"]).read_text().splitlines()
        for row, task in zip(work["slots"], tasks[:-1], strict=True):
            ref = row["primary_reference"]
            summary = e.outcome(task, ref, Path(ref["path"]).parent) if ref["exists"] else None
            cascade = next(
                (
                    s
                    for s in lines
                    if "GATE_BLOCK" in s and s.split("|")[2].strip() == task["title"][:50].strip()
                ),
                None,
            )
            if row != e.old.slot(task, ref, cascade, summary):
                return False
        with TemporaryDirectory(prefix="exp8578-replay-", dir="/var/tmp") as directory:
            diagnostic, receipt = e.samples(Path(directory))
        if not receipt["passed"] or work["samples"] != diagnostic:
            return False
        for receipt in [
            *work["branch_receipts"],
            *value["validation_receipts"],
            work["publication_receipt"],
        ]:
            original = Path(receipt["stdout_path"]).with_name(receipt["name"] + ".receipt.json")
            if json.loads(original.read_bytes()) != receipt:
                return False
            for stream in ["stdout", "stderr"]:
                if not checked(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                ):
                    return False
        return bool(
            value
            == build(
                work,
                value["validation_reference"],
                Path(value["primitive_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def consumer(output: Path) -> Json:
    """Exercise both unchanged downstream readers against exactly the published primary."""
    value = json.loads(output.read_bytes())
    return dict(
        reader_receipt(
            e.TASK,
            output.parent,
            field="required_checks_passed",
            expected=value["required_checks_passed"],
        )
    )
