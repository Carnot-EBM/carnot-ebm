"""REQ-REPORT-8614-REPLAY: original evidence controls claims and retirement.

Administration can finish while scientific prerequisites remain blocked. Final
claims are reconstructed so a repaired checksum cannot manufacture evidence.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from carnot.reporting import v739_capstone_evidence as e
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS

Json = dict[str, Any]


ARCHIVE_AUTHORITY = dict(
    completed_archive_ends_at="2026.10.737",
    current_evidence="V738 terminal artifacts and conductor receipts",
    successful_studies_not_implied=True,
    v738_slots=11,
)


def decisions(work: Json, prior: Json) -> list[Json]:
    """Compare registrations against authenticated prior outcomes, including absences."""
    history = prior["rows"] + prior["quarantined_inputs"] + prior["qualified_scientific_results"]
    result = []
    for task, row in zip(work["protocol"]["task_map_v739"][:-1], work["slots"], strict=True):
        actual = []
        for registered in task.get("prior_failures", []):
            number = int(registered["experiment_id"][3:7])
            recorded = next((r for r in history if r["experiment_id"] == number), None)
            ref = work["protocol"]["original_producers"].get(str(number))
            if recorded is None and ref is not None:
                recorded = e.read(ref)
            original = recorded or {}
            actual.append(
                dict(
                    experiment_id=number,
                    primary_reference=original.get("primary_reference", ref),
                    honest_verdict=original.get(
                        "original_honest_verdict", original.get("honest_verdict")
                    ),
                    disposition=original.get("disposition"),
                    failed_checks=original.get("checks", original.get("gate_check_summary", [])),
                    missing_reason=original.get("missing_reason"),
                )
            )
        result.append(
            dict(
                task_id=task["id"],
                prior_failures=task.get("prior_failures", []),
                actual_prior_outcomes=actual,
                observed_verdict=row["honest_verdict"],
                decision="close_qualified_null"
                if row["verdict_class"] == "null"
                else "retire_exact_unchanged_scope",
                reopen="Changed bounded prerequisite with authenticated authority and original required assertions; no unchanged rerun.",
            )
        )
    return result


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Keep zero scientific credit while separately reporting complete dispositions."""
    sample_ok = all(
        r["matches_original"]
        for r in work["samples"]["causal_losses"] + work["samples"]["lean_prototype_sessions"]
    ) and all(
        work["samples"][k]["matches_original"]
        for k in ["retention_maximum", "arc_horizons", "historical_resident_total"]
    )
    owned = (
        bool(receipts)
        and all(
            r["passed"] for r in receipts + work["branch_receipts"] + [work["publication_receipt"]]
        )
        and sample_ok
        and not work["precondition_failures"]
    )
    verdict = "blocked" if owned else "disqualified"
    prior = e.read(work["protocol"]["original_producers"]["8601"])
    rows = deepcopy(work["slots"]) + [
        dict(
            experiment_id=8614,
            task_id=e.TASK,
            unit_id=e.TASK,
            status="completed",
            eligible=owned,
            producer_executed=True,
            disposition="capstone_reconciliation",
            verdict_class=verdict,
            honest_verdict=f"complete_{verdict}_v739_capstone",
            censored=False,
            excluded=False,
            missing_reason=None,
            absolute_metric=1,
            effective_independent_groups=0,
        )
    ]
    checks = (
        work["contract"]["gate_check_summary"]
        + work["precondition_failures"]
        + [g for r in work["slots"] for g in r["checks"]]
    )
    checks += [
        e.gate(Path(str(r.get("stdout_path", work["raw"]))), r["name"], True, False)
        for r in receipts
        if not r["passed"]
    ]
    if not receipts or not sample_ok:
        checks.append(e.gate(Path(work["raw"]), "owned_receipts_and_samples", True, False))
    validation_path = (
        Path(work["raw"])
        / "validation_snapshots"
        / (e.checksum(dict(receipts=receipts)).split(":")[-1] + ".json")
    )
    e.atomic_json(validation_path, dict(receipts=receipts))
    value: Json = dict(
        experiment_id=8614,
        task_id=e.TASK,
        milestone=e.MILESTONE,
        run_date=work["run_date"],
        honest_verdict=f"complete_{verdict}_v739_capstone",
        verdict_class=verdict,
        gate_check_summary=checks,
        inference_substrate="immutable evidence aggregation and numerical arithmetic replay",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=prior["historical_model_provenance"],
        target_model=dict(
            name="Exp8431 spline9 seed17 calibrated energy head",
            checkpoint=e.read(
                e.reference(e.ROOT / "results/experiment_8608_v739_lean_resident_prototype.json")
            )["target_model"]["checkpoint"],
            checkpoint_head="heads.spline9[0]",
            parameter_count=109,
            evaluated_this_invocation=False,
            current_numerical_operations="historical probability loss arithmetic only",
        ),
        rows=rows,
        intended_count=13,
        completed_count=sum(r["status"] == "completed" for r in rows),
        failed_count=sum(r["verdict_class"] == "disqualified" for r in rows),
        censored_count=sum(r["censored"] for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            task_slots=13,
            independent_questions=0,
            fixed_diagnostic_only=True,
            repetitions_are_not_independent=True,
        ),
        verifier_is_oracle=True,
        exposure_scope="Historical exposed questions and private operation fixtures; no new human labels",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        acceptance_gates=dict(
            owned_validation=owned,
            all_slots_accounted=len(rows) == 13,
            original_authority=work["contract"]["ready"],
            qualified_learning=False,
            qualified_lean_cost=False,
        ),
        validation_receipts=receipts,
        validation_reference=e.reference(validation_path),
        terminal_validation_sidecar_path=str(Path(work["raw"]) / "terminal_validation.json"),
        adversarial_findings=[r for r in receipts if not r["passed"]],
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"] + sum(r.get("duration_s", 0) for r in receipts),
        phase_spans=work["phase_spans"],
        random_seed=7398614,
        source_artifact_hashes=work["source_refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[
            e.reference(Path(work["raw"]) / "measurement.json"),
            e.reference(validation_path),
        ],
        primitive_reference=e.reference(Path(work["raw"]) / "measurement.json"),
        publication_output=str(output),
        administrative_complete=all(r["status"] == "completed" for r in rows),
        task_count=13,
        task_contract_rows=work["contract"]["rows"],
        canonical_tasks_sha256=e.methods.TASKS_PIN,
        qualified_scientific_results=deepcopy(prior["qualified_scientific_results"]),
        quarantined_inputs=deepcopy(prior["quarantined_inputs"])
        + [r for r in rows if r["verdict_class"] == "disqualified" and r["experiment_id"] != 8614],
        blocked_inputs=[
            r for r in rows if r["verdict_class"] == "blocked" and r["experiment_id"] != 8614
        ],
        preserved_nulls=deepcopy(prior["preserved_nulls"]),
        continuation_decisions=decisions(work, prior),
        retirement_recommendations=decisions(work, prior),
        next_evidence_requirements=dict(
            deadline="Retire unchanged line94 qualification; a separately diagnosed changed coverage mechanism must cover the second deadline-clock read under every original assertion before fresh learning.",
            authority="Repair independently printed V739 full JSON contract with thirteen unchanged objects and original task digest; qualification is not benefit.",
            labels="Authenticate actual Trivia+ release, complete source and siblings, rights and confidence before the registered extraction study. Source faithfulness does not establish factual correctness. Unavailable MedExpert labels remain unavailable.",
            lean="Qualify complete original parity and authority before one new inclusive lean cost registration; prototype operation spans omit lifecycle costs.",
            arc="Changed causal supervisor with resolved cross-game support under original three-game and five-observation cell floors; observer receipts do not authorize unchanged panels.",
        ),
        g_gate_rows=[dict(gate=k, result=v) for k, v in work["publication"]["gates"].items()],
        paper_ready=work["publication"]["paper_ready"],
        unmet_gates=work["publication"]["unmet_gates"],
        publication_authorized=False,
        publication_gate_receipt=work["publication_receipt"],
        conditional_exposed_benefits=deepcopy(prior["conditional_exposed_benefits"]),
        native_lifecycle_findings=deepcopy(
            prior["continuation_decisions"]["resident_native_boundary"]
        ),
        qualification_failure_evidence=work["samples"]["deadline_failure"],
        deterministic_claim_samples=work["samples"],
        archive_authority=ARCHIVE_AUTHORITY,
        inherited_archive_authority=work["protocol"]["archive_authority"],
        future_dependencies=work["protocol"]["future_dependencies"],
        methodology_note="Thirteen scheduling dispositions, not thirteen successful studies. Retain V733 source null, V735 cold-cost null, V737 ARC null, original learning disqualification and V738 native/lifecycle findings. G1-G4 describe the older reproduced FoVer headline; paper readiness grants no publication authority. New numerical operations inspect authenticated operands without a model load.",
    )
    value["field_principles"] = {
        k: "Bind this invocation to original evidence and exact gates; administrative closure and diagnostic arithmetic provide zero generalization credit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = e.checksum(value)
    return value


def replay(path: Path) -> None:
    """Recompute every claim and verify receipts instead of trusting repaired hashes."""
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
            checked_path = Path(ref.get("snapshot_path", ref["path"]))
            if e.methods.sha256_file(checked_path) != ref["sha256"]:
                raise ValueError("source_drift")
    if e.read(value["validation_reference"])["receipts"] != value["validation_receipts"]:
        raise ValueError("validation_receipts_drift")
    with TemporaryDirectory(prefix="exp8614-replay-", dir="/var/tmp") as directory:
        recomputed, _ = e.sample(Path(directory))
    if (
        work["slots"] != e.slots(Path(work["raw"]) / "primaries", work["conductor"])
        or work["samples"] != recomputed
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
    if value != build(work, value["validation_receipts"], Path(value["publication_output"])):
        raise ValueError("claim_drift")
