"""REQ-VERIFY-8681: reconstruct fresh claims without historical report wrappers."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone, UTC
import json
from pathlib import Path
from typing import Any

from carnot.reporting import v745_contract_methods as e
from carnot.reporting import v744_contract_methods as original
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.primary_publication import reader_receipt

Json = dict[str, Any]


def build(work: Json, validation: Json, raw: Path, output: Path) -> Json:
    """Copy nested operands once so normalization cannot mutate the sealed measurement."""
    w = deepcopy(work)
    receipts = e.read(validation)["receipts"]
    plan: Json = (
        e.read(w["execution_manifest_reference"])
        if "execution_manifest_reference" in w
        else dict(commands=[], private_control=True)
    )
    matched = len(receipts) >= len(plan["commands"]) and all(
        r["name"] == s["name"] and r["argv"] == s["argv"] and r["expected_exit"] == s["expected"]
        for s, r in zip(plan["commands"], receipts)
    )
    controls = all(r["passed"] for r in w["authority_mutation_rows"])
    owned = (
        bool(receipts)
        and all(
            r["passed"]
            and r.get("normal_exit", True)
            and not r.get("timed_out", False)
            and r.get("actual_exit", r.get("exit_code", 0)) == r.get("expected_exit", 0)
            for r in receipts
        )
        and matched
        and controls
    )
    coverage = any(r["name"] == "coverage_report" and r["passed"] for r in receipts)
    qualified = bool(owned and coverage and not plan["private_control"])
    methods = bool(qualified and w["contract"]["ready"] and not w["precondition_failures"])
    verdict = "disqualified" if not owned else "circular_positive" if methods else "blocked"
    failures = w["contract"]["gate_check_summary"] + w["precondition_failures"]
    failures += [
        e.gate(Path(validation["path"]), r["name"], True, False, e.TASK)
        for r in receipts
        if not r["passed"]
    ]
    if not qualified:
        failures.append(
            e.gate(Path(validation["path"]), "owned_complete_validation", True, qualified, e.TASK)
        )
    for row in failures:
        row.setdefault(
            "operand_state",
            e.gate(Path(row["path"]), row["field"], row["expected"], row["observed"])[
                "operand_state"
            ],
        )
    retained = [
        ref
        for r in receipts
        if r["name"] == "original_coverage_retention"
        for ref in r["original_references"]
    ]
    rows = w["contract"]["rows"]
    immutable = []
    for receipt in receipts:
        if "stdout_path" in receipt:
            for line in Path(receipt["stdout_path"]).read_text().splitlines():
                if line.startswith("IMMUTABLE_8681 "):
                    immutable.append(json.loads(line.split(" ", 1)[1]))
    scores = {
        "methods_ready_score": int(methods),
        "runtime_repair_ready_score": int(bool(w["runtime"]["ready"])),
    }
    for field in [
        "learning_operands_ready_score",
        "extraction_inputs_ready_score",
        "deployment_inputs_ready_score",
    ]:
        scores[field] = int(w["history"][field])
    start = w["wall_ns"] / 1e9
    end = start + w["duration_s"]
    value: Json = dict(
        experiment_id=8681,
        task_id=e.TASK,
        milestone=e.MILESTONE,
        run_date=w["run_date"],
        honest_verdict=f"complete_{verdict}_v745_contract_methods",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=w["history"]["historical_model_provenance"],
        target_model=dict(
            name="nine-input 109-coefficient spline head",
            checkpoint=w["protocol"]["head_checkpoint_reference"],
            evaluated_this_invocation=True,
        ),
        substrate_by_phase=dict(
            readers="aggregation_from_upstream_artifacts",
            constructed_checks="deterministic_verifier",
            numerical_head="verifier_ensemble_against_cached_candidates",
        ),
        rows=rows,
        contract_rows=rows,
        intended_count=14,
        completed_count=sum(r["matched"] for r in rows),
        failed_count=0,
        censored_count=sum(not r["matched"] for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            task_slots=14, independent_questions=0, repeats_multiply_evidence=False
        ),
        verifier_is_oracle=True,
        exposure_scope="Constructed administrative controls; original exposed learning and constructed relational worlds.",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        acceptance_gates=dict(
            owned_validation=owned,
            complete_fourteen_objects=w["contract"]["ready"],
            newly_owned_statement_coverage=coverage,
            authority_mutations=controls,
            sealed_methods=methods,
        ),
        validation_receipts=receipts,
        validation_reference=validation,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=[
            finding for r in receipts for finding in r.get("adversarial_findings", [])
        ],
        preconditions_checked=w["preconditions"],
        duration_s=w["duration_s"],
        started_at_utc=datetime.fromtimestamp(start, UTC).isoformat(),
        ended_at_utc=datetime.fromtimestamp(end, UTC).isoformat(),
        phase_spans=w["phase_spans"],
        random_seed=7458681,
        source_artifact_hashes=w["source_refs"] + w["history"]["references"],
        code_config_hashes=w["code_refs"],
        primitive_reference=e.reference(raw / "measurement.json"),
        measurement_seal_reference=e.reference(raw / "measurement_seal.json"),
        invocation_id=raw.name,
        publication_output=str(output.absolute()),
        raw_shard_hashes=[
            e.reference(raw / "measurement.json"),
            e.reference(raw / "measurement_seal.json"),
            validation,
        ]
        + retained
        + [r["junit_reference"] for r in receipts if "junit_reference" in r]
        + w["methods_manifest_references"],
        cited_upstream_artifacts=[
            dict(r, fields_imported=["original scientific operands and declared producer custody"])
            for r in w["source_refs"]
        ],
        branch_gate_values=scores,
        **scores,
        immutable_build_rows=immutable,
        retained_shard_rows=[dict(r, retained=True) for r in retained],
        authority_mutation_rows=w["authority_mutation_rows"],
        methods_manifest=w["methods_manifest"],
        methods_manifest_references=w["methods_manifest_references"],
        objective_updates=w["protocol"]["objective_updates"],
        objective_audit=w["protocol"]["objective_audit"],
        cohort_partition_reference=w["protocol"]["cohort_partition_reference"],
        methods_protocol_reference=w["protocol_reference"],
        relational_evidence_protocol=w["protocol"]["relational_evidence_protocol"],
        installed_batching_protocol=w["protocol"]["installed_batching_protocol"],
        terminal_producer_rows=w["history"]["terminal_producer_rows"],
        small_head_numerical_inference=w["numerical"],
        runtime_reader=w["runtime"],
        optional_runtime_gate_check_summary=w["runtime"].get(
            "gate_check_summary", w["runtime"].get("failures", [])
        ),
        branch_input_gate_check_summary=w["history"]["gate_check_summary"],
        historical_input_gate_check_summary=w["history"]["historical_gate_check_summary"],
        historical_quarantines=w["protocol"]["historical_quarantines"],
        prior_invocation_failure_reference=original.PRIOR_FAILURE_REFERENCE,
        prior_invocation_failure=e.read(original.PRIOR_FAILURE_REFERENCE),
        preserved_v743_failures=w["history"]["preserved_v743_failures"],
        preserved_nulls=[
            "V733 source-information",
            "V735 cold-cost",
            "V737 supervisor",
            "V743 batch-one",
        ],
        future_dependencies=w["protocol"]["future_dependencies_v745"],
        methodology_note="Qualify immutable reporting and fourteen sealed tasks. Branch operands are independent of failed historical execution. Registration and cached numerical checks establish no learning, extraction, device performance or independent generalization.",
    )
    value["field_principles"] = {
        k: "Bind this invocation to exact original evidence. Readiness is distinct from scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        field_principles="Explain each published field.",
        reproducibility_checksum="Hash all fields except this checksum.",
    )
    for names, principle in {
        "experiment_id task_id milestone run_date invocation_id publication_output": "Bind each claim to this producer and exact invocation.",
        "honest_verdict verdict_class gate_check_summary branch_input_gate_check_summary optional_runtime_gate_check_summary": "Owned failure disqualifies; external absence blocks its scope. Missing operands differ from observed zero.",
        "inference_substrate inference_substrate_class MODEL_SPECS model_invocation_counts current_llm_invocations substrate_by_phase": "Language-model loads and calls are zero. Numerical head evaluations and historical models are separate evidence.",
        "target_model small_head_numerical_inference historical_model_provenance": "Authenticate the numerical checkpoint separately. Historical model identity does not describe current inference.",
        "rows contract_rows intended_count completed_count failed_count censored_count excluded_count independent_count sample_size_budget": "Keep fourteen scheduling objects. Administrative controls supply zero independent scientific questions.",
        "verifier_is_oracle exposure_scope independent_generalization_score generalized_learning_benefit_score": "Constructed or exposed controls cannot establish independent generalization or learning benefit.",
        "required_checks_passed flagged_adversarial acceptance_gates validation_receipts adversarial_findings terminal_validation_sidecar_path": "Exact child exits, owned producer coverage and unchanged terminal checks qualify methods. Preserve actual findings.",
        "preconditions_checked duration_s started_at_utc ended_at_utc phase_spans random_seed": "Use measured resource checks, real measurement clocks and separate validation child clocks. Never estimate duration from mtimes.",
        "source_artifact_hashes code_config_hashes raw_shard_hashes cited_upstream_artifacts primitive_reference measurement_seal_reference validation_reference": "Bind original bytes and independently located invocation evidence. Historical unavailable operands remain diagnosed by their original readers.",
        "methods_ready_score branch_gate_values learning_operands_ready_score extraction_inputs_ready_score deployment_inputs_ready_score runtime_repair_ready_score": "Only complete owned validation qualifies methods. Original operands qualify each branch separately; optional CUDA absence stays local.",
        "immutable_build_rows retained_shard_rows authority_mutation_rows": "Retain actual nested-hash proofs, untouched SQLite and full-object mutations. Missing or changed evidence rejects.",
        "methods_manifest methods_manifest_references objective_updates objective_audit cohort_partition_reference relational_evidence_protocol installed_batching_protocol future_dependencies": "Keep original objectives and branch methods exact. Register symbolic and integer studies separately; future deliverables are dependencies.",
        "historical_quarantines prior_invocation_failure_reference prior_invocation_failure preserved_v743_failures preserved_nulls historical_input_gate_check_summary": "Preserve stamped failures and closed nulls. This invocation cannot rehabilitate old artifacts.",
    }.items():
        value["field_principles"].update(dict.fromkeys(names.split(), principle))
    value["reproducibility_checksum"] = e.checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild from independently located invocation evidence, rejecting repaired local claims."""
    try:
        value = json.loads(path.read_bytes())
        output = Path(value["publication_output"])
        raw = output.parent / "raw" / e.NAME / "invocations" / value["invocation_id"]
        if value["measurement_seal_reference"] != e.reference(raw / "measurement_seal.json"):
            return False
        if value["primitive_reference"] != e.reference(raw / "measurement.json"):
            return False
        if value["validation_reference"] != e.reference(raw / "validation.json"):
            return False
        work = e.read(value["primitive_reference"])
        if e.read(value["measurement_seal_reference"]) != work:
            return False
        if not all(
            e.checked(ref)
            for ref in value["raw_shard_hashes"] + work["source_refs"] + work["code_refs"]
        ):
            return False
        if (
            e.protocol() != work["protocol"]
            or work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9
            or not 0 <= work["duration_s"] <= 3600
            or e.authority(Path(work["root"])) != work["contract"]
            or e.mutation_controls(Path(work["root"])) != work["authority_mutation_rows"]
            or e.historical(work["protocol"]) != work["history"]
            or original.numerical(work["protocol"]) != work["numerical"]
            or e.methods_manifests(work["protocol"]) != work["methods_manifest"]
        ):
            return False
        refs = list(work["runtime"].get("cached_references", []))
        if work["runtime"].get("inventory_reference"):
            refs += [work["runtime"]["inventory_reference"]]
        if not all(e.checked(ref) for ref in refs):
            return False
        for receipt in e.read(value["validation_reference"])["receipts"]:
            for stream in ["stdout", "stderr"]:
                if stream + "_path" in receipt and not e.checked(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                ):
                    return False
        return bool(value == build(work, value["validation_reference"], raw, output))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def consumer(output: Path) -> Json:
    """Keep both unchanged readers' literal acceptance or blocked result."""
    return dict(reader_receipt(e.TASK, output.parent, field="methods_ready_score"))
