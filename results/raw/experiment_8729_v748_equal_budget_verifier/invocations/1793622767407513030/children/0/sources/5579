"""REQ-REPORT-8579: registration readiness is distinct from scientific benefit.

Every reported scalar is reconstructed from immutable input and validation
receipts, so editing and rehashing a candidate cannot manufacture readiness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import v737_contract_methods as e
from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.primary_publication import reader_receipt

Json = dict[str, Any]


def build(work: Json, validation: Json, raw: Path, output: Path) -> Json:
    """Qualify owned work while preserving external blockage and every historical flag."""
    receipts = e.read(validation)["receipts"]
    planned = (
        e.read(work["execution_manifest_reference"])["commands"]
        if "execution_manifest_reference" in work
        else []
    )
    manifest_matched = len(receipts) >= len(planned) and all(
        row["name"] == spec["name"]
        and row["argv"] == spec["argv"]
        and row["expected_exit"] == spec["expected"]
        for spec, row in zip(planned, receipts)
    )
    owned = all(r["passed"] for r in receipts) and manifest_matched
    contract = work["contract"]["ready"]
    rows = work["contract"]["rows"]
    resources = not work["precondition_failures"]
    methods = bool(owned and contract and resources)
    failures = list(work["contract"]["gate_check_summary"]) + work["precondition_failures"]
    if not manifest_matched:
        failures.append(e.gate(Path(validation["path"]), "required_command_manifest", True, False))
    failures += [
        e.gate(Path(r.get("stdout_path", validation["path"])), r["name"], True, False)
        for r in receipts
        if not r["passed"]
    ]
    verdict = (
        "disqualified"
        if not owned
        else "circular_positive"
        if contract and resources
        else "blocked"
    )
    history = work["history"]
    p = work["protocol"]
    value: Json = dict(
        experiment_id=8579,
        task_id=e.TASK,
        milestone=e.MILESTONE,
        run_date="20261016",
        honest_verdict=f"complete_{verdict}_v737_contract_methods",
        verdict_class=verdict,
        gate_check_summary=failures,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        current_llm_invocations=0,
        historical_model_provenance=history["historical_model_provenance"],
        rows=rows,
        intended_count=12,
        completed_count=sum(bool(r["matched"]) for r in rows),
        failed_count=sum(r["status"] != "unstarted" and not r["matched"] for r in rows),
        censored_count=sum(r["status"] == "unstarted" for r in rows),
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            task_slots=12, independent_questions=0, repeats_multiply_evidence=False
        ),
        verifier_is_oracle=True,
        exposure_scope="administrative contract; original 300 questions and FaithBench remain exposed",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        acceptance_gates=dict(
            owned_validation=owned, complete_twelve_objects=contract, sealed_methods=methods
        ),
        validation_receipts=receipts,
        validation_reference=validation,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=[
            finding for r in receipts for finding in r.get("adversarial_findings", [])
        ],
        preconditions_checked=work["preconditions"],
        duration_s=work["duration_s"] + sum(r.get("duration_s", 0) for r in receipts),
        phase_spans=work["phase_spans"],
        random_seed=7378579,
        source_artifact_hashes=work["source_refs"] + history["references"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[e.reference(raw / "measurement.json"), validation],
        cited_upstream_artifacts=list(p["original_producers"].values()),
        current_contract_ready_score=int(contract and owned),
        methods_ready_score=int(methods),
        runtime_repair_ready_score=int(work["runtime"]["ready"]),
        canonical_tasks_sha256=e.TASKS_PIN,
        methods_protocol_reference=work["protocol_reference"],
        historical_family_receipts=history["families"],
        closed_comparisons=p["closed_comparisons"],
        resident_session_protocol=p["resident_session"],
        count_evidence_protocol=p["count_evidence"],
        inherited_operand_checks=p["inherited_operand_checks"],
        downstream_gate_field_bindings=p["gate_field_map"],
        hallutruth_release_admissibility=p["hallutruth_release_admissibility"],
        primitive_reference=e.reference(raw / "measurement.json"),
        publication_output=str(output),
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_observation")
            for n in range(8580, 8591)
        ],
        optional_runtime=work["runtime"],
        historical_input_gate_check_summary=history["gate_check_summary"],
        archive_authority=p["archive_authority"],
        preserved_quarantines=dict(
            never_clear=True,
            original_terminal_references=[
                p["original_producers"][str(n)] for n in [8564, 8565, 8568, 8578]
            ],
        ),
        methodology_note="Hash-authenticated V735 methods and original qualification receipts. Fourteen complete task dictionaries must match independent printed and machine authority. Register count-evidence and resident-session methods before measurements. No model, device command, source refit, learning benefit or serving benefit measured. Historical disqualifications remain unchanged.",
    )
    for branch in ["learning", "resident", "canary"]:
        value[branch + "_inputs_ready_score"] = history[branch + "_inputs_ready_score"]
    value["field_principles"] = {
        k: "Bind this invocation to immutable original operands; administrative readiness does not establish independent scientific benefit."
        for k in value
    }
    value["field_principles"].update(
        learning_inputs_ready_score="Original Exp8431 head/scaler/temperature plus exact Exp8554 158-node receipt; independent of optional resources.",
        resident_inputs_ready_score="Original head plus qualified Exp8571 prototype and Exp8572 resident receipt; Exp8559 null remains closed.",
        canary_inputs_ready_score="Original head and qualified source/family receipts; runtime evidence is a separate gate.",
        runtime_repair_ready_score="Only supplied Exp8410 causal repair and real context/copy evidence.",
        methods_ready_score="Owned validation and all twelve complete authority objects; optional hardware is excluded.",
        reproducibility_checksum="Canonical full-artifact digest; cold reconstruction also checks every scalar.",
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct from original primitives; a self-consistent candidate hash is insufficient."""
    try:
        value = json.loads(path.read_bytes())
        work = e.read(value["primitive_reference"])
        if (
            e.protocol() != work["protocol"]
            or e.checksum(value) != value["reproducibility_checksum"]
            or work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9
            or not 0 <= work["duration_s"] <= 3600
        ):
            return False
        from carnot.reporting.v737_contract_methods import checked, historical

        refs = work["source_refs"] + work["code_refs"]
        if not all(checked(ref) for ref in refs):
            return False
        if (
            e.authority(Path(work["root"])) != work["contract"]
            or historical(work["protocol"]) != work["history"]
        ):
            return False
        validation = value["validation_reference"]
        for receipt in e.read(validation)["receipts"]:
            for stream in ["stdout", "stderr"]:
                if stream + "_path" in receipt and not checked(
                    dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                ):
                    return False
        return bool(
            value
            == build(
                work,
                validation,
                Path(value["primitive_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def consumer(output: Path) -> Json:
    """The actual conductor and documentation resolvers must select this exact primary."""
    return dict(reader_receipt(e.TASK, output.parent, field="required_checks_passed"))
