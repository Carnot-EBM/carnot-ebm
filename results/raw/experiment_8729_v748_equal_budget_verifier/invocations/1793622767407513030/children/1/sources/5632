"""REQ-VERIFY-8667: reconstruct current readiness without changing historical readers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v741_contract_report as previous
from carnot.reporting import v744_contract_methods as e
from carnot.reporting.primary_publication import reader_receipt

Json = dict[str, Any]
_BUILD = previous.build


def build(work: Json, validation: Json, raw: Path, output: Path) -> Json:
    """Qualify the contract while keeping research inputs and future execution separate."""
    with patch.object(previous, "e", e):
        value = dict(_BUILD(work, validation, raw, output))
    p = work["protocol"]
    value.update(
        experiment_id=8667,
        intended_count=14,
        sample_size_budget=dict(
            task_slots=14, independent_questions=0, repeats_multiply_evidence=False
        ),
        cohort_partition_reference=p["cohort_partition_reference"],
        reset_callback_payload_protocol=p["reset_callback_payload_protocol"],
        random_seed=7448667,
        honest_verdict=f"complete_{value['verdict_class']}_v744_contract_methods",
        learning_operands_ready_score=work["history"]["learning_operands_ready_score"],
        circular_gate_regression_rows=work["history"]["circular_gate_regression_rows"],
        historical_quarantines=p["historical_quarantines"],
        reset_provenance_protocol=p["reset_provenance_protocol"],
        small_head_numerical_inference=work["numerical"],
        source_study_seed=p["source_study_seed"],
        installed_cost_cells=p["installed_cost_cells"],
        methodology_note="Register fourteen full V744 tasks against independent visible and machine authority. Authenticate research operands separately from failed historical execution. Evaluate one original nine-input 109-coefficient energy head without loading a language model. Future execution, training, natural extraction, installed costs and passive reset diagnostics require separate qualified producers. Preserve every historical null and quarantine. Optional runtime absence does not veto CPU work.",
    )
    for key in [
        "relational_evidence_protocol",
        "installed_batching_protocol",
        "observation_ownership_protocol",
        "qwen_question_protocol",
        "learning_registration",
    ]:
        value[key] = p[key]
    value["preserved_v743_failures"] = work["history"]["preserved_v743_failures"]
    value["source_artifact_hashes"] += p["protocol_operand_references"]
    value["producer_map_v744"] = p["producer_map_v744"]
    value["exposure_scope"] = (
        "Administrative contract; 120 constructed worlds; 190 cached responses in 128 exposed groups; original learning questions remain exposed."
    )
    value["substrate_by_phase"] = dict(
        readers="aggregation_from_upstream_artifacts",
        authority_controls="deterministic_verifier",
        checkpoint="verifier_ensemble_against_cached_candidates",
    )
    value["optional_model_pre_skip_task_ids"] = (
        ["exp8676-qwen-claim-canary"] if not work["runtime"]["ready"] else []
    )
    value["gate_check_summary"] += work["history"]["gate_check_summary"] + work["runtime"].get(
        "gate_check_summary", work["runtime"].get("failures", [])
    )
    value["gate_check_summary"] = deepcopy(value["gate_check_summary"])
    for row in value["gate_check_summary"]:
        row.setdefault(
            "operand_state",
            e.gate(Path(row["path"]), row["field"], row["expected"], row["observed"])[
                "operand_state"
            ],
        )
    value["acceptance_gates"].pop("complete_thirteen_objects", None)
    value["acceptance_gates"]["complete_fourteen_objects"] = bool(work["contract"]["ready"])
    value["raw_shard_hashes"] += [
        ref
        for receipt in value["validation_receipts"]
        if receipt["name"] == "original_coverage_retention"
        for ref in receipt["original_references"]
    ]
    value["newly_owned_statement_coverage_complete"] = any(
        r["name"] == "coverage_report" and r["passed"] for r in value["validation_receipts"]
    )
    value["prior_invocation_failure_reference"] = e.PRIOR_FAILURE_REFERENCE
    value["prior_invocation_failure"] = e.read(e.PRIOR_FAILURE_REFERENCE)
    value["raw_shard_hashes"] += [e.PRIOR_FAILURE_REFERENCE] + value["prior_invocation_failure"][
        "references"
    ]
    value["target_model"]["evaluated_this_invocation"] = True
    value["branch_gate_values"]["learning_operands_ready_score"] = value[
        "learning_operands_ready_score"
    ]
    value["historical_input_gate_check_summary"] = work["history"]["historical_gate_check_summary"]
    value["branch_input_gate_check_summary"] = work["history"]["gate_check_summary"]
    value["field_principles"].update(
        {
            k: "Bind this invocation to authenticated original operands; registration cannot establish scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"]["learning_operands_ready_score"] = (
        "Head, data, labels, validator, accounting, sealed objective and original test inventory; failed execution is not an input dependency."
    )
    value["field_principles"]["methods_ready_score"] = (
        "Owned validation, resources and all fourteen complete authority objects; registration establishes no scientific benefit."
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value


def replay(path: Path) -> bool:
    """Recompute numerical work as well as every inherited contract and validation claim."""
    try:
        value = json.loads(path.read_bytes())
        if not all(e.checked(ref) for ref in value["raw_shard_hashes"]):
            return False
        work = e.read(value["primitive_reference"])
        if work["numerical"] != e.numerical(work["protocol"]):
            return False
        with patch.multiple(previous, e=e, build=build):
            return bool(previous.replay(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def consumer(output: Path) -> Json:
    """Both existing publication readers must select the exact current artifact."""
    return dict(reader_receipt(e.TASK, output.parent, field="methods_ready_score"))
