"""REQ-VERIFY-8653: reconstruct current readiness without changing historical readers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v741_contract_report as previous
from carnot.reporting import v743_contract_methods as e
from carnot.reporting.primary_publication import reader_receipt

Json = dict[str, Any]
_BUILD = previous.build


def build(work: Json, validation: Json, raw: Path, output: Path) -> Json:
    """Qualify the contract while keeping research inputs and future execution separate."""
    with patch.object(previous, "e", e):
        value = dict(_BUILD(work, validation, raw, output))
    p = work["protocol"]
    value.update(
        experiment_id=8653,
        intended_count=14,
        sample_size_budget=dict(
            task_slots=14, independent_questions=0, repeats_multiply_evidence=False
        ),
        cohort_partition_reference=p["cohort_partition_reference"],
        reset_callback_payload_protocol=p["reset_callback_payload_protocol"],
        random_seed=7438653,
        honest_verdict=f"complete_{value['verdict_class']}_v743_contract_methods",
        learning_operands_ready_score=work["history"]["learning_operands_ready_score"],
        circular_gate_regression_rows=work["history"]["circular_gate_regression_rows"],
        historical_quarantines=p["historical_quarantines"],
        reset_provenance_protocol=p["reset_provenance_protocol"],
        small_head_numerical_inference=work["numerical"],
        source_study_seed=p["source_study_seed"],
        installed_cost_cells=p["installed_cost_cells"],
        methodology_note="Register fourteen full V743 tasks against independent visible and machine authority. Authenticate research operands separately from failed historical execution. Evaluate one original nine-input 109-coefficient energy head without loading a language model. Future execution, training, natural extraction, installed costs and passive reset diagnostics require separate qualified producers. Preserve every historical null and quarantine. Optional runtime absence does not veto CPU work.",
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
