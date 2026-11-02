"""REQ-VERIFY-8615: derive registration from immutable operands and actual checks.

A digest can be repaired after tampering. Reconstructing the report from original
family readers prevents that repaired digest from inventing qualification.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v737_contract_report as old
from carnot.reporting import v740_contract_methods as e
from carnot.reporting.primary_publication import reader_receipt

Json = dict[str, Any]


def build(work: Json, validation: Json, raw: Path, output: Path) -> Json:
    """Qualify administration independently of device availability and scientific benefit."""
    with patch.object(old, "e", e):
        value = dict(old.build(work, validation, raw, output))
    p = work["protocol"]
    controls_ok = all(row["passed"] for row in work["authority_mutation_rows"])
    if not controls_ok:
        value.update(
            methods_ready_score=0,
            current_contract_ready_score=0,
            required_checks_passed=False,
            flagged_adversarial=True,
            verdict_class="disqualified",
        )
        value["gate_check_summary"].append(
            e.gate(raw / "measurement.json", "authority_mutations", True, False, e.TASK)
        )
        value["acceptance_gates"].update(owned_validation=False, sealed_methods=False)
    value["acceptance_gates"]["authority_mutations"] = controls_ok
    head = e.read(p["original_producers"]["8431"])
    value.update(
        experiment_id=8615,
        run_date=work["run_date"],
        intended_count=12,
        honest_verdict=f"complete_{value['verdict_class']}_v740_contract_methods",
        sample_size_budget=dict(
            task_slots=12, independent_questions=0, repeats_multiply_evidence=False
        ),
        random_seed=7408615,
        literature_method_rows=p["literature_method_rows"],
        future_dependencies=p["future_dependencies"],
        preserved_quarantines=dict(
            never_clear=True,
            original_terminal_references=[
                p["original_producers"][str(n)]
                for n in [8564, 8565, 8578, 8582, 8584, 8588, 8590, 8592]
            ],
        ),
        gate_field_map=p["gate_field_map"],
        inherited_operand_checks=p["inherited_operand_checks"],
        triviaplus_coverage_plan=p["triviaplus_coverage_plan"],
        lean_worker_2x2=p["lean_worker_2x2"],
        action_observer_plan=p["action_observer_plan"],
        target_model=dict(
            name="Exp8431 nine-input109-coefficient spline energy head",
            checkpoint=dict(path=head["frozen_heads_path"], sha256=head["frozen_heads_sha256"]),
            evaluated_this_invocation=False,
        ),
        methods_protocol_reference=work["protocol_reference"],
        learning_protocol_reference=p["learning_protocol_reference"],
        authority_mutation_rows=work["authority_mutation_rows"],
        preserved_v739_failure=p["preserved_v739_failure"],
        lean_worker_comparisons=p["lean_worker_comparisons"],
        methodology_note="Register twelve full V740 task objects against current YAML and independently printed design. Preserve V738 numerical operands and original readers. Authenticate learning and resident inputs independently. No current model loads, numerical inference, device probe, learning benefit or serving measurement. Missing authority blocks; owned check failure disqualifies. Historical nulls and quarantines remain unchanged.",
    )
    value["acceptance_gates"].pop("complete_twelve_objects", None)
    value["acceptance_gates"]["complete_twelve_objects"] = bool(work["contract"]["ready"])
    value["branch_gate_values"] = {
        k: value[k]
        for k in [
            "methods_ready_score",
            "current_contract_ready_score",
            "learning_inputs_ready_score",
            "resident_inputs_ready_score",
            "canary_inputs_ready_score",
            "runtime_repair_ready_score",
        ]
    }
    value["field_principles"].update(
        {
            k: "Bind current invocation to original immutable evidence; method qualification cannot establish scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        learning_inputs_ready_score="Original head and158-node union with current Exp8580 validator and Exp8581 accounting; independent of resident and optional resources.",
        resident_inputs_ready_score="Original head plus independent Exp8593 provenance and Exp8596 complete costs; independent of learning qualification.",
        runtime_repair_ready_score="No readiness without supplied causal repair receipt and current validator custody; metadata alone is insufficient.",
        methods_ready_score="Owned validation, resources and all twelve complete authority objects; no scientific benefit follows.",
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct every output scalar and reject changed primitive or validation streams."""
    try:
        value = json.loads(path.read_bytes())
        work = e.read(value["primitive_reference"])
        if (
            e.protocol() != work["protocol"]
            or e.checksum(value) != value["reproducibility_checksum"]
            or work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9
            or not 0 <= work["duration_s"] <= 3600
            or not all(e.checked(r) for r in work["source_refs"] + work["code_refs"])
        ):
            return False
        if (
            e.mutation_controls(Path(work["root"])) != work["authority_mutation_rows"]
            or e.authority(Path(work["root"])) != work["contract"]
            or e.historical(work["protocol"]) != work["history"]
        ):
            return False
        runtime = work["runtime"]
        if (
            runtime["ready"] is not False
            or runtime["probe_count"] != 0
            or runtime["context_opened"] is not False
        ):
            return False
        if not all(
            e.checked(r)
            for r in runtime["cached_references"]
            + (
                [runtime["inventory_reference"]]
                if runtime["inventory_reference"] is not None
                else []
            )
        ):
            return False
        for receipt in e.read(value["validation_reference"])["receipts"]:
            for stream in ["stdout", "stderr"]:
                if stream + "_path" in receipt and not e.checked(
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
    """Actual downstream resolvers must retain their real acceptance or blocked outcome."""
    return dict(reader_receipt(e.TASK, output.parent, field="methods_ready_score"))
