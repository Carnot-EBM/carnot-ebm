"""REQ-VERIFY-8591: reconstruct qualification from sealed operands and actual checks.

A matching checksum alone cannot prove a report. Replaying original readers
also rejects summaries that were changed and then consistently rehashed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v737_contract_report as old
from carnot.reporting import v738_contract_methods as e
from carnot.reporting import v738_contract_inputs as inputs
from carnot.reporting.primary_publication import reader_receipt

Json = dict[str, Any]


def build(work: Json, validation: Json, raw: Path, output: Path) -> Json:
    """Keep administrative qualification separate from benefit and historical failure."""
    with patch.object(old, "e", e):
        value = dict(old.build(work, validation, raw, output))
    p = work["protocol"]
    value.update(
        experiment_id=8591,
        run_date=work["run_date"],
        intended_count=11,
        honest_verdict=f"complete_{value['verdict_class']}_v738_contract_methods",
        sample_size_budget=dict(
            task_slots=11, independent_questions=0, repeats_multiply_evidence=False
        ),
        random_seed=7388591,
        canary_revision_binding=work["canary_revision_binding"],
        literature_method_rows=p["literature_method_rows"],
        medexpert_audit_plan=p["medexpert_audit_plan"],
        arc_v737_manifest=p["arc_v737_manifest"],
        future_dependencies=p["future_dependencies"],
        actual_numerical_inference=dict(
            evaluated=False,
            head_producer="Exp8431",
            head_kind="nine-input 109-coefficient spline energy head",
            checkpoint_reference=e.read(p["original_producers"]["8431"])["frozen_heads_path"],
        ),
        preserved_quarantines=dict(
            never_clear=True,
            original_terminal_references=[
                p["original_producers"][str(n)] for n in [8564, 8565, 8578, 8582, 8584, 8588, 8590]
            ],
        ),
        methodology_note="Administrative registration of eleven full V738 task objects and unchanged V737 numerical methods. Original evidence and current revision checks qualify independent CPU input gates. No model load, head evaluation, scientific benefit or device probe occurs. Source and cold-cost nulls, label rejections and V737 quarantines stay unchanged.",
    )
    value["acceptance_gates"].pop("complete_twelve_objects")
    value["acceptance_gates"]["complete_eleven_objects"] = bool(work["contract"]["ready"])
    value["field_principles"].update(
        {
            k: "Bind current invocation and original immutable evidence; qualification does not establish scientific benefit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        methods_ready_score="Owned checks and eleven complete objects; independent of optional canary and CUDA.",
        learning_inputs_ready_score="Original head and exact Exp8554 union plus affected Exp8580/8581 evidence.",
        resident_inputs_ready_score="Original head and Exp8571/8572 parity, independent of optional CUDA.",
        canary_revision_binding="Explicit new revision descriptor retains the rejected original hash and actual CLI controls.",
        literature_method_rows="Authenticated bounded primary methods and use/defer criteria; no transferred result.",
    )
    value["reproducibility_checksum"] = e.checksum(value)
    return value


def replay(path: Path) -> bool:
    """Reconstruct original inputs, clocks, optional checks and every output scalar."""
    try:
        value = json.loads(path.read_bytes())
        work = e.read(value["primitive_reference"])
        if (
            e.protocol() != work["protocol"]
            or e.checksum(value) != value["reproducibility_checksum"]
            or work["duration_s"] != (work["end_ns"] - work["start_ns"]) / 1e9
            or not 0 <= work["duration_s"] <= 3600
            or not all(e.checked(ref) for ref in work["source_refs"] + work["code_refs"])
        ):
            return False
        binding = work["canary_revision_binding"]
        descriptor = {
            k: v for k, v in binding.items() if k not in {"private_cli_receipts", "ready"}
        }
        if descriptor != inputs.canary_descriptor(work["protocol"]):
            return False
        receipts = list(binding["private_cli_receipts"])
        canary = all(r["passed"] for r in receipts) and inputs.accept_canary(
            Path(receipts[0]["argv"][-1])
        )
        if binding["ready"] != canary:
            return False
        h = e.historical(work["protocol"])
        h["canary_inputs_ready_score"] = int(canary)
        if e.authority(Path(work["root"])) != work["contract"] or h != work["history"]:
            return False
        receipts += e.read(value["validation_reference"])["receipts"]
        for receipt in receipts:
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
    """Both actual resolvers must choose these exact bytes and report their gate result."""
    return dict(reader_receipt(e.TASK, output.parent, field="methods_ready_score"))
