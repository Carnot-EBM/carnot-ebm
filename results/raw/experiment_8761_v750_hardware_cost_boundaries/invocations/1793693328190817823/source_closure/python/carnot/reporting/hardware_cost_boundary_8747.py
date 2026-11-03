"""REQ-REPORT-8747: model-free cost mapping preserves complete workload limits."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil
from typing import Any

from carnot.reporting import hardware_operation_boundary_8721 as original_report
from carnot.reporting import hardware_operation_runner_8721 as original_runner
from carnot.reporting import hardware_operation_runner_8707 as original_checks
from carnot.reporting import hardware_cost_rows_8735 as original_rows
from carnot.reporting import hardware_cost_inputs_8747 as source
from carnot.reporting import v748_collection_contract as namespaces
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
e = namespaces.namespace(original_report, "hardware_cost_report_8747")
r = namespaces.namespace(original_runner, "hardware_cost_runner_8747")
r.retained = namespaces.namespace(original_checks, "hardware_cost_checks_8747")
c = namespaces.namespace(original_rows, "hardware_cost_rows_8747")
i, progress, authenticate, admit, producer_operands = (
    source.i,
    source.progress,
    source.authenticate,
    source.admit,
    source.producer_operands,
)
BOUNDARIES = ("causal_learning", "installed_inference_only", "bounded_durable_window")
c.BOUNDARIES = BOUNDARIES
e.NAME, e.TASK = "experiment_8747_v749_hardware_cost_boundary", "exp8747-hardware-cost-boundary"
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.TESTS = ["tests/python/test_hardware_cost_boundary_8747.py"]
e.OWNED = [
    "python/carnot/reporting/hardware_cost_boundary_8747.py",
    "python/carnot/reporting/hardware_cost_inputs_8747.py",
    e.CLI,
]
BASE_BUILD, BASE_VALIDATE = e.build, r.validate
FULL_SUITE_LOGS = Path("/var/tmp/carnot8747-full-suite")


def cost_rows(branches: list[Json], support: list[Json]) -> Json:
    """Only nonoverlapping leaf spans in a named parent may enter an ideal ceiling."""
    prepared = deepcopy(branches)
    for branch in prepared:
        if not branch["qualified"]:
            continue
        spans = branch.get("value", {}).get(
            "component_span_rows", branch.get("value", {}).get("complete_cost_rows", [])
        )
        groups: dict[tuple[str, str], list[Json]] = {}
        for row in spans:
            if row.get("nested_parent_span") or row.get("includes_child_spans"):
                raise ValueError("nested_parent_child_spans")
            if not row.get("parent"):
                row["parent_duration_s"] = None
            key = (str(row.get("process_id")), str(row.get("parent")))
            groups.setdefault(key, []).append(row)
        for rows in groups.values():
            if len({row.get("clock_domain", row.get("parent")) for row in rows}) > 1:
                for row in rows:
                    row["reported_parent_duration_s"] = row.get("parent_duration_s")
                    row["parent_duration_s"] = None
            timed = sorted(
                (row["start_ns"], row["end_ns"])
                for row in rows
                if "start_ns" in row and "end_ns" in row
            )
            if any(start < previous_end for (_, previous_end), (start, _) in zip(timed, timed[1:])):
                raise ValueError("overlapping_operation_spans")
    result: Json = c.cost_rows(prepared, support)
    result["boundary_cost_rows"] = result["complete_cost_rows"]
    result["cost_scope_note"] = (
        "Separate causal_learning, installed_inference_only and bounded_durable_window. Windows and arithmetic prefixes cannot fill missing complete learning or service costs. Host extraction, transfer and per-feedback persistence remain unaccelerated."
    )
    return result


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Mapping eligibility never substitutes for physical execution or scientific benefit."""
    value: Json = BASE_BUILD(work, receipts, output)
    value.update(
        experiment_id=8747,
        task_id=e.TASK,
        milestone=source.contract.MILESTONE,
        random_seed=7498747,
        honest_verdict=f"complete_{value['verdict_class']}_hardware_cost_or_physical_execution",
        device_commands_executed=[],
        canonical_tasks_sha256=source.contract.PIN,
        numerical_training=dict(performed=False),
        model_invocation_counts=dict(
            value["model_invocation_counts"], language_model_calls=0, model_loads=0
        ),
        ideal_required_compatible_fraction={"10x": 0.9, "100x": 0.99},
        acquisition_priority_delta=dict(
            changed=False, reason="No admitted compatible physical cost evidence."
        ),
        preserved_historical_limits=[
            "V748 ran ten producers and skipped four studies. Exp8723 qualified collection.",
            "Exp8724 passed39 core nodes; deadline_smoke timed out at300 seconds and readiness stayed zero.",
            "Exp8729 equal-call gain0.0667 CI95[0.0333,0.1000] is exposed constructed circular evidence.",
            "Exp8731 resident integer parity is not fabric support or speed.",
            "Exp8732 projected71507.5252224 seconds for122880 durable updates and stopped.",
            "Exp8733 duplicate extraction share45.63% uses native child-service clocks only.",
            "No useful learning, current CUDA repair, new ARC outcome or physical acceleration established.",
        ],
        methodology_note="Read-only independently authenticated upstream aggregation. No model load, generation, benchmark or device command. Retain exact source bytes, separate complete and window denominators, and allow ideal Amdahl ceilings only for disjoint measured compatible operations inside one named clock boundary. CPU feature reuse is never device acceleration.",
    )
    value.update(cost_rows(work["evidence"]["branches"], value["operation_support_rows"]))
    value["gate_check_summary"] += [
        dict(
            upstream=e.TASK,
            path=receipt.get("stdout_path"),
            sha256=receipt.get("stdout_sha256"),
            check=receipt.get("name"),
            field="passed",
            operator="==",
            expected=True,
            observed=False,
            passed=False,
            operand_state="failed_owned_validation",
        )
        for receipt in receipts
        if not receipt["passed"]
    ]
    for row, boundary in zip(value["rows"], BOUNDARIES, strict=True):
        row.update(
            boundary=boundary, duration_s=None, raw_denominator=1, raw_numerator=row["value"]
        )
    value["cpu_feature_reuse_rows"] = [
        dict(
            producer=b["task_id"],
            boundary=BOUNDARIES[1],
            device_attribution=None,
            source_references=b["refs"],
            cost_rows=[
                row for row in value["boundary_cost_rows"] if row["producer"] == b["task_id"]
            ],
        )
        for b in work["evidence"]["branches"][1:2]
        if b["qualified"]
    ]
    value["acquisition_prerequisites"].update(
        purchase_authorized=False,
        recommended=False,
        required=[
            "supported operation",
            "complete relevant workload",
            "measured transfer",
            "actual device execution",
        ],
    )
    value.pop("reproducibility_checksum")
    value["field_principles"] = {
        k: "Bind claims to exact producers and measured named boundaries; missing evidence is not zero cost. Mapping and physical acceleration are separate."
        for k in [*value, "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def validate(raw: Path) -> list[Json]:
    """Reuse real disjoint coverage and private readers; retain the full-suite result."""
    receipts: list[Json] = BASE_VALIDATE(raw)
    directory = raw / "full_suite"
    directory.mkdir()
    # The full suite is run once; its original command receipt remains auditable.
    for path in FULL_SUITE_LOGS.glob("*"):
        shutil.copyfile(path, directory / path.name)
    ref = i.reference(directory / "full_python_suite.receipt.json")
    receipt = i.load(ref)
    receipt["references"] = [i.reference(p) for p in sorted(directory.glob("*"))]
    receipts.append(receipt)
    return receipts


e.inputs, e.build, e.progress = i, build, progress
r.e, r.i, r.validate = e, i, validate
r.BASE_VALIDATE, r.BASE_REPLAY = r.retained.validate, r.retained.replay
r.BASE_TERMINAL, r.BASE_RUN = r.retained.terminal, r.retained.run
main = r.main
