"""REQ-REPORT-8761: qualified administrative mapping preserves absent science."""

from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any

from carnot.reporting import hardware_cost_boundary_8747 as historical
from carnot.reporting import hardware_cost_inputs_8761 as source
from carnot.reporting import hardware_operation_boundary_8721 as original
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
e = namespace(original, "cost_report_8761")
c = namespace(historical, "cost_rows_8761")
e.NAME, e.TASK = "experiment_8761_v750_hardware_cost_boundaries", "exp8761-hardware-cost-boundaries"
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.TESTS = ["tests/python/test_hardware_cost_boundaries_8761.py"]
e.OWNED = [
    "python/carnot/reporting/hardware_cost_boundaries_8761.py",
    "python/carnot/reporting/hardware_cost_inputs_8761.py",
    "python/carnot/reporting/hardware_cost_runner_8761.py",
    e.CLI,
]
BOUNDARIES = ("causal_learning", "installed_inference_only", "ordinary_durable_window")
c.c.BOUNDARIES = BOUNDARIES
BASE_BUILD = e.build


def cost_rows(branches: list[Json], support: list[Json]) -> Json:
    """Only authenticated quadratic fabric work can enter a disjoint parent fraction."""
    prepared = deepcopy(branches)
    for branch in prepared:
        for row in branch.get("value", {}).get("complete_cost_rows", []):
            if row.get("duration_s") is not None and not math.isfinite(row["duration_s"]):
                raise ValueError("non_finite_cost")
            row["clock_domain_original"] = row.get("clock_domain", row.get("parent"))
            # Original measurements use integer nanoseconds; floating sums can exceed a parent by one ulp.
            for field in ("duration_s", "parent_duration_s"):
                if row.get(field) is not None:
                    row[field] = round(row[field] * 1e9)
    support = deepcopy(support)
    for row in support:
        if row["support"] == "supported" and (
            row["board"] != "KV260" or row["operation"] != e.OPERATIONS[0]
        ):
            row["support"] = "unsupported"
    result: Json = c.cost_rows(prepared, support)
    for key in ("boundary_cost_rows", "operation_support_rows", "conditional_acceleration_bounds"):
        for row in result[key]:
            for field in (
                "duration_s",
                "parent_duration_s",
                "reported_parent_duration_s",
                "measured_duration_s",
                "compatible_duration_s",
            ):
                if row.get(field) is not None:
                    row[field] /= 1e9
    for row in result["operation_support_rows"]:
        if row["support"] == "missing-evidence":
            row["support"] = "unknown"
    for row in result["boundary_cost_rows"]:
        row["clock_domain"] = row.pop("clock_domain_original")
    for row in result["conditional_acceleration_bounds"]:
        row.update(
            includes_measured_host_and_transfer=False,
            includes_measured_host_costs=True,
            physical_transfer_duration_s=None,
            transfer_included_in_ideal_bound=False,
        )
    result["cost_scope_note"] = (
        "Separate causal learning, installed inference-only and ordinary durable windows. Ordinary windows cannot estimate natural renormalization frequency. Features, host nonlinear work, durability and unknown transfer remain outside claimed fabric speedups."
    )
    return result


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Administrative readiness can complete while hardware benefit remains unmeasured."""
    value: Json = BASE_BUILD(work, receipts, output)
    evidence = work["evidence"]
    missing = any(not b["qualified"] for b in evidence["branches"])
    verdict = (
        "disqualified"
        if not value["required_checks_passed"]
        else "blocked"
        if missing or not value["hardware_mapping_ready_score"]
        else "null"
    )
    value.update(cost_rows(evidence["branches"], value["operation_support_rows"]))
    value.update(
        experiment_id=8761,
        task_id=e.TASK,
        milestone=source.contract.MILESTONE,
        random_seed=7508761,
        verdict_class=verdict,
        honest_verdict=f"complete_{verdict}_hardware_cost_boundaries",
        device_commands_executed=[],
        numerical_training=dict(performed=False),
        model_invocation_counts=dict(language_model_calls=0, model_loads=0),
        canonical_tasks_sha256=source.contract.PIN,
        historical_disqualified_producers=evidence["historical_disqualified_producers"],
        imported_process_cost_rows=[
            row
            for branch in evidence["branches"]
            if branch["qualified"]
            for row in branch["value"].get("process_cost_rows", [])
        ],
        repository_health_reference=dict(
            reference=work.get("repository_health_reference"),
            proves_complete_full_suite_pass=False,
            acceptance_dependency=False,
        ),
        scoped_validation_scope=dict(
            test_files=e.TESTS, e2e=["E2E-018", "E2E-021"], repository_health_is_dependency=False
        ),
        ideal_required_compatible_fraction={"10x": 0.9, "100x": 0.99},
        acquisition_priority_delta=dict(
            changed=False, purchase_authorized=False, reason="No compatible physical cost evidence."
        ),
        methodology_note="Independently authenticate each exact producer and its original replay. Import disjoint installed-process costs from the unchanged reducer; do not multiply historical child fractions. No training, model calls, benchmark or device commands. Preserve every process, operation and parent denominator; administrative mapping and physical acceleration are separate.",
        interface_hypotheses=[
            dict(
                source="https://arxiv.org/abs/2602.02056v4",
                workload="nine-input109-coefficient signed32/24 sparse basis, residual and Brier updates with checked signed64 accumulation",
                required_measurement="Exact state and probability parity, measured actual dense events, complete host/transfer/durability/restart clocks; no paper latency transfers.",
            ),
            dict(
                source="https://extropic.ai/writing/from-one-to-one-billion",
                workload="explicit quadratic Ising sampling graph with frozen couplings and sample-quality target",
                required_measurement="Authenticated device/SDK access, exact graph embedding, distribution quality, programming and transfer plus complete-service clocks.",
            ),
            dict(
                source="https://extropic.ai/writing/z1t/",
                workload="sparse probabilistic sampling operations only; no deterministic spline-update equivalence established",
                required_measurement="Operation and precision parity plus independent sample quality and full host/device costs before TSU/NPU experiment or acquisition.",
            ),
        ],
        preserved_historical_limits=[
            "V749 seven producers and five skipped studies; no new qualified learning or cost result.",
            "Exp8737 historical tests read mutable vNEXT; Exp8738 profiling signal11 does not resolve original timeout cause.",
            "Exp8743 and Exp8747 stay disqualified. Exp8745 natural middle window had zero dense writes.",
            "Historical batching1.1382x and same-batch10x null remain historical CPU results.",
        ],
    )
    for row, boundary in zip(value["rows"], BOUNDARIES, strict=True):
        row.update(boundary=boundary, raw_denominator=1, raw_numerator=row["value"])
    value["gate_check_summary"] += [
        dict(
            upstream=e.TASK,
            path=r.get("stdout_path"),
            sha256=r.get("stdout_sha256"),
            check=r.get("name"),
            field="passed",
            operator="==",
            expected=True,
            observed=False,
            passed=False,
            operand_state="failed_owned_validation",
        )
        for r in receipts
        if not r["passed"]
    ]
    value["acquisition_prerequisites"].update(
        purchase_authorized=False,
        recommended=False,
        required=[
            "supported exact workload and precision",
            "complete relevant parent boundary",
            "measured transfer",
            "actual device execution and quality",
        ],
    )
    value.pop("reproducibility_checksum")
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Bind exact producer bytes and separate qualified execution, clock boundary, device support and scientific benefit. Absence is not zero cost.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


e.inputs, e.build, e.progress = source.i, build, source.progress

# Late import lets the shared runner call this already-configured report module.
from carnot.reporting import hardware_cost_runner_8761 as r  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Defer runner lookup so either public module can be imported first."""
    return int(r.main(argv))
