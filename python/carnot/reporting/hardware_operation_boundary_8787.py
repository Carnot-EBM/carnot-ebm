"""REQ-REPORT-8787: complete mappings and physical acceleration are separate results."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import hardware_operation_boundary_8721 as original
from carnot.reporting import hardware_operation_inputs_8787 as source
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
e = namespace(original, "boundary_8787")
e.NAME, e.TASK = (
    "experiment_8787_v752_hardware_operation_boundary",
    "exp8787-hardware-operation-boundary",
)
e.CLI = f"scripts/experiments/{e.NAME}.py"
e.TESTS = ["tests/python/test_hardware_operation_boundary_8787.py"]
e.OWNED = [
    f"python/carnot/reporting/hardware_operation_{p}_8787.py"
    for p in ["boundary", "inputs", "runner"]
] + [e.CLI]
BASE_BUILD = e.build
MODEL_SPECS: list[Json] = []
BOARDS = ("KV260", "PolarFire", "GateMate", "CUDA", "CPU")
BOUNDARIES = ("new_template_learning", "installed_inference_only", "ordinary_durable_window")
OPERATIONS = [
    source.PAIRWISE,
    "constraint_lookup",
    "relational_exact_checks",
    "template_admission",
    "feature_construction",
    "calibration_sigmoid",
    "small_head_Brier_training",
    "sparse_update_arithmetic",
    "dense_renormalization",
    "global_decay",
    "exact_spline_energy",
    "digest_serialization",
    "durable_fsync",
    "checkpoint_import",
    "restart_verification",
    "host_device_transfer",
    "durable_window_unpartitioned",
    "distributed_Ising_communication",
]
METHOD = dict(
    boundaries=[*BOUNDARIES, "old_sparse_arithmetic", "memory_benefit_audit"],
    fraction="admitted disjoint leaves divided by the entire same-clock parent",
    target="finite 100x requires compatible fraction strictly greater than 0.99",
    model_loads=0,
    numerical_training_calls=0,
    device_commands=0,
    physical_transfer="unknown; complete service bounds remain unknown",
)


def support_rows(qualified: bool, imported_operations: list[str] | None = None) -> list[Json]:
    """Installed pairwise fabric does not implement arbitrary joins or spline learning."""
    return [
        dict(
            board=board,
            operation=op,
            support="supported"
            if board == "CPU" or (qualified and board == "KV260" and op == source.PAIRWISE)
            else "unknown"
            if board in {"GateMate", "CUDA"}
            else "unsupported",
            execution_location="software_reference"
            if board == "CPU"
            else "historical_fabric"
            if board == "KV260" and op == source.PAIRWISE
            else "host_or_unqualified",
            interface="Linux CPU dispatch only"
            if board == "PolarFire"
            else "JTAG blocked:0xffffffff; expected0x20000001 after dated physical change"
            if board == "GateMate"
            else "SSH/scp/xmutil; pairwise UIO k_max<=5"
            if board == "KV260"
            else "No exact workload CUDA kernel authenticated"
            if board == "CUDA"
            else "Python/native CPU",
            physical_execution_this_invocation=False,
            measured_acceleration=False,
        )
        for board in BOARDS
        for op in dict.fromkeys([*OPERATIONS, *(imported_operations or [])])
    ]


def bounds(branches: list[Json], support: list[Json]) -> list[Json]:
    """Retain unassigned parent time; mixed clocks and nested spans cannot yield a bound."""
    groups: dict[tuple[str, str, str], list[Json]] = defaultdict(list)
    for b in branches:
        if b["qualified"]:
            for row in b["value"].get("complete_cost_rows", []):
                d = row.get("duration_s")
                if d is not None and (not math.isfinite(d) or d < 0):
                    raise ValueError("invalid_duration")
                if row.get("nested_parent_span") or row.get("includes_child_spans"):
                    raise ValueError("nested_costs")
                groups[b["task_id"], str(row.get("process_id")), str(row.get("parent"))].append(row)
    result = []
    for (producer, process, parent), rows in groups.items():
        totals = {r.get("parent_duration_s") for r in rows}
        total = next(iter(totals)) if len(totals) == 1 else None
        valid = (
            total is not None
            and math.isfinite(total)
            and total > 0
            and all(r.get("duration_s") is not None for r in rows)
            and len({r.get("clock_domain") for r in rows}) == 1
        )
        spans = sorted(
            (r["start_ns"], r["end_ns"]) for r in rows if "start_ns" in r and "end_ns" in r
        )
        if any(b < a or (i and a < spans[i - 1][1]) for i, (a, b) in enumerate(spans)):
            raise ValueError("overlapping_costs")
        assigned = sum(r["duration_s"] for r in rows) if valid else None
        if valid and assigned > total + 1e-9:
            raise ValueError("cost_exceeds_parent")
        for board in BOARDS:
            compatible = (
                sum(
                    r["duration_s"]
                    for r in rows
                    if any(
                        s["board"] == board
                        and s["operation"] == r["operation"]
                        and s["support"] == "supported"
                        and board == "KV260"
                        and r["operation"] == source.PAIRWISE
                        for s in support
                    )
                )
                if valid
                else None
            )
            fraction = compatible / total if valid else None
            result.append(
                dict(
                    board=board,
                    producer=producer,
                    process_id=process,
                    parent=parent,
                    parent_duration_s=total,
                    compatible_duration_s=compatible,
                    compatible_fraction=fraction,
                    unassigned_duration_s=max(0, total - assigned) if valid else None,
                    conditional_ceiling=1 / (1 - fraction)
                    if fraction is not None and fraction < 1
                    else None,
                    infinite_ceiling=fraction == 1,
                    feasible_100x=bool(fraction is not None and fraction > 0.99),
                    feasibility_target="100x",
                    feasibility_target_passed=bool(fraction is not None and fraction > 0.99),
                    complete_service_bound=None,
                    physical_setup_duration_s=None,
                    physical_transfer_duration_s=None,
                    measured_acceleration=False,
                    missing_reason=None if valid else "incomplete_or_mixed_clock_costs",
                )
            )
    return result


def build(work: Json, receipts: list[Json], output: Path) -> Json:
    """Mapping readiness describes completed evidence work, never a measured device gain."""
    with patch.object(e, "ROOT", Path(work["root"])):
        value: Json = BASE_BUILD(work, receipts, output)
    evidence = work["evidence"]
    branches = evidence["branches"]
    qualification = evidence["qualification"]
    costs = [
        dict(r, producer=b["task_id"], boundary=boundary, source_reference=b["refs"][0])
        for b, boundary in zip(branches, BOUNDARIES, strict=True)
        if b["qualified"]
        for r in b["value"].get("complete_cost_rows", [])
    ]
    operations = support_rows(evidence["board_qualified"], [r["operation"] for r in costs])
    mapped = bounds(branches, operations)
    for row, boundary in zip(value["rows"], BOUNDARIES, strict=True):
        row.update(
            boundary=boundary,
            raw_denominator=1,
            raw_numerator=row["value"],
            duration_s=None,
            missing_reason=None if row["value"] else "unqualified_operand",
        )
    for task, boundary, qualified in [
        ("historical_sparse_arithmetic", "old_sparse_arithmetic", False),
        (qualification["task_id"], "memory_benefit_audit", qualification["qualified"]),
    ]:
        value["rows"].append(
            dict(
                unit_id=task,
                arm=boundary,
                boundary=boundary,
                status="completed" if qualified else "censored",
                value=int(qualified),
                raw_denominator=1,
                raw_numerator=int(qualified),
                duration_s=None,
                censored=not qualified,
                excluded=False,
                effective_independent_groups=0,
                missing_reason=None if qualified else "no_current_complete_cost_or_audit",
            )
        )
    owned = value["required_checks_passed"]
    mapping = bool(value["hardware_mapping_ready_score"] and evidence["authority"]["ready"])
    verdict = "blocked" if owned else "disqualified"
    value.update(
        experiment_id=8787,
        task_id=e.TASK,
        milestone=source.contract.MILESTONE,
        honest_verdict=f"complete_{verdict}_hardware_operation_boundary",
        verdict_class=verdict,
        random_seed=7528787,
        method=METHOD,
        MODEL_SPECS=[],
        numerical_training_calls=0,
        numerical_training=dict(performed=False),
        model_invocation_counts=dict(language_model_calls=0, model_loads=0),
        intended_count=5,
        completed_count=sum(r["status"] == "completed" for r in value["rows"]),
        censored_count=sum(r["censored"] for r in value["rows"]),
        hardware_mapping_ready_score=int(mapping),
        physical_hardware_ready_score=0,
        complete_100x_learning_ready_score=0,
        device_commands_executed=[],
        canonical_tasks_sha256=source.contract.PIN,
        operation_support_rows=operations,
        complete_cost_rows=costs,
        conditional_acceleration_bounds=mapped,
        compatible_fraction_rows=[
            dict(r, required_fraction_strictly_greater_than=0.99) for r in mapped
        ],
        branch_admission_rows=[
            {k: v for k, v in b.items() if k != "value"} for b in [*branches, qualification]
        ],
        diagnostic_cost_rows=[
            dict(r, eligible_for_bound=False, producer=b["task_id"])
            for b in branches
            if b["qualified"]
            for r in b["value"].get("component_cost_rows", [])
        ],
        component_admission_rows=[
            source.i.gate(b["refs"][0], "component_cost_ready_score", 1, b["task_id"])
            for b in branches
            if b["qualified"] and b["value"]["experiment_id"] == 8784
        ],
        acceptance_gates=dict(
            owned_validation=owned,
            board_authority=evidence["board_qualified"],
            current_task_authority=evidence["authority"]["ready"],
        ),
        scientific_promotion_gates=dict(physical_execution=False, complete_100x_learning=False),
        feasibility_target_100x=dict(
            passed=False,
            measured=False,
            reason="No admitted complete learning denominator with compatible fraction>0.99 and physical setup/transfer.",
        ),
        acquisition_prerequisites=dict(
            purchase_authorized=False,
            recommended=False,
            required=[
                "authenticated device access",
                "available SDK and compiler",
                "exact operation and precision support",
                "graph/topology embedding and sample or deterministic parity",
                "complete host, communication, durable and device clocks",
            ],
        ),
        conditional_architecture_rows=[
            dict(
                architecture="sparse_KAN_locality",
                source="https://arxiv.org/abs/2602.02056v4",
                question="Can local basis/update state stay on chip while lookup, calibration and durable costs remain below1 percent?",
            ),
            dict(
                architecture="distributed_Ising",
                source="https://extropic.ai/writing/from-one-to-one-billion",
                question="Can graph partitioning include communication, synchronization and sample quality in the complete cost?",
            ),
            dict(
                architecture="Extropic_Z1T_topology",
                source="https://extropic.ai/writing/z1t/",
                question="Can sparse probabilistic topology encode the required operation without unaffordable embedding and host work?",
            ),
        ],
        architecture_comparison_scope="Conditional redesign questions; no vendor latency, energy or throughput number is transferred.",
        repository_health_reference=dict(
            reference=work.get("repository_health_reference"),
            proves_complete_full_suite_pass=False,
            acceptance_dependency=False,
        ),
        scoped_validation_scope=dict(
            test_files=e.TESTS, e2e=["E2E-018", "E2E-021"], repository_health_is_dependency=False
        ),
        cost_scope_note="Inference, new template learning, old sparse arithmetic and durable windows remain separate. Unknown parent work is retained. Rejected components never supply fractions. CPU dispatch is not FPGA fabric.",
    )
    value["gate_check_summary"] += [c for c in qualification["checks"] if not c["passed"]]
    value["gate_check_summary"] += [
        dict(
            upstream=e.TASK,
            path=r.get("stdout_path"),
            sha256=r.get("stdout_sha256"),
            check=r.get("name", "owned_validation"),
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
    value.pop("reproducibility_checksum")
    value["field_principles"] = dict.fromkeys(
        [*value, "reproducibility_checksum"],
        "Bind exact task and invocation bytes. Missing costs are unknown; mapping, scientific benefit and physical execution are separate.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


e.inputs, e.build, e.progress = source.i, build, source.progress

from carnot.reporting import hardware_operation_runner_8787 as r  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Use the established publisher and fresh sealed replay without device execution."""
    return int(r.main(argv))
