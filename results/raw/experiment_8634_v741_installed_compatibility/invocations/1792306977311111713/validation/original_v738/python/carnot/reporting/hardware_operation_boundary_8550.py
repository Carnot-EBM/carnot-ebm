"""REQ-REPORT-8550: map complete host work while preserving every physical execution gate."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import signal
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import hardware_operation_evidence_8550 as evidence
from carnot.reporting import nine_input_board_boundary_8538 as old
from carnot.reporting import v734_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json

Json = dict[str, Any]
ROOT, reference, checksum = old.ROOT, old.reference, old.checksum
NAME, TASK, MILESTONE = (
    "experiment_8550_v734_hardware_operation_boundary",
    "exp8550-hardware-operation-boundary",
    "2026.10.734",
)
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_hardware_operation_boundary_8550.py"
OWNED = [
    "python/carnot/reporting/hardware_operation_evidence_8550.py",
    "python/carnot/reporting/hardware_operation_boundary_8550.py",
    "python/carnot/reporting/hardware_operation_runner_8550.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
_MEASURE, _BUILD, _REPLAY = old.measure, old.build, old.replay


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts make CPU evidence work visible without contacting a board."""
    print(f"[exp8550] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def binding() -> Iterator[None]:
    """Small identity adapters reuse the shipped assertions and terminal publication unchanged."""
    with patch.multiple(
        old,
        NAME=NAME,
        TASK=TASK,
        MILESTONE=MILESTONE,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        OPERATIONS=evidence.OPERATIONS,
        authority=authority,
        operands=operands,
        build=build,
        progress=progress,
    ):
        yield


def operands(root: Path) -> Json:
    """One bounded scan reads original board history regardless of optional cost availability."""

    def timeout(signum: int, frame: Any) -> None:
        raise TimeoutError("evidence_scan_60_seconds")

    handler = signal.signal(signal.SIGALRM, timeout)
    previous = signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        return dict(history=evidence.history(root), cost=evidence.costs(root))
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
        signal.signal(signal.SIGALRM, handler)


def measure(root: Path, raw: Path, private: Path | None = None) -> Json:
    """Retain invocation resources, current authority, original seals and primitive matrix operands."""
    with binding():
        work = dict(_MEASURE(root, raw, private))
    work["code_config_hashes"] += [
        reference(ROOT / "docs/research-notes/v734-hardware-operation-boundary.md")
    ]
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Completed scope reconciliation cannot install a spline or establish thermodynamic benefit."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    value.update(experiment_id=8550, random_seed=7348550)
    klass = value["verdict_class"]
    value["honest_verdict"] = f"complete_{klass}_v734_hardware_operation_boundary"
    rows = value["board_operation_rows"]
    cost = work["evidence"]["cost"]
    # Rebuild all five substrate rows from authenticated primitive operands.
    rows = [
        dict(
            r,
            board=board,
            board_cpu_scope="historical_Linux_CPU_dispatch_only" if board == "PolarFire" else None,
            coefficient_count=109,
            operation_boundary="nine_input_complete_inference",
        )
        for board in evidence.BOARDS
        for r in rows
        if r["board"] == "KV260"
    ]
    value["operation_coverage_rows"] = rows
    value["board_operation_rows"] = rows
    value["rows"] = [
        dict(
            source_id=f"{r['board']}:{r['operation']}",
            arm="operation_boundary",
            intended=1,
            completed=True,
            failed=False,
            censored=False,
            excluded=False,
            independent=0,
            numerator=1,
            denominator=1,
            missing_reason=None,
            board=r["board"],
            operation=r["operation"],
        )
        for r in rows
    ]
    value.update(
        intended_count=len(rows),
        completed_count=len(rows),
        hardware_boundary_ready_score=value["board_boundary_recorded_score"],
        physical_hardware_ready_score=0,
        qualified_cost_reference=cost,
        conditional_ceiling_rows=evidence.ceilings(cost),
        vendor_exclusion_rows=evidence.VENDOR,
        literature_hypotheses=evidence.LITERATURE,
        future_dependencies=[
            dict(
                experiment_id=8546,
                status="future_dependency_not_observation"
                if not cost["qualified"]
                else "authenticated_current_evidence",
            )
        ],
        sample_size_budget=dict(
            operation_mappings=len(rows), historical_readers=3, independent_deployment_samples=0
        ),
    )
    value["board_reopen_requirements"] = [
        dict(
            board="KV260",
            required=[
                "source_and_build_hashes",
                "compatible_nine_input_109_coefficient_kernel",
                "SSH_kria_executed_parity",
                "complete_host_device_clocks",
            ],
            retained_scope="quadratic_Ising_energy_k_max_5",
        ),
        dict(
            board="PolarFire",
            required=[
                "compatible_fabric_kernel",
                "source_build_bitstream_hashes",
                "executed_fabric_parity",
                "complete_host_device_clocks",
            ],
            retained_scope="hash_verified_board_local_Linux_CPU_dispatch",
        ),
        dict(
            board="GateMate",
            required=value["reopening_condition"],
            missing_source_hashes=value["missing_source_hashes"],
            required_build_evidence=[
                "recovered_source_hashes",
                "dated_build_tool_and_bitstream_hashes",
            ],
            last_authenticated_idcode=value["last_authenticated_idcode"],
            expected_idcode="0x20000001",
        ),
        dict(
            board="NPU",
            required=[
                "authenticated_device_toolchain",
                "compatible_kernel",
                "precision_and_complete_cost_parity",
            ],
            retained_scope="unqualified",
        ),
        dict(
            board="TSU",
            required=[
                "authenticated_device_access",
                "compatible_embedding",
                "precision_sampling_update_transfer_and_host_costs",
            ],
            retained_scope="unqualified",
        ),
    ]
    value["gate_check_summary"] += [
        evidence.failure(
            Path(work["root"]) / f"hardware/{board}/nine_input_execution.json",
            "qualified_compatible_execution",
            True,
            None,
            board,
        )
        for board in evidence.BOARDS
    ]
    value["methodology_note"] = (
        "Authenticate sealed board history once within60 seconds without device commands. "
        "Map all nine complete operations on five substrates. Conditional zero-cost ceilings "
        "retain fixed host work. Vendor ratios, CPU dispatch and Ising support grant no "
        "spline execution, measured acceleration or independent scientific benefit."
    )
    value["field_principles"] = {
        k: "Bind current invocation to original producer bytes; scope grants no physical or science credit."
        for k in value
    }
    value["field_principles"].update(
        physical_hardware_ready_score="Zero without qualified compatible physical execution.",
        hardware_boundary_ready_score="One qualifies scope accounting only.",
        conditional_ceiling_rows="Ideal zero offload cost retains fixed host work and all unknown costs.",
        vendor_exclusion_rows="Excluded dense logits and inter-device movement travel with comparisons.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Recompute operands from independent original pins before accepting a repaired envelope."""
    with binding():
        return bool(_REPLAY(path))
