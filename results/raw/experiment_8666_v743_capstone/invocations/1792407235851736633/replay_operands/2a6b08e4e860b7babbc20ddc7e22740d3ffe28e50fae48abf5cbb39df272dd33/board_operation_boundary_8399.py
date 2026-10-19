"""REQ-REPORT-8399: bind current authority without upgrading historical hardware.

The shipped reader already authenticates board transcripts and complete costs.
Small process-local adapters retain its guards and keep old artifacts unchanged.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import board_operation_evidence_8385 as prior
from carnot.reporting import v723_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
ROOT, base = prior.ROOT, prior.base
NAME, TASK, MILESTONE = (
    "experiment_8399_v723_board_operation_boundary",
    "exp8399-board-operation-boundary",
    "2026.10.723",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_board_operation_boundary_8399.py"
OWNED = [
    "python/carnot/reporting/board_operation_boundary_8399.py",
    "python/carnot/reporting/board_operation_runner_8399.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
TASK_PIN = "sha256:15f74b227041d435e52d33f85324dc0453fd6ea5572559751f610d160979319b"
READER = "results/experiment_8385_v722_board_operation_evidence.json"
READER_PIN = "sha256:31c5b670140ef17b5fb66bc11ab7e5f661ca4702c6770ea20dcc9d07da14d4fb"
TERMINAL = "results/experiment_8371_v721_hardware_operation_boundary.json"
TERMINAL_PIN = "sha256:2beefe2f1f0f62a643ef9eeac1647dcb82d80b8d557ed3838f14c19104798efa"
DISPATCH_PIN = "sha256:81b5e66a7900896168558c536474c903e8166a1055d0ecf7ab1d713b48a850a4"
PROTOCOL_PIN = prior.authority.PROTOCOL_PIN
SOURCES = [
    "experiment_8393_v723_python_transaction_cost",
    "experiment_8394_v723_native_transaction_cost",
]
OPERATIONS = [
    "input_validation",
    "snapshot_pinning",
    "direct_scoring",
    "calibration",
    "delayed_release",
    "coefficient_update",
    "serialization",
    "file_fsync",
    "directory_fsync",
    "publication",
    "response",
    "transfers",
]
gate = failure = prior.gate
_operands, _measure, _build, _replay, _map, _probe = (
    prior.operands,
    prior.measure,
    prior.build,
    prior.replay,
    prior.operation_map,
    base.probe,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so evidence reads and synthesis remain visible to the operator."""
    print(f"[exp8399] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def bindings() -> Iterator[None]:
    """Reuse guards in this process so frozen producers keep their original code and identity."""
    with (
        patch.multiple(
            prior,
            NAME=NAME,
            TASK=TASK,
            MILESTONE=MILESTONE,
            TASK_PIN=TASK_PIN,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            SOURCES=SOURCES,
            OPERATIONS=OPERATIONS,
            operands=operands,
            build=build,
            operation_map=operation_map,
            progress=progress,
        ),
        patch.object(prior.authority, "authority", authority.authority),
        patch.object(base, "probe", probe),
    ):
        yield


def probe(path: Path, raw: Path, work: Json, field: str | None) -> Json | None:
    """Optional cost inputs must qualify independently before any timing is imported."""
    fields = dict(zip(SOURCES, ["python_cost_ready_score", "native_cost_ready_score"], strict=True))
    value: Json | None = _probe(path, raw, work, fields.get(path.stem, field))
    if path.stem in fields and value is not None:
        qualified = value.get("verdict_class") in {"positive", "circular_positive", "null"}
        work["checks"].append(gate(path, "qualified_cost_terminal", True, qualified))
        if not qualified:
            return None
    return value


def operation_map(rows: list[Json]) -> tuple[list[Json], float | None, list[str]]:
    """Every durable transaction stage needs a clock before a measured fraction is meaningful."""
    with patch.object(prior, "OPERATIONS", OPERATIONS):
        result: tuple[list[Json], float | None, list[str]] = _map(rows)
    for row in result[0]:
        row["missing_reason"] = "complete_stage_cost_absent" if row["cost_ns"] is None else None
    return result


def operands(root: Path, raw: Path) -> Json:
    """Bind the qualified reader and original terminal bytes, keeping absent costs explicit."""
    with bindings():
        work: Json = _operands(root, raw)
    progress("qualified_reader_before")
    ready = False
    try:
        for name, expected in [
            (READER, READER_PIN),
            (TERMINAL, TERMINAL_PIN),
            (prior.authority.PROTOCOL, PROTOCOL_PIN),
            (authority.PROTOCOL, authority.PROTOCOL_PIN),
        ]:
            path = root / name
            observed = base.sha256_file(path)
            if observed != expected:
                raise ValueError("source_pin:" + name)
            base.pin(path, raw, work["refs"])
        reader = _probe(root / READER, raw, work, "board_reader_ready_score")
        if reader is None or reader["polarfire_graduation"] != work["history"].get("graduation"):
            raise ValueError("qualified_reader_history_binding")
        if reader["polarfire_graduation"]["dispatch_sha256"] != DISPATCH_PIN:
            raise ValueError("original_dispatch_hash")
        ready = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        work["checks"].append(gate(root / READER, "qualified_board_reader", "valid", str(error)))
    work["reader_ready"] = ready
    work["checks"].append(gate(root / READER, "reader_history_binding", True, ready))
    progress("qualified_reader_after", int(ready), int(not ready))
    return work


def measure(root: Path, raw: Path) -> Json:
    """Seal all operands before checks so future producers cannot change this invocation."""
    with bindings():
        work: Json = _measure(root, raw)
    for name in [
        "python/carnot/reporting/v723_contract_methods.py",
        "python/carnot/reporting/board_operation_evidence_8385.py",
        "docs/research-notes/v723-hardware-operation-boundary.md",
    ]:
        refs: list[Json] = []
        base.pin(ROOT / name, raw, refs)
        work["code_config_hashes"].extend(refs)
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Reader readiness records validated aggregation; unsupported hardware remains blocked."""
    with bindings():
        value: Json = _build(work, receipts, raw, output)
    value.update(
        experiment_id=8399,
        run_date="20261011",
        random_seed=7238399,
        honest_verdict="complete_" + value["verdict_class"] + "_board_operation_boundary",
        board_reader_ready_score=int(
            value["board_reader_ready_score"]
            and work["reader_ready"]
            and work["contract"].get("activated", False)
        ),
        qualified_board_reader_reference=next(
            (r for r in work["refs"] if r["original_path"] == str(Path(work["root"]) / READER)),
            None,
        ),
        pre_gate_reference=None,
        supported_operations=[
            dict(
                operation="quadratic_Ising_energy",
                k_max=5,
                scope="historically_authenticated_overlay_only",
            )
        ],
        transaction_stage_policy="all_frozen_steps_plus_calibration_and_transfers_required",
        optional_cost_dependencies=["results/" + name + ".json" for name in SOURCES],
        NPU_available=False,
        TSU_available=False,
        vendor_performance_imported=False,
        global_health_status="not_run_in_this_invocation; consult_authenticated_historical_reader",
        methodology_note="Authenticate sealed board evidence and optional complete-stage costs. No device command, fresh timing or current LLM call occurs. Spline learning requires a future installed kernel and transfer protocol. H1=-0.00390625 and H2=0 remain closed exposed-development findings; the dyadic certificate is a separate undeployed numerical policy.",
    )
    reader_ref = value["qualified_board_reader_reference"]
    value["historical_repository_health"] = (
        json.loads(base.checked(reader_ref).read_bytes()).get("repository_health", [])
        if reader_ref
        else []
    )
    value["board_obligations"]["gatemate"]["task_id"] = "exp8400-gatemate-obligation-delta"
    value["gate_check_summary"] = [dict(check) for check in value["gate_check_summary"]]
    for check in value["gate_check_summary"]:
        check.setdefault("check", check["artifact_field"])
        check.setdefault("path", check.get("artifact_path"))
        check.setdefault("hash", check.get("artifact_hash"))
        check.setdefault("field", check["artifact_field"])
        check.setdefault("operator", check["op"])
    value["field_principles"] = {
        k: "Bind actual V723 authority and sealed operands; missing costs stay null, CPU graduation grants no fabric or generalization credit."
        for k in [*value, "field_principles"]
    }
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(path: Path) -> bool:
    """Rehashed reader claims must still agree with independently authenticated primitive gates."""
    with bindings():
        if not _replay(path):
            return False
    value = json.loads(path.read_bytes())
    work = json.loads(base.checked(value["work_reference"]).read_bytes())
    return bool(
        work["reader_ready"]
        == next(
            g["observed"] for g in work["checks"] if g["artifact_field"] == "reader_history_binding"
        )
    )
