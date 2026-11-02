"""REQ-REPORT-8413: authenticate board custody without inventing supported work.

The previous reader already checks terminal dispatch and original transcripts.
Process-local adapters bind current authority while preserving those old bytes.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import board_operation_boundary_8399 as prior
from carnot.reporting import historical_consumer_roots_8402 as authority
from carnot.reporting import python_transaction_cost_8407 as costs
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary

Json = dict[str, Any]
ROOT, base = prior.ROOT, prior.base
NAME, TASK, MILESTONE = (
    "experiment_8413_v724_board_operation_boundary",
    "exp8413-board-operation-boundary",
    "2026.10.724",
)
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_board_operation_boundary_8413.py"
OWNED = [
    "python/carnot/reporting/board_operation_boundary_8413.py",
    "python/carnot/reporting/board_operation_runner_8413.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
TASK_PIN = "sha256:96d70a51d3822d3963e7dec6b299485c553a6d3530716af64292f64989a975ab"
READER = "results/experiment_8399_v723_board_operation_boundary.json"
READER_PIN = "sha256:c3da081389ccaf7b7aed3f7ba9c344bb902e90395a638b525f65902220c50aad"
SOURCES = [
    "experiment_8407_v724_python_transaction_cost",
    "experiment_8408_v724_native_transaction_cost",
]
OPERATIONS = [
    "input_validation",
    "snapshot_pinning",
    "python_boundary_conversion",
    "direct_scoring",
    "calibration",
    "delayed_release",
    "coefficient_update",
    "scheduling",
    "temporary_write",
    "file_fsync",
    "pointer_publication",
    "pointer_replace",
    "directory_fsync",
    "acknowledgment",
    "response",
    "transaction_overhead",
    "transfers",
]
gate = failure = prior.gate
_operands, _measure, _build, _replay, _map = (
    prior.operands,
    prior.measure,
    prior.build,
    prior.replay,
    prior.operation_map,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual phase counts so evidence reduction cannot hide a stalled child."""
    print(f"[exp8413] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def bindings() -> Iterator[None]:
    """Reuse frozen guards with current authority without editing historical modules."""
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
    ):
        yield


def operation_map(rows: list[Json]) -> tuple[list[Json], float | None, list[str]]:
    """Only complete clocks permit a fraction; none of these host operations fit Ising."""
    with patch.object(prior, "OPERATIONS", OPERATIONS):
        result: tuple[list[Json], float | None, list[str]] = _map(rows)
    return result


def cost_branch(path: Path, raw: Path, work: Json) -> Json:
    """Import only independently authenticated primitive costs, never a pre-gate score."""
    branch: Json = dict(
        path=str(path),
        present=path.is_file(),
        reference=None,
        rows=[],
        disposition="unavailable",
        serialized_bytes=None,
        transferred_bytes=None,
    )
    field = "python_cost_ready_score" if path.stem == SOURCES[0] else "native_cost_ready_score"
    work["checks"].append(gate(path, "cost_primary_present", True, path.is_file()))
    if not path.is_file():
        return branch
    try:
        value = json.loads(base.pin(path, raw, work["refs"]).read_bytes())
        validate_primary(value, path)
        terminal = Path(value["terminal_validation_sidecar_path"])
        publication = json.loads(base.pin(terminal, raw, work["refs"]).read_bytes())
        bound = read_bound_sidecar(path, Path(publication["sidecar_path"]))
        base.pin(Path(publication["sidecar_path"]), raw, work["refs"])
        if (
            publication["primary_sha256"] != base.sha256_file(path)
            or publication["primary_path"] != str(path.absolute())
            or bound["primary_path"] != str(path.absolute())
            or not bound["report"]["passed"]
            or value["required_checks_passed"] is not True
            or value["flagged_adversarial"] is not False
            or value.get(field) != 1
        ):
            raise ValueError("cost_terminal_checks")
        for source_ref in value["raw_shard_hashes"]:
            base.checked(source_ref)
        if value["verdict_class"] not in {"positive", "circular_positive", "null"}:
            raise ValueError("unqualified_cost_terminal")
        if (
            value["experiment_id"] != int(path.stem.split("_")[1])
            or value["milestone"] != MILESTONE
        ):
            raise ValueError("cost_producer_identity")
        ref = value["work_reference"]
        source = json.loads(base.pin(base.checked(ref), raw, work["refs"]).read_bytes())
        selected = [
            r for r in value["repeat_rows"] if r.get("cell") == value["primary_cell"]["cell"]
        ]
        if len(selected) != 5 or {r["repeat"] for r in selected} != set(range(1, 6)):
            raise ValueError("five_primary_repeats")
        clocks: Json = {}
        serialized, wall_ns = 0, 0
        for index, summary in enumerate(selected):
            progress("cost_repeat_before", index, 5 - index)
            primitive_ref = next(
                r
                for r in source["results"]
                if not r["warmup"]
                and json.loads(base.checked(r).read_bytes())["repeat_id"] == summary["repeat_id"]
            )
            primitive = json.loads(
                base.pin(base.checked(primitive_ref), raw, work["refs"]).read_bytes()
            )
            reduced = costs.k.reduce_repeat(source["bundle"], primitive)
            if reduced != summary:
                raise ValueError("primitive_cost_reduction")
            for name, duration in reduced["stage_ns"].items():
                clocks[name] = clocks.get(name, 0) + duration
            wall_ns += primitive["end_ns"] - primitive["start_ns"]
            serialized += reduced["serialized_bytes"]
            progress("cost_repeat_after", index + 1, 4 - index)
        clocks["transaction_overhead"] = wall_ns - sum(clocks.values())
        rows = [dict(operation=name, cost_ns=duration) for name, duration in clocks.items()]
        operation_map(rows)
        branch.update(
            reference=ref,
            rows=rows,
            disposition="qualified_primary_cell",
            serialized_bytes=serialized,
            complete_wall_ns=wall_ns,
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        work["checks"].append(gate(path, "primitive_cost_reduction", "authenticated", str(error)))
    return branch


def operands(root: Path, raw: Path) -> Json:
    """Keep the qualified historical reader and optional cost branches independent."""
    with (
        bindings(),
        patch.object(
            prior,
            "SOURCES",
            [
                "experiment_8393_v723_python_transaction_cost",
                "experiment_8394_v723_native_transaction_cost",
            ],
        ),
    ):
        work: Json = _operands(root, raw)
    ready = False
    try:
        authority.checked(root / READER, READER_PIN)
        reader = base.probe(root / READER, raw, work, "board_reader_ready_score")
        if reader is None or reader["polarfire_graduation"] != work["history"].get("graduation"):
            raise ValueError("qualified_reader_dispatch_binding")
        authority.checked(root / authority.PROTOCOL, authority.PROTOCOL_PIN)
        base.pin(root / authority.PROTOCOL, raw, work["refs"])
        ready = bool(work["reader_ready"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        work["checks"].append(gate(root / READER, "qualified_V723_reader", READER_PIN, str(error)))
    work["reader_ready"] = ready
    work["checks"].append(gate(root / READER, "V724_reader_history_binding", True, ready))
    work["branches"] = [
        cost_branch(root / "results" / (name + ".json"), raw, work) for name in SOURCES
    ]
    work["absent"].extend(b["path"] for b in work["branches"] if not b["present"])
    return work


def measure(root: Path, raw: Path) -> Json:
    """Seal current code and reused guards before validation reads any outcomes."""
    with bindings():
        work: Json = _measure(root, raw)
    mounts = [line.split() for line in Path("/proc/self/mountinfo").read_text().splitlines()]
    mounted = max(
        (m for m in mounts if raw.resolve().is_relative_to(Path(m[4]))), key=lambda m: len(m[4])
    )
    filesystem = mounted[mounted.index("-") + 1]
    work["preconditions"].update(
        filesystem=filesystem,
        disk_backed=filesystem not in {"tmpfs", "ramfs"},
        tool_hashes=[
            base.reference(ROOT / ".venv/bin" / name)
            for name in ["python", "pytest", "coverage", "ruff", "mypy"]
        ],
    )
    work["checks"].append(
        gate(raw, "disk_backed_evidence", True, work["preconditions"]["disk_backed"])
    )
    for name in [
        "python/carnot/reporting/board_operation_boundary_8399.py",
        "python/carnot/reporting/historical_consumer_roots_8402.py",
        "python/carnot/reporting/python_transaction_cost_8407.py",
        "python/carnot/verify/python_transaction_cost_8407.py",
        "docs/research-notes/v724-hardware-operation-boundary.md",
    ]:
        base.pin(ROOT / name, raw, work["code_config_hashes"])
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Qualification keeps the unsupported hardware branch blocked and science closed."""
    with bindings():
        value: Json = _build(work, receipts, raw, output)
    value.update(
        experiment_id=8413,
        random_seed=7248413,
        run_date="20261011",
        honest_verdict="complete_" + value["verdict_class"] + "_board_operation_boundary",
        eligible_cost_fraction=value["compatible_cost_fraction"],
        eligible_operation_count=0,
        board_operation_rows=value["operation_rows"],
        unsupported_operation_rows=value["operation_rows"],
        optional_cost_dispositions=[dict(b) for b in work["branches"]],
        kv260_k_max=5,
        polarfire_execution_scope="board_local_Linux_CPU_only",
        device_command_count=0,
        hardware_purchase_required=False,
        closed_exposed_utility=dict(H1=-0.00390625, H2=0, reopened=False),
        optional_cost_dependencies=["results/" + name + ".json" for name in SOURCES],
        qualified_board_reader_reference=next(
            (r for r in work["refs"] if r["original_path"] == str(Path(work["root"]) / READER)),
            None,
        ),
        next_device_evidence=[
            "compatible_spline_and_calibration_kernel_identity",
            "complete_request_and_response_transfer_bytes_and_clocks",
            "exclusive_lease_and_firmware_hash",
            "hash_verified_matched_device_transactions_including_fsync",
        ],
        methodology_note="Authenticate the qualified Exp8399 reader and original PolarFire CPU dispatch. Reduce five preselected Python primary-cell repeats through the shipped primitive reducer. Serialized host bytes are not device transfers; absent calibration/transfer clocks remain unavailable. No model, benchmark or device command runs. H1 and H2 stay closed.",
    )
    value["board_obligations"]["gatemate"]["task_id"] = "exp8414-gatemate-obligation-delta"
    value["raw_shard_hashes"].extend(
        base.reference(path) for path in sorted((raw / "coverage_shards").glob(".coverage*"))
    )
    value["field_principles"].update(
        {
            k: f"Bind {k} to sealed V724 operands; absent clocks stay null and historical CPU custody grants no acceleration or scientific benefit."
            for k in value
        }
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(path: Path) -> bool:
    """Independently rebuilt operands reject summary tampering even after a rehash."""
    with bindings():
        return bool(_replay(path))
