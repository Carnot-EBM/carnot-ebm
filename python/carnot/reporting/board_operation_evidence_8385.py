"""REQ-REPORT-8385: immutable board evidence cannot grant new kernel capability."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import hardware_operation_boundary_8371 as old
from carnot.reporting import kv260_local_cost_boundary_8315 as base
from carnot.reporting import v722_contract_methods as authority
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
ROOT = base.ROOT
NAME = "experiment_8385_v722_board_operation_evidence"
TASK, MILESTONE = "exp8385-board-operation-evidence", "2026.10.722"
TASK_PIN = "sha256:391a8c96688b93c3dd70fd01f8482c2f752cd307c0c6dbe4c0213b6ddd9bef1d"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_board_operation_evidence_8385.py"
OWNED = [
    "python/carnot/reporting/board_operation_evidence_8385.py",
    "python/carnot/reporting/board_operation_runner_8385.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
OPERATIONS = ["direct_scoring", "updates", "serialization", "transfers", "fsync"]
SOURCES = [
    "experiment_8378_v722_python_transaction_cost",
    "experiment_8380_v722_native_transaction_cost",
]
HISTORY = old.HISTORY
_typed_map = old.operation_map


def gate(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Name each blocked operand so readers can distinguish absence from zero."""
    return dict(old.gate(path, field, expected, observed), check=field)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush phase counts so sealed evidence reads remain visible."""
    print(f"[exp8385] phase={phase} completed={completed} pending={pending}", flush=True)


@contextmanager
def source_aliases(refs: list[Json], absent: list[str] | None = None) -> Iterator[None]:
    """Read immutable bytes through historical names without changing those sources."""
    aliases = {
        r.get("original_path", r.get("source_path", r["path"])): base.checked(r) for r in refs
    }
    opened, is_file = Path.open, Path.is_file
    missing = set(absent or [])

    def read(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        target = aliases.get(str(path.absolute()), path) if "r" in mode else path
        return opened(target, mode, *args, **kwargs)

    def present(path: Path) -> bool:
        location = str(path.absolute())
        return location not in missing and is_file(aliases.get(location, path))

    with patch.object(Path, "open", read), patch.object(Path, "is_file", present):
        yield


def operation_map(rows: list[Json]) -> tuple[list[Json], float | None, list[str]]:
    """Reuse the strict cost schema while naming direct transaction operations."""
    with patch.object(old, "OPERATIONS", OPERATIONS):
        mapped, fraction, missing = _typed_map(rows)
    for row in mapped:
        row["reason"] = (
            "Ising_energy_k_max_le_5_has_no_direct_spline_update_serialization_transfer_or_fsync_kernel"
        )
    return mapped, fraction, missing


def operands(root: Path, raw: Path) -> Json:
    """Authenticate each external branch independently; failed history stays failed."""
    work: Json = dict(
        root=str(root), checks=[], refs=[], history={}, contract={}, branches=[], absent=[]
    )
    progress("authority_before")
    try:
        contract = authority.authority(root, raw / "authority")
        task = next(t for t in contract["tasks"] if t["id"] == TASK)
        if canonical_hash(task) != TASK_PIN:
            raise ValueError("active_task_digest")
        work["contract"] = dict(
            activated=contract["activated"],
            task_sha256=canonical_hash(task),
            canonical_tasks_sha256=contract["canonical_tasks_sha256"],
        )
        work["checks"].append(
            gate(
                root / authority.DESIGN, "independent_design_contract", True, contract["activated"]
            )
        )
        for name, expected in [
            (authority.previous.legacy.PROTOCOL, authority.previous.legacy.base.PIN),
            (authority.previous.PROTOCOL, authority.previous.DEPLOYMENT_PIN),
        ]:
            if base.sha256_file(root / name) != expected:
                raise ValueError("preserved_protocol_hash")
            base.pin(root / name, raw, work["refs"])
        for name in [authority.DESIGN, authority.ACTIVE]:
            base.pin(root / name, raw, work["refs"])
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        work["checks"].append(
            gate(root / authority.ACTIVE, "active_task_authority", TASK_PIN, str(error))
        )
    progress("authority_after_history_before")
    try:
        work["history"] = old.board_history(root, raw, work)
    except (OSError, ValueError, KeyError, TypeError) as error:
        work["checks"].append(
            gate(root / HISTORY, "historical_board_terminal", "valid", str(error))
        )
    progress("history_after_sources_before")
    for index, name in enumerate(SOURCES):
        path = root / "results" / (name + ".json")
        branch: Json = dict(
            path=str(path), present=path.is_file(), reference=None, rows=[], disposition="absent"
        )
        if not path.is_file():
            work["absent"].append(str(path.absolute()))
        else:
            try:
                declared = json.loads(base.pin(path, raw, work["refs"]).read_bytes())
                branch["disposition"] = declared.get(
                    "verdict_class", declared.get("status", "unqualified")
                )
            except (ValueError, TypeError, AttributeError) as error:
                branch["disposition"] = "unreadable:" + str(error)
        value = base.probe(path, raw, work, None)
        if value is not None:
            try:
                if (
                    value["experiment_id"] != int(name.split("_")[1])
                    or value["milestone"] != MILESTONE
                ):
                    raise ValueError("producer_identity")
                branch["disposition"] = value["verdict_class"]
                ref = value["operation_level_workload_reference"]
                branch["reference"] = ref
                trace = json.loads(base.pin(base.checked(ref), raw, work["refs"]).read_bytes())
                operation_map(trace["operation_rows"])
                branch["rows"] = trace["operation_rows"]
            except (OSError, ValueError, KeyError, TypeError) as error:
                work["checks"].append(
                    gate(
                        path,
                        "operation_level_workload_reference",
                        "authenticated_typed_rows",
                        str(error),
                    )
                )
        work["branches"].append(branch)
        progress("source_after", index + 1, len(SOURCES) - index - 1)
    for field in ["compatible_direct_spline_kernel", "complete_transaction_transfer_protocol"]:
        work["checks"].append(gate(root / HISTORY, field, "authenticated", None))
    return work


def measure(root: Path, raw: Path) -> Json:
    """Seal operands before validation so later producers cannot change this invocation."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    work = operands(root, raw)
    work["operand_check_count"] = len(work["checks"])
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    work["preconditions"] = dict(
        memory_available_bytes=int(memory["MemAvailable"].split()[0]) * 1024,
        memory_budget_bytes=1024**3,
        task_cap_s=4800,
        no_model_load=True,
        private_disk_scratch_required=True,
    )
    work["checks"].append(
        gate(
            Path("/proc/meminfo"),
            "available_memory_at_least_1GiB",
            True,
            work["preconditions"]["memory_available_bytes"] >= 1024**3,
        )
    )
    pre_gate = root / "results/experiment_8378_python_transaction_cost.json"
    work["pre_gate_reference"] = None
    if pre_gate.is_file():
        sealed = base.pin(pre_gate, raw, work["refs"])
        work["pre_gate_reference"] = base.reference(sealed)
    work.update(
        started_monotonic_ns=start, ended_monotonic_ns=time.monotonic_ns(), code_config_hashes=[]
    )
    for name in [
        *OWNED,
        TEST,
        *old.REUSED,
        "python/carnot/reporting/hardware_operation_boundary_8371.py",
        "python/carnot/reporting/kv260_workload_cost_8329.py",
        "python/carnot/reporting/v722_contract_methods.py",
    ]:
        refs: list[Json] = []
        base.pin(ROOT / name, raw, refs)
        work["code_config_hashes"].extend(refs)
    atomic_json(raw / "measurement.json", work)
    progress("measurement_sealed", len(work["branches"]), 0)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Reuse the historical terminal schema while preserving direct-service limits."""
    with (
        patch.object(old, "TASK", TASK),
        patch.object(old, "MILESTONE", MILESTONE),
        patch.object(old, "operation_map", operation_map),
    ):
        value: Json = old.build(
            work, [r for r in receipts if r.get("scope") != "global"], raw, output
        )
    value.update(
        experiment_id=8385,
        random_seed=7228385,
        honest_verdict="complete_" + value["verdict_class"] + "_board_operation_evidence",
        board_reader_ready_score=int(
            value["required_checks_passed"]
            and bool(work["history"])
            and not value["flagged_adversarial"]
        ),
        compatible_cost_fraction=value["compatible_fraction"],
        unsupported_operations=list(OPERATIONS),
        sample_size_budget=dict(
            branches=2, historical_boards=2, timing_repeats_are_independent=False
        ),
        pre_gate_reference=work["pre_gate_reference"],
        preconditions=work["preconditions"],
        validation_receipts=receipts,
        repository_health=[r for r in receipts if r.get("scope") == "global"],
        branch_dispositions=[
            dict(path=b["path"], disposition=b["disposition"], present=b["present"])
            for b in work["branches"]
        ],
        acquisition_relevance="larger_FPGA_or_TSU_purchase_requires_compatible_workload_and_complete_cost_evidence",
        vendor_context=dict(
            source="https://extropic.ai/writing/baby-thermo-rsi/",
            update="October_2026_research_agent_vendor_software_context",
            local_substrate=False,
        ),
        methodology_note="Direct scoring and complete durable transaction costs require qualified producers. Kernel-only Exp8356 clocks are historical only. Numerical parity is not semantic benefit.",
    )
    value["board_obligations"]["kv260"]["missing"] = [
        "compatible_direct_spline_kernel",
        "complete_transaction_transfer_protocol",
    ]
    value["board_obligations"]["gatemate"]["task_id"] = "exp8386-gatemate-obligation-delta"
    for row in value["rows"]:
        row["missing_reason"] = (
            None if row["completed"] else "qualified_operation_cost_or_authenticated_history_absent"
        )
    value["preserved_exp8371_verdict"] = "disqualified"
    value["field_principles"] = {
        k: "Bind sealed operands and task authority; absent costs remain null and board CPU scope grants no fabric or semantic benefit."
        for k in [*value, "field_principles", "reproducibility_checksum"]
    }
    value["field_principles"].update(
        board_reader_ready_score="Require owned validation and authenticated original board history; grant no current execution readiness.",
        compatible_cost_fraction="Divide supported cost by a complete positive measured denominator; incomplete or all-zero costs remain null.",
        unsupported_operations="Name each direct transaction operation absent from the authenticated Ising overlay.",
        branch_dispositions="Preserve producer class, primary presence and branch identity independently of performance.",
        pre_gate_reference="Retain exact conductor receipt bytes without treating them as measured operation costs.",
        repository_health="Retain the one full-suite command separately from owned qualification.",
        preconditions="Record available memory, declared budget, private disk requirement and bounded time cap.",
        vendor_context="Identify vendor software context without importing its performance or local access claims.",
        preserved_exp8371_verdict="Keep the original owned-validation failure disqualified after source aliases qualify private replay.",
    )
    value["reproducibility_checksum"] = canonical_hash(
        {k: v for k, v in value.items() if k != "reproducibility_checksum"}
    )
    return value


def replay(path: Path) -> bool:
    """Recompute authority, original dispatch and cost rows from immutable source aliases."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(base.checked(value["work_reference"]).read_bytes())
        for ref in [
            *value["source_artifact_hashes"],
            *value["code_config_hashes"],
            *value["raw_shard_hashes"],
        ]:
            base.checked(ref)
        if work["refs"] != value["source_artifact_hashes"]:
            return False
        for field in [
            "execution_manifest_reference",
            "owned_coverage_reference",
            "pre_gate_reference",
        ]:
            if work.get(field):
                base.checked(work[field])
        for receipt in value["validation_receipts"]:
            for stream in ["stdout", "stderr"]:
                if stream + "_path" in receipt:
                    base.checked(
                        dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"])
                    )
        with TemporaryDirectory(prefix="replay-", dir=path.parent) as directory:
            with source_aliases(work["refs"], work["absent"]):
                actual = operands(Path(work["root"]), Path(directory))
            for field in ["contract", "history", "branches", "absent"]:
                if actual[field] != work[field]:
                    return False
            if actual["checks"] != work["checks"][: work["operand_check_count"]]:
                return False
        rebuilt = build(
            work,
            value["validation_receipts"],
            Path(value["work_reference"]["path"]).parent,
            Path(value["publication_output"]),
        )
        return bool(rebuilt == value)
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False
