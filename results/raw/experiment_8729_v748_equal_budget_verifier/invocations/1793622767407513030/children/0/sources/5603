"""REQ-REPORT-8639: bind finite CPU reductions to this invocation's evidence."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v739_capstone_evidence as previous
from carnot.reporting import v741_contract_methods as methods
from carnot.reporting import v741_capstone_inputs as inputs
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
__all__ = ["methods"]
ROOT = methods.ROOT
NAME, TASK, MILESTONE = "experiment_8639_v741_capstone", "exp8639-capstone", "2026.10.741"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_v741_capstone_8639.py"]
OWNED = [
    f"python/carnot/reporting/v741_capstone{s}.py"
    for s in ["_inputs", "_evidence", "_samples", "_report", ""]
] + [CLI]
MODEL_SPECS: list[Json] = []
reference, read, checksum, atomic_json = (
    methods.reference,
    methods.read,
    methods.checksum,
    methods.atomic_json,
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush finite phase counts so the supervisor can detect real progress."""
    print(f"[exp8639] phase={phase} completed={completed} pending={pending}", flush=True)


def slots(raw: Path, log: Json) -> list[Json]:
    """Bind all earlier slots, including two cascade skips without primaries."""
    lines = Path(log["snapshot_path"]).read_text().splitlines()
    rows = []
    for i, task in enumerate(methods.protocol()["task_map_v741"][:-1]):
        progress("slot_before_" + task["id"], i, 13 - i)
        rows.append(inputs.resolve(task, raw, lines))
        progress("slot_after_" + task["id"], i + 1, 12 - i)
    return rows


def sample(raw: Path) -> tuple[Json, Json]:
    """The shipped bounded child releases historical allocations after reduction."""
    with patch.object(previous, "CLI", CLI):
        value, receipt = previous.sample(raw)
    return dict(value), dict(receipt)


def measure(raw: Path, run_date: str) -> Json:
    """Optional hardware and credentials are irrelevant to this CPU evidence branch."""
    start = time.monotonic_ns()
    progress("preconditions_before", 0, 13)
    private = Path(tempfile.mkdtemp(prefix="exp8639-", dir="/var/tmp"))
    pre = resources(private)
    pre.update(
        optional_cuda_required=False,
        optional_boards_required=False,
        scratch_write_read_passed=(private / "write_probe").read_bytes() == b"private disk scratch",
    )
    failures = []
    if not (
        pre["disk_backed"]
        and pre["tools_available"]
        and pre["scratch_write_read_passed"]
        and pre["free_disk_bytes"] >= 1024**3
        and pre["memory_available_bytes"] >= 1500 * 1024**2
    ):
        failures.append(methods.gate(private, "disk_memory_tools", True, False, TASK))
    contract = methods.authority(ROOT)
    head = methods.protocol()["head_checkpoint_reference"]
    read(head)
    log = previous.freeze(ROOT / "ops/conductor-log.md", raw / "conductor")
    rows = slots(raw / "primaries", log)
    progress("reduction_before", 12, 1)
    diagnostic, receipt = sample(raw / "samples")
    publication = child(
        "publication_gate",
        [str(ROOT / ".venv/bin/python"), "-u", "scripts/publication_gate.py", "--json"],
        raw / "publication",
        deadline=60,
    )
    end = time.monotonic_ns()
    if end - start >= 3600_000_000_000:
        raise TimeoutError("measurement_deadline")
    refs = (
        [log]
        + [r["primary_reference"] for r in rows]
        + [ref for r in rows for ref in r["evidence_references"]]
    )
    refs.append(head)
    refs += [
        previous.freeze(ROOT / p, raw / "source_inputs")
        for p in [
            methods.PROTOCOL,
            "results/experiment_8626_v740_capstone.json",
            "ops/exclusion_manifest.yaml",
            "scripts/publication_gate.py",
            "scripts/recurring_blocker_ledger.py",
            "scripts/experiment_template.py",
            "research-complete.yaml",
            "research-references.md",
        ]
    ]
    refs += [
        previous.freeze(ROOT / p, raw / "publication_inputs")
        for p in [
            "ops/publication_gate_state.json",
            "docs/technical-report.md",
            "docs/arxiv-paper/main.tex",
        ]
    ]
    work = dict(
        root=str(ROOT),
        raw=str(raw),
        run_date=run_date,
        start_ns=start,
        end_ns=end,
        duration_s=(end - start) / 1e9,
        phase_spans=[
            dict(
                phase="authentication_and_reduction",
                start_ns=start,
                end_ns=end,
                duration_s=(end - start) / 1e9,
            )
        ],
        protocol=methods.protocol(),
        contract=contract,
        conductor=log,
        slots=rows,
        declared_gates=inputs.bindings(methods.protocol()["task_map_v741"]),
        preconditions=pre,
        precondition_failures=failures,
        branch_receipts=[receipt],
        publication_receipt=publication,
        publication=json.loads(Path(publication["stdout_path"]).read_bytes()),
        samples=diagnostic,
        source_refs=refs,
        code_refs=[reference(ROOT / p) for p in OWNED + TESTS],
    )
    atomic_json(raw / "measurement.json", work)
    progress("measurement_complete", 12, 1)
    return work
