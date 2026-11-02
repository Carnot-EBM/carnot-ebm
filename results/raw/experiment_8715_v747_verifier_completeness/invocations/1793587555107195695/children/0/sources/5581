"""REQ-REPORT-8626: original producer seals distinguish scheduling from science."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v739_capstone_evidence as previous
from carnot.reporting import v740_contract_methods as methods
from carnot.reporting.board_operation_boundary_8454 import resources
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
__all__ = ["methods"]
ROOT = methods.ROOT
NAME, TASK, MILESTONE = "experiment_8626_v740_capstone", "exp8626-capstone", "2026.10.740"
CLI = f"scripts/experiments/{NAME}.py"
TESTS = ["tests/python/test_v740_capstone_8626.py"]
OWNED = [
    f"python/carnot/reporting/v740_capstone{s}.py" for s in ["_evidence", "_samples", "_report", ""]
] + [CLI]
MODEL_SPECS: list[Json] = []
reference, read, checksum, atomic_json = (
    methods.reference,
    methods.read,
    methods.checksum,
    methods.atomic_json,
)


def gate(path: Path, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """Attribute newly owned failures to this capstone instead of an older helper."""
    return dict(methods.gate(path, field, expected, observed, upstream))


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush finite CPU boundaries so supervisors can see completed and pending work."""
    print(
        f"[exp8626] phase={phase} completed={min(completed, 12)} pending={min(pending, 12)}",
        flush=True,
    )


def slots(raw: Path, log: Json) -> list[Json]:
    """Use unchanged producer and conductor readers with the twelve current objects."""
    lines = Path(log["snapshot_path"]).read_text().splitlines()
    rows = []
    for i, task in enumerate(methods.protocol()["task_map_v740"][:-1]):
        progress("slot_before_" + task["id"], i, 12 - i)
        rows.append(previous.resolve(task, raw, lines))
        progress("slot_after_" + task["id"], i + 1, 11 - i)
    return rows


def sample(raw: Path) -> tuple[Json, Json]:
    """A new bounded interpreter releases historical allocations after arithmetic replay."""
    with patch.object(previous, "CLI", CLI):
        value, receipt = previous.sample(raw)
    return dict(value), dict(receipt)


def measure(raw: Path, run_date: str) -> Json:
    """Authenticate tools and disk scratch without making optional hardware a CPU gate."""
    start = time.monotonic_ns()
    progress("preconditions_before", 0, 12)
    private = Path(tempfile.mkdtemp(prefix="exp8626-", dir="/var/tmp"))
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
        failures.append(gate(private, "disk_memory_tools", True, False, TASK))
    contract = methods.authority(ROOT)
    log = previous.freeze(ROOT / "ops/conductor-log.md", raw / "conductor")
    rows = slots(raw / "primaries", log)
    progress("reduction_before", 11, 1)
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
    refs += [
        reference(ROOT / p)
        for p in [
            methods.PROTOCOL,
            "results/experiment_8614_v739_capstone.json",
            "results/experiment_8601_v738_capstone.json",
            "ops/exclusion_manifest.yaml",
            "scripts/publication_gate.py",
            "scripts/experiment_template.py",
            "research-complete.yaml",
            "research-references.md",
            "openspec/change-proposals/research-roadmap-v738-preserved-20261017.md",
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
    progress("measurement_complete", 11, 1)
    return work
