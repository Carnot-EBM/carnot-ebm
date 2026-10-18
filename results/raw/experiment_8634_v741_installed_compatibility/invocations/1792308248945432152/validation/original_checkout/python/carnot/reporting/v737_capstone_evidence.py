"""REQ-REPORT-8590: original seals separate task completion from usable science."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v735_capstone_evidence as old
from carnot.reporting import v736_capstone_evidence as previous
from carnot.reporting import v737_contract_methods as methods
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v709_execution import child
from scripts.conductor_gates import _find_artifact_by_task_id

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK, MILESTONE = "experiment_8590_v737_capstone", "exp8590-capstone", "2026.10.737"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v737_capstone_8590.py"
TESTS = [TEST]
OWNED = [f"python/carnot/reporting/v737_capstone{s}.py" for s in ["_evidence", "_report", ""]] + [
    CLI
]
MODEL_SPECS: list[Json] = []
__all__ = ["methods", "old"]
reference, read, checksum, atomic_json, _gate = (
    methods.reference,
    methods.read,
    methods.checksum,
    methods.atomic_json,
    methods.gate,
)
freeze = old.freeze
PINS = dict(
    zip(
        range(8579, 8590),
        [
            "59aa488c9a37dd33cc5564bbe2f87b54ef6744cb63323dc6a291959985741a64",
            "7656f6f49b69237528263cd6dc38ac9503e853b1f53d1b5b1037f6510d2892f4",
            "f60e202794f220440094c940ea5d5e84d3a70b7fc8ba840ecf09fb915bf1896c",
            "4f7ec92ec04fb3cecc16af7e4c8a87682c63c2e1d575bfbeccdc8b02f1470c78",
            "a5e8c9c4dbfca7812b64a134d46ae90eb8b66bb80a0f669d334bd5caf02c7c05",
            "356aad75439ecaedbbe102e372dc28c2b92e15c848e0d6099165adb6a53995ed",
            "7ac56c0cb5d0eb3c67e3bb590fa2320c7c06138de53215a1aff5212ebbc1b758",
            "f224fcba439c4b6b4211cbe8f110f72ff7e3ae9aa321c79e82368a7a9ab9cf0a",
            "94f8314c802dfc49eb559c579c1f767b43d8f5718ef4811d0fe05abed9023816",
            "2624f20279c06245a393266f9ca793b3ed43ad889e422b64de44fd7688633dda",
            "b4aa757fd672669e95165c8aef649a4bdfc1a5905b4ba0182d65002abc0ad6a1",
        ],
        strict=True,
    )
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush counts so the conductor can supervise work without a model heartbeat."""
    print(
        f"[exp8590] phase={phase} completed={min(completed, 12)} pending={min(pending, 12)}",
        flush=True,
    )


def gate(path: Path, field: str, expected: Any, observed: Any, upstream: str = TASK) -> Json:
    """A missing resource belongs to this invocation even when a shared reader found it."""
    return dict(_gate(path, field, expected, observed, upstream))


def outcome(task: Json, primary: Json, raw: Path) -> Json:
    """Reuse original readers so new count receipts cannot clear old producer failures."""
    with patch.dict(old.PINS, PINS):
        value = dict(old.outcome(task, primary, raw))
    if value["verdict_class"] == "blocked":
        value["disposition"] = (
            "pre_gate_receipt" if not value["producer_executed"] else "external_blocked_producer"
        )
    return value


def slots(root: Path, raw: Path) -> tuple[list[Json], list[Json]]:
    """Resolve actual pre-gates through the reader; absent cascades remain absent."""
    tasks = methods.protocol()["task_map_v737"]
    if tasks[-1].get("requires") or tasks[-1].get("gated_on"):
        raise ValueError("unconditional_capstone")
    log = freeze(root / "ops/conductor-log.md", raw / "conductor")
    lines = Path(log["snapshot_path"]).read_text().splitlines() if log["exists"] else []
    rows, refs = [], [log]
    for index, task in enumerate(tasks[:-1]):
        progress("slot_before_" + task["id"], index, 11 - index)
        selected = _find_artifact_by_task_id(task["id"], root / "results")
        primary = freeze(selected or root / task["deliverable"], raw / "primaries" / str(index))
        summary = outcome(task, primary, raw) if primary["exists"] else None
        cascade = next(
            (
                s
                for s in lines
                if "GATE_BLOCK" in s and s.split("|")[2].strip() == task["title"][:50].strip()
            ),
            None,
        )
        rows.append(old.slot(task, primary, cascade, summary))
        refs += [primary, *rows[-1].get("evidence_references", [])]
        progress("slot_after_" + task["id"], index + 1, 10 - index)
    return rows, refs


def sample_worker(output: Path) -> int:
    """Small original reductions add no independent questions or timing measurements."""
    from carnot.reporting import arc_capability_panel_8537 as arc

    previous.sample_worker(output)
    diagnostic = read(reference(output))
    for number, suffix in [(8584, "resident_complete_cost"), (8588, "arc_capability_panel")]:
        path = ROOT / f"results/experiment_{number}_v737_{suffix}.json"
        value = read(reference(path))
        work = read(value["work_reference"])
        if number == 8584:
            row = value["paired_session_rows"][0]
            primitive = read(row["primitive_reference"])
            total = sum((s["end_ns"] - s["start_ns"]) / 1e9 for s in primitive["sessions"])
            diagnostic["resident_cost_sample"] = dict(
                source=reference(path),
                unit_id=row["unit_id"],
                recomputed_wall_s=total,
                matches_original=total == row["total_session_wall_s"],
                qualified=False,
            )
        else:
            observed, effects, exclusions = arc.reduce_episode(work["references"][0])
            expected = next(
                r for r in value["per_episode_rows"] if r["episode_id"] == observed["episode_id"]
            )
            diagnostic["current_arc"] = dict(
                source=reference(path),
                observed=observed,
                effect_count=len(effects),
                exclusion_count=len(exclusions),
                matches_original=observed == expected,
            )
        diagnostic["resident_cost_consumer" if number == 8584 else "current_arc_consumer"] = (
            reader_receipt(
                value["task_id"],
                path.parent,
                field="resident_cost_ready_score"
                if number == 8584
                else "arc_panel_rows_ready_score",
            )
        )
    path = ROOT / "results/experiment_8572_v736_resident_inference_qualification.json"
    work = read(read(reference(path))["primitive_reference"])
    clocks = work["primitive"]["clock_partition_receipt"]["sessions"][:2]
    diagnostic["resident_clock_sample"] = [
        dict(
            source=reference(path),
            attributed_ns=c["attributed_ns"],
            recomputed_ns=sum(i["end_ns"] - i["start_ns"] for i in c["intervals"]),
            matches_original=c["attributed_ns"]
            == sum(i["end_ns"] - i["start_ns"] for i in c["intervals"]),
        )
        for c in clocks
    ]
    atomic_json(output, diagnostic)
    progress("sample_worker_complete", 3, 0)
    return 0


def samples(raw: Path) -> tuple[Json, Json]:
    """A fresh bounded interpreter isolates historical operand allocations."""
    output = raw / "sample.json"
    receipt = child(
        "sample_operands",
        [sys.executable, "-u", str(ROOT / CLI), "--sample-output", str(output)],
        raw / "logs",
        deadline=120,
    )
    if not receipt["passed"]:
        raise ValueError("sample_child_failed")
    return read(reference(output)), receipt


def measure(root: Path, raw: Path) -> Json:
    """Authenticate resources and authority before independent bounded CPU reductions."""
    with patch.multiple(methods, OWNED=OWNED, TESTS=TESTS):
        work = dict(methods.measure(root, raw))
    for failure in work["precondition_failures"]:
        failure.update(upstream=TASK, upstream_id=TASK)
    work["slots"], refs = slots(ROOT, raw)
    work["source_refs"] += refs
    work["source_refs"] += [
        reference(ROOT / p)
        for p in [
            "ops/exclusion_manifest.yaml",
            "scripts/experiment_template.py",
            "docs/research-notes/v737-capstone.md",
        ]
    ]
    work["preconditions"].update(optional_cuda_required=False, optional_boards_required=False)
    progress("diagnostic_before", 11, 1)
    work["samples"], receipt = samples(raw / "samples")
    work["branch_receipts"] = [receipt]
    work["end_ns"] = time.monotonic_ns()
    work["duration_s"] = (work["end_ns"] - work["start_ns"]) / 1e9
    if work["duration_s"] >= 3600:
        raise TimeoutError("measurement_deadline")
    work["phase_spans"].append(dict(phase="capstone_aggregation", duration_s=work["duration_s"]))
    atomic_json(raw / "measurement.json", work)
    progress("diagnostic_after", 12, 0)
    return work
