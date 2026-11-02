"""REQ-REPORT-8578: original seals separate scheduling, qualification and benefit."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v735_capstone_evidence as old
from carnot.reporting import v736_contract_methods as methods
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v709_execution import child
from scripts.conductor_gates import _find_artifact_by_task_id

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK, MILESTONE = "experiment_8578_v736_capstone", "exp8578-capstone", "2026.10.736"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v736_capstone_8578.py"
TESTS = [TEST]
OWNED = [f"python/carnot/reporting/v736_capstone{s}.py" for s in ["_evidence", "_report", ""]] + [
    CLI
]
MODEL_SPECS: list[Json] = []
__all__ = ["methods", "old"]
reference, read, checksum, atomic_json, gate = (
    methods.reference,
    methods.read,
    methods.checksum,
    methods.atomic_json,
    methods.gate,
)
freeze = old.freeze
PINS = dict(
    zip(
        [8565, 8566, 8567, 8568, 8569, 8571, 8572, 8573, 8574, 8575, 8577],
        [
            "66843de62020d69730b2796f0966ca2038d36ea49e02c337c6dd0071f69a0dd4",
            "ef6064315aee6e7f2ad32fff2d4163075ce852d54b82236ae9b342480aa4041b",
            "ca6b5a53c8f4e0de8eb2f3dfd01ec1d08c462d50eac19dde904275c9c8b17a86",
            "cf4b256742a74783617ce692415bf8a4ae3a5abd966c22564eccaa5b73742e3a",
            "d820b8cd42f58e40a548295c0302acdc34096d992b85c8bcd5546d1045929aa9",
            "0fc4bc1e2a5f65960292af3f6ecbea9255d7387946e9047d41b554df980336f3",
            "12adfee3ca0fe07bf305438484eab8362ff573f715c8379715f4c1399da0d3db",
            "1e123a804e9d17f85121f2caa7a99159d1acd4b82972e00dc0e9ba9e63852f10",
            "9f87b02237c8cf6047fe294dbb438b8b0d2681628dbd54e02f95cbe153e18e6b",
            "a5b3dbaa803fa0c51f49b8cb5c719d0e9e5ea7aeae96902aba467e17a6d8cce2",
            "2ac5a35abcbdf4d78462368a1b733add80d416cc0e1a133585af9223b63cbda9",
        ],
        strict=True,
    )
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so a conductor can supervise aggregation without model activity."""
    print(f"[exp8578] phase={phase} completed={completed} pending={pending}", flush=True)


def outcome(task: Json, primary: Json, raw: Path) -> Json:
    """Reuse original terminal readers while keeping external rejection distinct from owned failure."""
    with patch.dict(old.PINS, PINS):
        value = dict(old.outcome(task, primary, raw))
    if value["verdict_class"] == "blocked":
        value["disposition"] = (
            "pre_gate_receipt" if not value["producer_executed"] else "external_blocked_producer"
        )
    return value


def slots(root: Path, raw: Path) -> tuple[list[Json], list[Json]]:
    """Resolve real pre-gates through the shipped reader; cascade artifacts remain absent."""
    tasks = methods.protocol()["task_map_v736"]
    log = freeze(root / "ops/conductor-log.md", raw / "conductor")
    lines = Path(log["snapshot_path"]).read_text().splitlines() if log["exists"] else []
    rows, refs = [], [log]
    for index, task in enumerate(tasks[:-1]):
        progress("slot_before_" + task["id"], index, 13 - index)
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
        progress("slot_after_" + task["id"], index + 1, 12 - index)
    return rows, refs


def sample_worker(output: Path) -> int:
    """Replay small original operands while retaining the retired ARC consumer rejection."""
    from carnot.reporting.v734_capstone_samples import samples as original_samples
    from carnot.reporting import arc_panel_replay_8561 as arc

    progress("sample_worker_before", 0, 3)
    diagnostic = original_samples()
    try:
        consumed = arc.panel_consumer(ROOT / "results" / (arc.NAME + ".json"))
    except StopIteration:
        consumed = dict(
            passed=False,
            missing_reason="Original exp8562 consumer task retired from active V736 authority",
            status="blocked",
        )
    diagnostic["arc"]["actual_panel_consumer"] = consumed
    atomic_json(output, diagnostic)
    progress("sample_worker_after", 3, 0)
    return 0


def samples(raw: Path) -> tuple[Json, Json]:
    """Two resident clocks augment small learning and ARC diagnostics, never independence."""
    output = raw / "sample.json"
    receipt = child(
        "sample_operands",
        [sys.executable, "-u", str(ROOT / CLI), "--sample-output", str(output)],
        raw / "logs",
        deadline=120,
    )
    if not receipt["passed"]:
        raise ValueError("sample_child_failed")
    diagnostic = read(reference(output))
    path = ROOT / "results/experiment_8572_v736_resident_inference_qualification.json"
    value = read(reference(path))
    work = read(value["primitive_reference"])
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
    diagnostic["resident_consumer"] = reader_receipt(
        "exp8572-resident-inference-qualification",
        path.parent,
        field="resident_inference_ready_score",
    )
    diagnostic["resident_complete_cost"] = dict(
        measured=False,
        missing_reason="Exp8573 pre-gated; qualification clocks are not a serving benchmark",
    )
    return diagnostic, receipt


def measure(root: Path, raw: Path) -> Json:
    """Authenticate resources and immutable original families before bounded CPU reconciliation."""
    with patch.multiple(methods, OWNED=OWNED, TESTS=TESTS):
        work = dict(methods.measure(root, raw))
    work["slots"], refs = slots(ROOT, raw)
    work["source_refs"] += refs
    work["source_refs"] += [
        reference(ROOT / p)
        for p in [
            "ops/exclusion_manifest.yaml",
            "scripts/experiment_template.py",
            "docs/research-notes/v736-capstone.md",
        ]
    ]
    work["preconditions"].update(optional_cuda_required=False, optional_boards_required=False)
    progress("diagnostic_before", 13, 1)
    work["samples"], receipt = samples(raw / "samples")
    work["branch_receipts"] = [receipt]
    work["end_ns"] = time.monotonic_ns()
    work["duration_s"] = (work["end_ns"] - work["start_ns"]) / 1e9
    if work["duration_s"] >= 3600:
        raise TimeoutError("measurement_deadline")
    work["phase_spans"].append(dict(phase="capstone_aggregation", duration_s=work["duration_s"]))
    atomic_json(raw / "measurement.json", work)
    progress("diagnostic_after", 14, 0)
    return work
