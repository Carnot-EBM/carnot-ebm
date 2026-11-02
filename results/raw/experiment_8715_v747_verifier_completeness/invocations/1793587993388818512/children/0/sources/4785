"""REQ-VERIFY-8651: retain actual child evidence and validate exact publication bytes."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from coverage import CoverageData

from carnot.reporting import hardware_operation_evidence_8651 as e
from carnot.reporting import hardware_operation_runner_8625 as previous
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v726_feature_panel_runner import scan as original_scan

Json = dict[str, Any]


def scan(path: Path, raw: Path) -> tuple[list[Json], list[Json]]:
    """INFO describes a concern to retain; command failures and serious findings still fail."""
    checks, findings = original_scan(path, raw)
    receipt = checks[0]
    receipt["passed"] = bool(
        receipt["normal_exit"]
        and receipt["actual_exit"] in {0, 1}
        and (receipt["actual_exit"] == 0 or findings)
        and all(f["severity"].lower() == "info" for f in findings)
    )
    receipt["severity_classification"] = [dict(f, retained=True) for f in findings]
    return checks, findings


def validate(raw: Path) -> list[Json]:
    """Give each child its own coverage path, then combine copies of retained originals."""
    folder = raw / "validation"
    retained, copies = folder / "retained", folder / "combine_copies"
    inventories = []

    def retained_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        if name.startswith("tests"):
            directory = folder / "children" / name
            directory.mkdir(parents=True, exist_ok=True)
            argv = [
                f"COVERAGE_FILE={directory / '.coverage'}" if a.startswith("COVERAGE_FILE=") else a
                for a in argv
            ]
        if name == "combine":
            copies.mkdir(parents=True, exist_ok=True)
            for original in retained.rglob(".coverage*"):
                shutil.copyfile(
                    original, copies / (".coverage." + original.parent.name + "." + original.name)
                )
            argv = [*argv[:-1], "--keep", str(copies)]
        receipt = child(name, argv, logs, **kwargs)
        if name.startswith("tests"):
            snapshots = retained / name
            snapshots.mkdir(parents=True, exist_ok=True)
            inventory = []
            for database in sorted((folder / "children" / name).glob(".coverage*")):
                snapshot = snapshots / database.name
                shutil.copyfile(database, snapshot)
                snapshot.chmod(0o444)
                data = CoverageData(basename=str(snapshot))
                data.read()
                inventory.append(
                    dict(
                        e.reference(snapshot),
                        measured_source_lines={
                            p: sorted(data.lines(p) or []) for p in sorted(data.measured_files())
                        },
                    )
                )
            atomic_json(snapshots / "inventory.json", dict(databases=inventory))
            inventories.append(e.reference(snapshots / "inventory.json"))
        return receipt

    suites = [e.TEST, "tests/python/test_hardware_operation_delta_8638.py", *previous.SUITES[1:]]
    with (
        patch.object(previous, "e", e),
        patch.object(previous, "SUITES", suites),
        patch.object(previous, "child", retained_child),
    ):
        receipts = list(previous.validate(raw))
    receipts.append(
        dict(name="original_coverage_retention", passed=bool(inventories), inventories=inventories)
    )
    return receipts


def terminal(path: Path, raw: Path) -> Json:
    """A new bounded process authenticates operands before the shared readers inspect bytes."""
    cold = child(
        "cold",
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
        raw,
        deadline=60,
    )
    checks, findings = scan(path, raw)
    return dict(
        passed=cold["passed"] and all(r["passed"] for r in checks),
        receipts=[cold, *checks],
        findings=findings,
    )


def run(
    root: Path, output: Path, private: Path, date: str, control: bool, owned_failure: bool
) -> int:
    """Retain failed owned checks as disqualified evidence; external absence remains blocked."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    start = time.monotonic_ns()
    work = e.measure(root, raw, private, date)
    e.progress("validation", 0, 1)
    receipts = (
        [
            child(
                "private_child",
                [
                    sys.executable,
                    "-u",
                    "-c",
                    "raise SystemExit(7)"
                    if owned_failure
                    else "print('actual private child',flush=True)",
                ],
                raw / "validation",
                deadline=30,
            )
        ]
        if control
        else validate(raw)
    )
    work["owned_coverage"] = next((r for r in receipts if r["name"] == "owned_coverage"), None)
    work["exact_union_receipt"] = next(
        (r for r in receipts if r["name"] == "exact_node_union"), None
    )
    work["phase_spans"] += [
        dict(phase=r["name"], duration_s=r["duration_s"]) for r in receipts if "duration_s" in r
    ]
    work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    work["work_path"] = str(raw / "preterminal_work.json")
    atomic_json(Path(work["work_path"]), work)
    Path(work["work_path"]).chmod(0o444)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, e.build(work, receipts, raw, output))
    checks, findings = scan(candidate, raw / "preterminal")
    receipts += checks
    work["adversarial_findings"] = findings
    work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    work["work_path"] = str(raw / "final_work.json")
    atomic_json(Path(work["work_path"]), work)
    Path(work["work_path"]).chmod(0o444)
    value = e.build(work, receipts, raw, output)
    publication = publish_primary(output, value, lambda p: terminal(p, raw / "terminal"))
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="hardware_mapping_ready_score",
        expected=value["hardware_mapping_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            consumer=consumer,
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    e.progress("published_" + value["verdict_class"], 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Private controls stay outside the checkout and all production work has a hard deadline."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261018")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--owned-failure", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if len(args.date) != 8:
            return 2
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        return 2
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    if args.private_control and args.output.resolve().is_relative_to(e.ROOT):
        parser.error("private output requires scratch outside checkout")
    timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8651-private-", dir="/var/tmp") as directory:
            return run(
                args.root,
                args.output.absolute(),
                Path(directory),
                args.date,
                args.private_control,
                args.owned_failure,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)
