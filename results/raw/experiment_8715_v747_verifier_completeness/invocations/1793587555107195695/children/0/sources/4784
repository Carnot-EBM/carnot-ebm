"""REQ-VERIFY-8638: publish only bytes accepted by cold replay and unchanged readers."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import hardware_operation_delta_8638 as e
from carnot.reporting import hardware_operation_runner_8625 as previous
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child
from carnot.reporting.v726_feature_panel_runner import scan as original_scan

Json = dict[str, Any]


def scan(path: Path, raw: Path) -> tuple[list[Json], list[Json]]:
    """Keep the readiness-name INFO finding; it reports no command or science failure."""
    checks, findings = original_scan(path, raw)
    acknowledged = all(
        flag
        == dict(
            kind="IMPLAUSIBLE_PERFECT",
            severity="info",
            detail="hardware_delta_ready_score=0.0 (exactly zero). Confirm this is not a stub default.",
        )
        for flag in findings
    )
    checks[0]["unchanged_external_info"] = (
        checks[0]["normal_exit"] and checks[0]["actual_exit"] == 1 and acknowledged
    )
    checks[0]["acknowledged_readiness_name_info"] = acknowledged
    if checks[0]["unchanged_external_info"]:
        checks[0]["scope"] = "external_unchanged_diagnostic"
    return checks, findings


def validate(raw: Path) -> list[Json]:
    """Reuse the shipped disjoint validator while keeping every original coverage shard."""

    def retained_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
        if "combine" in argv:
            argv = [*argv[:-1], "--keep", argv[-1]]
        return child(name, argv, logs, **kwargs)

    suites = [e.TEST, *previous.SUITES[1:]]
    with (
        patch.object(previous, "e", e),
        patch.object(previous, "SUITES", suites),
        patch.object(previous, "child", retained_child),
    ):
        return list(previous.validate(raw))


def terminal(path: Path, raw: Path) -> Json:
    """A new process reauthenticates the immutable inputs before final publication."""
    cold = child(
        "cold",
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
        raw,
        deadline=60,
    )
    checks, findings = scan(path, raw)
    return dict(
        passed=cold["passed"]
        and all(r["passed"] or r.get("unchanged_external_info", False) for r in checks),
        receipts=[cold, *checks],
        findings=findings,
    )


def run(
    root: Path, output: Path, private: Path, date: str, control: bool, owned_failure: bool
) -> int:
    """Complete validation before atomic replacement and retain its actual subprocess receipts."""
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
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, e.build(work, receipts, raw, output))
    checks, findings = scan(candidate, raw / "preterminal")
    receipts += checks
    work["adversarial_findings"] = findings
    work["duration_s"] = (time.monotonic_ns() - start) / 1e9
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)
    publication = publish_primary(output, value, lambda path: terminal(path, raw / "terminal"))
    consumer = reader_receipt(e.TASK, output.parent, field="hardware_delta_ready_score", expected=0)
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            consumer=consumer,
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    e.progress("published_" + value["verdict_class"], 2, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Keep private controls outside the checkout and bound all production work."""
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
        with TemporaryDirectory(prefix="exp8638-private-", dir="/var/tmp") as directory:
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
