"""REQ-VERIFY-8613: bounded private checks precede exact-byte atomic publication."""

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

from carnot.reporting import hardware_operation_boundary_8613 as e
from carnot.reporting import nine_input_board_runner_8538 as plans
from carnot.reporting import v726_feature_panel_runner as scanners
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.hardware_operation_boundary_8563 import node_union
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]
SUITES = [
    e.TEST,
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
    "tests/python/test_primary_publication_7928.py",
]


def manifest(private: Path) -> list[Json]:
    """Use collection helpers, never an old experiment's validation controller."""
    with patch.multiple(plans, e=e, SUITES=SUITES):
        result = list(plans.manifest(private))
    for row in result:
        row["argv"] = ["-vv" if token == "-q" else token for token in row["argv"]]
    return result


def run(
    root: Path, output: Path, private: Path, *, control: bool, owned_failure: bool, date: str
) -> int:
    """Keep real failed children and downstream findings in the terminal record."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = e.measure(root, raw, private, date)
    plan = (
        [
            dict(
                name="private_child",
                argv=[
                    sys.executable,
                    "-u",
                    "-c",
                    "raise SystemExit(7)"
                    if owned_failure
                    else "print('actual private child',flush=True)",
                ],
                expected=0,
                deadline=30,
                scope="owned",
            )
        ]
        if control
        else manifest(private)
    )
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=plan, owned=e.OWNED, task_cap_s=4800, validation_reserve_s=900, child_poll_s=30
        ),
    )
    receipts = execute(plan, raw / "validation")
    if (private / "coverage.json").is_file():
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
        shards = raw / "coverage_shards"
        shards.mkdir()
        for source in private.glob(".coverage*"):
            shutil.copyfile(source, shards / source.name)
        work["owned_coverage"] = json.loads((raw / "coverage.json").read_bytes())["totals"]
    work["exact_union_receipt"] = (
        node_union(raw, receipts)
        if not control
        else dict(passed=True, scope="private_control_no_scientific_qualification")
    )
    work["refs"].append(e.reference(raw / "execution_manifest.json"))
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, e.build(work, receipts, raw, output))
    checks, findings = scanners.scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work["adversarial_findings"] = findings
    work["phase_spans"] += [dict(phase=r["name"], duration_s=r["duration_s"]) for r in receipts]
    work["duration_s"] += sum(r["duration_s"] for r in receipts)
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)

    def validate(path: Path) -> Json:
        """Cold reduction and both unchanged validators inspect the bytes being published."""
        cold = child(
            "terminal_cold",
            [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
            raw / "terminal_validators",
            deadline=60,
        )
        checked, flags = scanners.scan(path, raw / "terminal_validators")
        return dict(
            passed=cold["passed"] and all(r["passed"] for r in checked) and not flags,
            findings=flags,
            receipts=[cold, *checked],
        )

    publication = publish_primary(output, value, validate)
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="board_boundary_recorded_score",
        expected=value["board_boundary_recorded_score"],
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
    e.progress(
        "published_" + value["verdict_class"], value["completed_count"], value["censored_count"]
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Private controls stay outside the checkout; runtime dates bind each invocation."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261017")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--owned-failure", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    try:
        if len(args.date) != 8:
            return 2
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        return 2
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    if args.private_control and args.output.resolve().is_relative_to(e.ROOT):
        parser.error("private output requires disk scratch outside checkout")
    timer = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8613-private-", dir="/var/tmp") as directory:
            return run(
                args.root,
                args.output.absolute(),
                Path(directory),
                control=args.private_control,
                owned_failure=args.owned_failure,
                date=args.date,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *timer)
