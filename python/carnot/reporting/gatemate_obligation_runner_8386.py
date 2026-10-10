"""REQ-VERIFY-8386: actual bounded children qualify a read-only continuity receipt."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import gatemate_obligation_delta_8386 as e
from carnot.reporting import kv260_local_cost_boundary_8315 as base
from carnot.reporting import v717_contract_runner as qualified
from carnot.reporting import v718_replay_runner as terminal
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Reuse private authority/publication checks and cover only newly owned statements."""
    with patch.object(qualified, "m", e):
        plan = list(qualified.manifest(private))
    plan[0]["deadline"] = 600
    plan[1]["argv"].insert(
        -1, "tests/python/test_gatemate_missing_evidence_8372.py::test_private_receipt_consumers"
    )
    plan[1]["argv"].insert(
        -1,
        "tests/python/test_gatemate_missing_evidence_8372.py::test_authenticated_failure_rejections",
    )
    plan.append(
        dict(
            name="repository_health_once",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(private / "global-suite"),
            ],
            expected=0,
            deadline=900,
            scope="global",
        )
    )
    return plan


def run(
    root: Path,
    output: Path,
    private: Path,
    supplied: list[Path] | None = None,
    *,
    control: bool = False,
) -> int:
    """Owned validation failures disqualify; unavailable external evidence stays blocked."""
    start = time.monotonic_ns()
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    plan = manifest(private)
    if control:
        plan = [
            dict(
                name="private_child",
                argv=[sys.executable, "-u", "-c", "print('private child')"],
                expected=0,
                deadline=10,
                scope="owned",
            )
        ]
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=plan,
            task_cap_s=4800,
            private_scratch=str(private),
            heartbeat_s=30,
            evidence_scan_cap_s=300,
            parent_memory_budget_bytes=4 * 1024**3,
            no_model_load=True,
            private_control=control,
            cold_controls=[
                "valid",
                "negative",
                "rehashed_tamper",
                "missing_input",
                "deliberate_child_error",
            ],
        ),
    )
    e.progress("preconditions_before", 0, 1)
    with patch.object(qualified, "m", e):
        failures = qualified.preflight(plan)
    probe = private / "scratch_probe"
    probe.write_bytes(b"private-disk-backed-scratch")
    mem = next(
        line
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )
    mounts = [line.split() for line in Path("/proc/mounts").read_text().splitlines()]
    mount = max(
        (r for r in mounts if private.resolve().is_relative_to(r[1])), key=lambda r: len(r[1])
    )
    checks = [
        e.gate(
            private, "private_scratch", True, probe.read_bytes() == b"private-disk-backed-scratch"
        ),
        e.gate(private, "disk_backed_scratch", True, mount[2] not in {"tmpfs", "ramfs"}),
        e.gate(
            private,
            "available_disk_at_least_1GiB",
            True,
            shutil.disk_usage(private).free >= 1024**3,
        ),
        e.gate(
            private, "available_memory_at_least_1GiB", True, int(mem.split()[1]) * 1024 >= 1024**3
        ),
        e.gate(
            private,
            "planned_children_within_task_cap",
            True,
            sum(r["deadline"] for r in plan) + 660 < 4800,
        ),
    ]
    failures.extend(c for c in checks if not c["passed"])
    e.progress("preconditions_after", 1, 0)
    work = e.measure(root, raw, supplied)
    work["invocation_argv"] = list(sys.argv)
    work.update(preflight_checks=checks + failures, preflight_passed=not failures)
    receipts = execute(plan, raw / "checks") if not failures else []
    if (private / "coverage.json").is_file():
        atomic_json(
            raw / "owned_coverage.json", json.loads((private / "coverage.json").read_bytes())
        )
        work["owned_coverage_reference"] = base.reference(raw / "owned_coverage.json")
    work["execution_manifest_reference"] = base.reference(raw / "execution_manifest.json")
    work["started_monotonic_ns"], work["ended_monotonic_ns"] = start, time.monotonic_ns()
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)
    with patch.object(qualified, "m", e):
        receipts += qualified.controls(value, raw)
    receipts.append(
        child(
            "cold_missing_input",
            [
                sys.executable,
                "-u",
                str(e.ROOT / e.CLI),
                "--cold-replay",
                str(raw / "absent-input.json"),
            ],
            raw / "cold",
            expected=1,
            deadline=60,
            heartbeat=20,
        )
    )
    receipts.append(
        child(
            "deliberate_child_error",
            [sys.executable, "-u", "-c", "raise SystemExit(7)"],
            raw / "cold",
            expected=7,
            deadline=10,
            heartbeat=20,
        )
    )
    atomic_json(raw / "audit_candidate.json", e.build(work, receipts, raw, output))
    e.progress("terminal_checks_before", 0, 1)
    with patch.object(terminal, "e", e):
        audit = terminal.audit(raw / "audit_candidate.json", raw / "audit", {})
        work["adversarial_findings"] = audit["findings"]
        receipts.append(audit["receipt"])
        atomic_json(raw / "measurement.json", work)
        terminal.publish(e.build(work, receipts, raw, output), output, raw)
    e.progress("terminal_checks_after", 1, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Private controls exercise the real CLI while keeping their results outside the checkout."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261010"], default="20261010")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--supplied-receipt", type=Path, action="append")
    parser.add_argument("--private-e2e", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_e2e and output.is_relative_to(e.ROOT):
        parser.error("private controls require output outside the repository")
    with TemporaryDirectory(prefix="exp8386-owned-", dir="/var/tmp") as directory:
        return run(
            args.root, output, Path(directory), args.supplied_receipt, control=args.private_e2e
        )
