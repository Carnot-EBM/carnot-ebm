"""REQ-VERIFY-8416: bounded real children qualify exact terminal bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v725_contract_methods as e
from carnot.reporting import v717_contract_runner as shipped
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Reuse scoped serial tests, subprocess coverage, E2E-018 and strict checks without global health."""
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    with patch.object(shipped, "m", e):
        plan = shipped.manifest(private)
    next(p for p in plan if p["name"] == "coverage_combine")["argv"].append("--keep")
    return list(plan)


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Capture each real child and validation shard under this invocation before publishing."""
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    plan = (
        [
            dict(
                name="private_child",
                argv=[sys.executable, "-u", "-c", "print('actual private child',flush=True)"],
                deadline=10,
                expected=0,
                scope="owned",
            )
        ]
        if control
        else manifest(private)
    )
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=plan,
            owned=e.OWNED,
            task_cap_s=4800,
            validation_reserve_s=1200,
            child_poll_s=30,
            global_health="not_run",
        ),
    )
    work = e.measure(root, raw)
    receipts = execute(plan, raw / "validation")
    if (private / "coverage.json").is_file():
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
        shards = raw / "coverage_shards"
        shards.mkdir()
        for path in private.glob(".coverage*"):
            shutil.copyfile(path, shards / path.name)
        work["coverage_shards"] = [e.reference(path) for path in sorted(shards.iterdir())]
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"] += [dict(name=r["name"], duration_s=r["duration_s"]) for r in receipts]
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)
    with patch.object(shipped, "m", e):
        receipts.extend(shipped.controls(value, raw))
    for name, args, expected in [
        ("missing", ["--cold-replay", str(raw / "missing.json")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(
            child(
                name,
                [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
                raw / "controls",
                expected=expected,
                deadline=60,
            )
        )
    candidate = raw / "preterminal.json"
    value = e.build(work, receipts, raw, output)
    atomic_json(candidate, value)
    scan = child(
        "adversarial_preterminal",
        [sys.executable, "-u", "scripts/adversarial_verify.py", "--json", str(candidate)],
        raw / "validators",
        deadline=60,
    )
    reports = json.loads(Path(scan["stdout_path"]).read_bytes())["reports"]
    work["adversarial_findings"] = [flag for report in reports for flag in report.get("flags", [])]
    receipts.append(scan)
    receipts.append(
        child(
            "all_findings_accounted",
            [
                sys.executable,
                "-u",
                "-c",
                "import json,pathlib; reports=json.loads(pathlib.Path("
                + repr(scan["stdout_path"])
                + ").read_bytes())['reports']; assert not any(r.get('flags') for r in reports); print('all findings retained; none unresolved')",
            ],
            raw / "validators",
            deadline=30,
        )
    )
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"] = [
        work["phase_spans"][0],
        *[dict(name=row["name"], duration_s=row["duration_s"]) for row in receipts],
    ]
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)
    with patch.object(shipped, "m", e):
        shipped.publish(value, output, raw)
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="required_checks_passed",
        expected=value["required_checks_passed"],
    )
    terminal = Path(value["terminal_validation_sidecar_path"])
    sidecar = json.loads(terminal.read_bytes())
    atomic_json(
        terminal,
        dict(sidecar, consumer=consumer, preterminal_findings=work["adversarial_findings"]),
    )
    e.progress("finished", 13, 0)
    return 0


def main(argv: list[str] | None = None) -> int:
    """The dated runner owns its cap; private controls never write historical or active repository files."""
    e.progress("start_no_model_load_LLM_loads_0_generations_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261011"], default="20261011")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private control outputs must be outside the repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8416-owned-", dir="/var/tmp") as directory:
            return run(args.root, output, Path(directory), control=args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
