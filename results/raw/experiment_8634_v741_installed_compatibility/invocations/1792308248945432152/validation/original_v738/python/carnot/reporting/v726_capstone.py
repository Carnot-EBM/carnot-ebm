"""REQ-VERIFY-8442: real private checks precede atomic publication of terminal bytes."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_capstone_evidence as e
from carnot.reporting import v717_contract_runner as validation
from carnot.reporting import v718_replay_runner as terminal
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v709_execution import child, execute
from carnot.reporting.v726_capstone_report import build
from carnot.reporting.gatemate_continuity_8400 import budget

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Shipped serial validation creates genuine Coverage.py subprocess shards."""
    with patch.object(validation, "m", e):
        plan = list(validation.manifest(private))
    next(p for p in plan if p["name"] == "owned_tests")["deadline"] = 900
    next(p for p in plan if p["name"] == "coverage_combine")["argv"].append("--keep")
    plan.append(
        dict(
            name="private_E2E021",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(private / "E2E021"),
            ],
            expected=0,
            deadline=300,
            scope="owned",
        )
    )
    return plan


def controls(value: Json, raw: Path) -> list[Json]:
    """Actual children must reject missing operands, deliberate errors and rehashed summaries."""
    receipts = []
    for name, mutation, expected in [("valid", None, 0), ("rehashed", "task_count", 1)]:
        changed = deepcopy(value)
        if mutation:
            changed[mutation] = 13
            changed["reproducibility_checksum"] = e.checksum(changed)
        path = raw / (name + ".json")
        atomic_json(path, changed)
        receipts.append(
            child(
                "cold_" + name,
                [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
                raw / "logs",
                expected=expected,
                deadline=180,
                heartbeat=30,
            )
        )
    for name, args, expected in [
        ("missing", ["--cold-replay", str(raw / "absent.json")], 1),
        ("deliberate_error", ["--deliberate-error"], 7),
    ]:
        receipts.append(
            child(
                "cold_" + name,
                [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
                raw / "logs",
                expected=expected,
                deadline=60,
                heartbeat=30,
            )
        )
    return receipts


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, failed: bool = False
) -> int:
    """External absence blocks science while owned failed checks disqualify this producer."""
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    plan = manifest(private)
    if control:
        plan = [
            dict(
                name="actual_private_child",
                argv=[
                    sys.executable,
                    "-u",
                    "-c",
                    f"print('private mechanical child',flush=True);raise SystemExit({int(failed)})",
                ],
                expected=0,
                deadline=30,
                scope="owned",
            )
        ]
    atomic_json(
        raw / "command_manifest.json",
        dict(
            commands=plan,
            private_control=control,
            task_cap_s=4800,
            validation_reserve_s=900,
            private_scratch=str(private),
        ),
    )
    work = e.measure(root, raw, private)
    work["refs"].append(e.freeze(raw / "command_manifest.json", raw / "commands"))
    e.progress("validation_before", 0, len(plan))
    receipts = execute(plan, raw / "validation")
    target = raw / "coverage_shards"
    target.mkdir(parents=True, exist_ok=True)
    for source in private.glob(".coverage*"):
        shutil.copyfile(source, target / source.name)
    if (private / "coverage.json").is_file():
        shutil.copyfile(private / "coverage.json", target / "coverage.json")
    shutil.copyfile(private / "coverage.ini", target / "coverage.ini")
    work["coverage_shards"] = [e.reference(p) for p in sorted(target.iterdir())]
    publication = child(
        "publication_gate",
        [sys.executable, "-u", "scripts/publication_gate.py", "--json"],
        raw / "publication",
        deadline=90,
        heartbeat=30,
        scope="publication",
    )
    work["publication"] = json.loads(Path(publication["stdout_path"]).read_bytes())
    work["publication_receipt"] = publication
    work["ended_monotonic_ns"] = time.monotonic_ns()
    atomic_json(raw / "measurement.json", work)
    receipts.extend(controls(build(work, receipts, raw, output), raw / "controls"))
    candidate = raw / "audit_candidate.json"
    atomic_json(candidate, build(work, receipts, raw, output))
    with (
        patch.object(terminal, "e", e),
        patch.object(e, "build", build, create=True),
        patch.object(terminal, "child", terminal_child),
    ):
        audit = terminal.audit(candidate, raw / "audit", {})
        work["adversarial_findings"] = audit["findings"]
        receipts.append(audit["receipt"])
        work["ended_monotonic_ns"] = time.monotonic_ns()
        work["phase_spans"].append(
            dict(
                name="owned_validation",
                started_monotonic_ns=work["phase_spans"][0]["ended_monotonic_ns"],
                ended_monotonic_ns=work["ended_monotonic_ns"],
            )
        )
        atomic_json(raw / "measurement.json", work)
        terminal.publish(build(work, receipts, raw, output), output, raw)
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="required_checks_passed",
        expected=build(work, receipts, raw, output)["required_checks_passed"],
    )
    atomic_json(raw / "consumer_receipt.json", consumer)
    e.progress("terminal_publication", 14, 0)
    return 0


def terminal_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """A complete cold replay includes all bounded branches within its reserved deadline."""
    if name == "terminal_cold":
        kwargs["deadline"] = 240
    return dict(child(name, argv, logs, **kwargs))


def main(argv: list[str] | None = None) -> int:
    """The CLI owns private disk scratch and rejects dates outside this task contract."""
    e.progress("start_no_model_load_current_LLM_invocations_0", 0, 14)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261012"], default="20261012")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--failed-child", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker_request:
        return e.worker_process(args.worker_request, args.worker_output)
    if args.deliberate_error:
        e.progress("deliberate_error")
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private controls must publish outside the repository")
    with budget(4800), TemporaryDirectory(prefix="exp8442-owned-", dir="/var/tmp") as directory:
        return run(
            args.root,
            output,
            Path(directory),
            control=args.private_control,
            failed=args.failed_child,
        )
