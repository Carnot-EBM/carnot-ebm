"""REQ-VERIFY-8430: bounded private children qualify one atomic terminal artifact.

Each real command keeps its exit, duration and log hashes. Private control mode
exists for CLI tests so those tests cannot recursively start another test run.
The production command runs the complete scoped validation plan before publish.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_feature_panel as e
from carnot.reporting import v717_contract_runner as shipped
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Reuse genuine serial child coverage and add the task's frozen audit E2E."""
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    with patch.object(shipped, "m", e):
        plan = list(shipped.manifest(private))
    next(row for row in plan if row["name"] == "coverage_combine")["argv"].append("--keep")
    next(row for row in plan if row["name"] == "owned_tests")["deadline"] = 300
    plan.insert(
        2,
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
                "--basetemp=" + str(private / "e2e021"),
            ],
            expected=0,
            deadline=240,
            scope="owned",
        ),
    )
    return plan


def scan(candidate: Path, raw: Path) -> tuple[list[Json], list[Json]]:
    """Retain every unchanged consumer finding; exit zero alone cannot qualify them."""
    receipts = execute(
        [
            dict(
                name=name,
                argv=[sys.executable, "-u", script, *args, str(candidate)],
                expected=0,
                deadline=60,
                scope="owned",
            )
            for name, script, args in [
                ("adversarial", "scripts/adversarial_verify.py", ["--json"]),
                ("strict_rows", "scripts/verdict_row_consistency_lint.py", ["--strict"]),
            ]
        ],
        raw,
    )
    report = json.loads(Path(receipts[0]["stdout_path"]).read_bytes())
    findings = [flag for row in report["reports"] for flag in row.get("flags", [])]
    if findings:
        receipts[0]["passed"] = False
    return receipts, findings


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Freeze operands, execute controls, and expose only validated terminal bytes."""
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    work = e.measure(root, raw)
    plan = (
        [
            dict(
                name="private_child",
                argv=[sys.executable, "-u", "-c", "print('actual private control',flush=True)"],
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
            commands=plan,
            owned=e.OWNED,
            task_cap_s=4800,
            validation_reserve_s=900,
            child_poll_s=30,
            full_repository_suite="not_run",
        ),
    )
    receipts = execute(plan, raw / "validation")
    if (private / "coverage.json").is_file():
        shards = raw / "coverage_shards"
        shards.mkdir()
        for source in private.glob(".coverage*"):
            shutil.copyfile(source, shards / source.name)
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
        work["coverage_shards"] = [e.reference(p) for p in sorted(shards.iterdir())]
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, e.build(work, receipts, raw, output))
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent.json")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(
            child(name, [*argv, *args], raw / "controls", expected=expected, deadline=60)
        )
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["feature_panel_ready_score"] = 999
    changed["reproducibility_checksum"] = e.checksum(changed)
    tamper = raw / "rehashed_tamper.json"
    atomic_json(tamper, changed)
    receipts.append(
        child(
            "rehashed_tamper",
            [*argv, "--cold-replay", str(tamper)],
            raw / "controls",
            expected=1,
            deadline=60,
        )
    )
    atomic_json(candidate, e.build(work, receipts, raw, output))
    checks, findings = scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work["adversarial_findings"] = findings
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"] += [dict(name=r["name"], duration_s=r["duration_s"]) for r in receipts]
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, raw, output)

    def validator(path: Path) -> Json:
        """Qualification binds actual scanner reports and cold reduction to final bytes."""
        checks, flags = scan(path, raw / "terminal_validators")
        return dict(
            passed=e.replay(path) and all(c["passed"] for c in checks),
            findings=flags,
            receipts=checks,
        )

    publication = publish_primary(output, value, validator)
    consumer = reader_receipt(
        e.TASK,
        output.parent,
        field="feature_panel_ready_score",
        expected=value["feature_panel_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            consumer=consumer,
            primary_sha256=publication["primary_sha256"],
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    e.progress("finished", value["completed_count"], value["censored_count"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """The date and private-output guard bind this invocation to its authorized task."""
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
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8430-owned-", dir="/var/tmp") as directory:
            return run(args.root, output, Path(directory), control=args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
