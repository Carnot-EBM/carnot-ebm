"""REQ-REPORT-8306: bounded child validation precedes atomic primary publication.

The existing supervisor owns child process groups, deadlines and heartbeat logs.
Private fixture routes exercise publication without gaining scientific credit.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as execution
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import local_update_isolation_8306 as kernel
from carnot.verify.hard_exit_learning_qualification_8206 import run_check
from carnot.reporting import local_update_isolation_8306 as e

Json = dict[str, Any]
BASE_MANIFEST = execution.manifest


def manifest(private: Path, candidate: Path) -> Json:
    """Reuse qualified command recipes while restricting coverage to new files."""
    with patch.object(execution, "e", e), patch.object(execution, "OWNED", e.OWNED):
        specs = BASE_MANIFEST(private, candidate)
    specs.pop("repository_health")
    specs["commands"][0]["argv"].remove("-s")
    specs["commands"][0]["argv"].append("--basetemp=" + str(private / "owned-tests"))
    specs["commands"][0]["deadline_s"] = 480
    specs["commands"][1]["argv"] = [
        str(e.ROOT / ".venv/bin/pytest"),
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "--basetemp=" + str(private / "consumer-tests"),
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_experiment_7942_v689_sentence_labels.py",
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_hard_exit_learning_qualification_8206.py",
        "tests/python/test_experiment_7425_v651_spline_prototype.py",
    ]
    specs["commands"][1]["deadline_s"] = 900
    specs["commands"][-1]["argv"].insert(2, "--files")
    specs["terminal_commands"].extend(
        [
            dict(
                name="negative_replay",
                argv=[
                    str(e.ROOT / ".venv/bin/python"),
                    str(e.ROOT / e.CLI),
                    "--cold-replay",
                    str(private / "negative.json"),
                ],
                expected_exit=1,
                deadline_s=30,
            ),
            dict(
                name="rehashed_tamper_replay",
                argv=[
                    str(e.ROOT / ".venv/bin/python"),
                    str(e.ROOT / e.CLI),
                    "--cold-replay",
                    str(private / "rehashed.json"),
                ],
                expected_exit=1,
                deadline_s=30,
            ),
        ]
    )
    specs["measurement"] = dict(
        name="measurement",
        argv=[
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            str(e.ROOT / e.CLI),
            "--worker-output",
            str(private / "unused.json"),
        ],
        deadline_s=600,
        expected_exit=0,
        classification="required",
    )
    return specs


def publish(
    value: Json, output: Path, private: Path, raw: Path, terminal: list[Json], fixture: bool
) -> None:
    """Unchanged publication checks can recover failure as a disqualified record."""
    with patch.object(execution, "e", e):
        execution.publish(value, output, private, raw, terminal, fixture)


def main(argv: list[str] | None = None) -> int:
    """Freeze commands, run a real worker, validate, then expose checked bytes."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261008"], default="20261008")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--trajectory", type=int, default=0)
    parser.add_argument("--cohort-count", type=int, default=48)
    parser.add_argument("--arm", choices=kernel.ARMS, default="indexed")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--crash", type=int, default=-1)
    args = parser.parse_args(argv)
    if args.worker:
        plan = json.loads(args.worker.read_text())
        if args.checkpoint is None:
            parser.error("worker requires checkpoint")
        if args.trajectory == -1:
            for n, trajectory in enumerate(plan["trajectories"][: args.cohort_count]):
                kernel.execute(
                    trajectory,
                    args.arm,
                    args.checkpoint / (trajectory["id"] + ".json"),
                    pause=args.crash,
                )
                e.progress("cohort_recovery", n + 1, args.cohort_count - n - 1)
            if args.crash != -1:
                import coverage

                current = coverage.Coverage.current()
                if current is not None:
                    current.save()
                e.progress("cohort_kill_boundary", args.cohort_count, 0)
                os.kill(os.getpid(), signal.SIGKILL)
        else:
            kernel.execute(
                plan["trajectories"][args.trajectory], args.arm, args.checkpoint, args.crash
            )
        return 0
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "reduction_drift")
        return 0 if passed else 1
    if args.worker_output:
        e.measure(args.root, args.worker_output.parent)
        return 0
    fixture = args.fixture_output is not None
    output = (args.fixture_output or args.output).absolute()
    if fixture and output.is_relative_to(e.ROOT / "results"):
        parser.error("fixture outputs must be private")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    with TemporaryDirectory(prefix="carnot-8306-") as directory:
        private = Path(directory)
        candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
        specs = manifest(private, candidate)
        specs["measurement"]["argv"] = [
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            str(e.ROOT / e.CLI),
            "--root",
            str(args.root),
            "--worker-output",
            str(raw / "measurement.json"),
        ]
        atomic_json(raw / "validation_commands.json", specs)
        e.progress("before_measurement_subprocess")
        if fixture:
            work = e.measure(args.root, raw, fixture=True)
            work["fixture"] = True
            atomic_json(raw / "measurement.json", work)
            receipts = [dict(name="private_fixture_normal_exit", passed=True)]
        else:
            receipts = [
                run_check(e.ROOT, specs["measurement"], private, raw / "logs", heartbeat_s=20)
            ]
            work = json.loads((raw / "measurement.json").read_text())
            for spec in specs["commands"]:
                e.progress(
                    "before_subprocess_" + spec["name"],
                    len(receipts) - 1,
                    len(specs["commands"]) - len(receipts) + 1,
                )
                receipts.append(run_check(e.ROOT, spec, private, raw / "logs", heartbeat_s=20))
                e.progress(
                    "after_subprocess_" + spec["name"],
                    len(receipts) - 1,
                    len(specs["commands"]) - len(receipts) + 1,
                )
            report = private / "coverage.json"
            if report.is_file():
                atomic_json(raw / "owned_coverage.json", json.loads(report.read_text()))
                work["owned_coverage_reference"] = e.reference(raw / "owned_coverage.json")
                atomic_json(raw / "measurement.json", work)
        e.progress("after_measurement_subprocess")
        work["invocation_argv"] = list(sys.argv if argv is None else [e.CLI, *argv])
        work["duration_s"] = time.monotonic() - began
        atomic_json(raw / "measurement.json", work)
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        value = e.build(work, raw, receipts)
        atomic_json(private / "negative.json", {})
        tamper = dict(value, local_kernel_ready_score=2)
        tamper.pop("reproducibility_checksum")
        tamper["reproducibility_checksum"] = canonical_hash(tamper)
        atomic_json(private / "rehashed.json", tamper)
        with patch.object(execution, "run_check", run_check):
            publish(value, output, private, raw, specs["terminal_commands"], fixture)
    return 0
