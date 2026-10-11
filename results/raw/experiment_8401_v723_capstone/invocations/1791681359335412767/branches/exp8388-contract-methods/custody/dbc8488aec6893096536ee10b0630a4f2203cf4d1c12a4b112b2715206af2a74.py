"""REQ-REPORT-8333: bounded validation precedes atomic terminal publication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import local_qualification_8333 as e
from carnot.reporting import methods_stream_execution_8111 as qualified
from carnot.reporting import v718_replay_runner as findings
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.verify import local_update_isolation_8306 as k

Json = dict[str, Any]
BASE_MANIFEST = qualified.manifest


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze execution recipes separately from unchanged scientific parameters."""
    with patch.object(qualified, "e", e), patch.object(qualified, "OWNED", e.OWNED):
        plan = BASE_MANIFEST(private, candidate)
    plan.pop("repository_health")
    plan["commands"][0]["argv"].remove("-s")
    plan["commands"][0]["argv"].append("--basetemp=" + str(private / "owned-tests"))
    plan["commands"][0]["deadline_s"] = 600
    plan["commands"][1]["name"] = "consumer_and_E2E020_021"
    plan["commands"][1]["argv"] = [
        str(e.ROOT / ".venv/bin/pytest"),
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "--basetemp=" + str(private / "consumer-tests"),
        "tests/python/test_local_update_isolation_8306.py",
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_hard_exit_learning_qualification_8206.py",
        "tests/python/test_restricted_decision_audit_8210.py",
    ]
    plan["commands"][1]["deadline_s"] = 600
    plan["commands"][-1]["argv"].insert(2, "--files")
    return plan


def check(spec: Json, raw: Path) -> Json:
    """Keep argv, both output streams, clocks and exits under the existing harness."""
    return child(
        spec["name"],
        spec["argv"],
        raw,
        deadline=spec["deadline_s"],
        expected=spec.get("expected_exit", 0),
        heartbeat=20,
    )


def publish(value: Json, output: Path, raw: Path) -> None:
    """Preserve every failed terminal check in a validated disqualification receipt."""
    attempts: list[Json] = []

    def validate(candidate: Path) -> Json:
        cold = check(
            dict(
                name="cold_replay",
                argv=[
                    str(e.ROOT / ".venv/bin/python"),
                    "-u",
                    str(e.ROOT / e.CLI),
                    "--cold-replay",
                    str(candidate),
                ],
                deadline_s=60,
            ),
            raw / "terminal_logs" / str(len(attempts)),
        )
        found = findings.audit(
            candidate, raw / "terminal_findings" / str(len(attempts)), value["false_zero_control"]
        )
        rows = check(
            dict(
                name="strict_row_lint",
                argv=[
                    str(e.ROOT / ".venv/bin/python"),
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ],
                deadline_s=60,
            ),
            raw / "terminal_logs" / str(len(attempts)),
        )
        report = dict(
            passed=all(r["passed"] for r in [cold, found["receipt"], rows]),
            checks=[cold, found["receipt"], rows],
            adversarial=found,
        )
        attempts.append(report)
        return report

    try:
        publication = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        atomic_json(raw / "rejected_candidate.json", value)
        work = json.loads((raw / "measurement.json").read_bytes())
        value = e.build(work, raw, value["validation_receipts"] + attempts[0]["checks"])
        publication = publish_primary(output, value, validate)
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            attempts=attempts,
            checks=attempts[-1]["checks"],
            normal_process_completion=True,
        ),
    )
    e.progress("published", 1, 0)


def main(argv: list[str] | None = None) -> int:
    """Run measured workers and recovery children through a single direct CLI."""
    e.progress("start")
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261009"], default="20261009")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cohort-count", type=int, default=48)
    parser.add_argument("--crash", type=int, choices=[-1, 31, 63], default=-1)
    args = parser.parse_args(argv)
    if args.worker:
        if args.checkpoint is None:
            parser.error("worker requires checkpoint")
        protocol = json.loads(args.worker.read_bytes())
        for index, trajectory in enumerate(protocol["trajectories"][: args.cohort_count]):
            k.execute(
                trajectory,
                "indexed",
                args.checkpoint / (trajectory["id"] + ".json"),
                pause=args.crash,
            )
            e.progress("worker_recovery", index + 1, args.cohort_count - index - 1)
        if args.crash >= 0:
            import coverage

            measured = coverage.Coverage.current()
            e.progress("durable_kill", args.cohort_count, 0)
            # Save inside the call after its line is traced, because SIGKILL
            # cannot flush coverage after the actual kill has happened.
            os.kill(
                os.getpid(),
                (measured.save() or signal.SIGKILL) if measured is not None else signal.SIGKILL,
            )
        return 0
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "reduction_drift")
        return int(not passed)
    if args.worker_output:
        e.measure(args.root, args.worker_output.parent)
        return 0
    output = (args.fixture_output or args.output).absolute()
    if args.fixture_output and output.is_relative_to(e.ROOT / "results"):
        parser.error("fixture outputs must be private")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    with TemporaryDirectory(prefix="carnot-8333-") as directory:
        private = Path(directory)
        plan = manifest(private, raw / "candidate.json")
        plan["measurement"] = dict(
            name="measurement",
            argv=[
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                str(e.ROOT / e.CLI),
                "--root",
                str(args.root),
                "--worker-output",
                str(raw / "measurement.json"),
            ],
            deadline_s=600,
            expected_exit=0,
        )
        atomic_json(raw / "execution_manifest.json", plan)
        if args.fixture_output:
            work = e.measure(args.root, raw, fixture=True)
            receipts = [dict(name="private_constructed_control", passed=True)]
        else:
            receipts = [check(plan["measurement"], raw / "logs")]
            work = json.loads((raw / "measurement.json").read_bytes())
            for index, spec in enumerate(plan["commands"]):
                receipts.append(check(spec, raw / "logs"))
                e.progress("owned_checks", index + 1, len(plan["commands"]) - index - 1)
        coverage_path = private / "coverage.json"
        if coverage_path.is_file():
            atomic_json(raw / "owned_coverage.json", json.loads(coverage_path.read_bytes()))
            work["owned_coverage_reference"] = e.reference(raw / "owned_coverage.json")
        work.update(
            invocation_argv=list(sys.argv if argv is None else [e.CLI, *argv]),
            duration_s=time.monotonic() - began,
            execution_manifest_reference=e.reference(raw / "execution_manifest.json"),
        )
        atomic_json(raw / "measurement.json", work)
        provisional = e.build(work, raw, receipts)
        for name, candidate in [
            ("negative", {}),
            ("rehashed", dict(provisional, local_kernel_ready_score=99)),
        ]:
            candidate.pop("reproducibility_checksum", None)
            candidate["reproducibility_checksum"] = canonical_hash(candidate)
            path = private / (name + ".json")
            atomic_json(path, candidate)
            receipts.append(
                check(
                    dict(
                        name=name + "_replay",
                        argv=[
                            str(e.ROOT / ".venv/bin/python"),
                            str(e.ROOT / e.CLI),
                            "--cold-replay",
                            str(path),
                        ],
                        deadline_s=60,
                        expected_exit=1,
                    ),
                    raw / "logs",
                )
            )
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        value = e.build(work, raw, receipts)
        publish(value, output, raw)
    return 0
