"""REQ-REPORT-8347: frozen bounded commands qualify consumers before publication."""

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

import coverage

from carnot.reporting import local_consumer_qualification_8347 as e
from carnot.reporting import local_qualification_execution_8333 as qualified
from carnot.reporting import v718_replay_runner as findings
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def cli() -> list[str]:
    """Measure real child entry while including its statements in owned coverage."""
    prefix = [str(e.ROOT / ".venv/bin/python"), "-u"]
    config = os.environ.get("COVERAGE_RCFILE")
    if config:
        prefix += ["-m", "coverage", "run", "--rcfile=" + config]
    return prefix + [str(e.ROOT / e.CLI)]


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze the failed cases first, then the exact consumer command and scoped tools."""
    with patch.object(qualified, "e", e):
        plan = qualified.manifest(private, candidate)
    consumers = plan["commands"][1]
    test = "tests/python/test_hard_exit_learning_qualification_8206.py"
    cases = [
        "test_readiness_and_failed_owned_check",
        "test_real_cli_replay_blocked_worker_and_guard",
        "test_coverage_summary_is_measured",
    ]
    first = dict(
        consumers,
        name="first_three_consumers",
        argv=consumers["argv"][:7]
        + ["--basetemp=" + str(private / "first-three")]
        + [test + "::" + c for c in cases],
    )
    plan["commands"].insert(0, first)
    for spec in plan["commands"]:
        if spec["name"].startswith("ruff"):
            spec["argv"].append(e.consumer.TEST)
    plan["commands"][-1]["argv"].append(e.consumer.TEST)
    plan["owned_child_environment"] = dict(
        unset=["COVERAGE_PROCESS_START"],
        reason="Invocation coverage must not become ambient coverage in consumer children.",
    )
    return plan


def check(spec: Json, raw: Path) -> Json:
    """Retain real exits, timing and both streams while polling every20 seconds."""
    with patch.dict(os.environ):
        os.environ.pop("COVERAGE_PROCESS_START", None)
        return child(
            spec["name"],
            spec["argv"],
            raw,
            deadline=spec["deadline_s"],
            expected=spec.get("expected_exit", 0),
            heartbeat=20,
        )


def retain_coverage(private: Path, raw: Path, work: Json) -> None:
    """Keep the measured report after private validation scratch disappears."""
    path = private / "coverage.json"
    if path.is_file():
        atomic_json(raw / "owned_coverage.json", json.loads(path.read_bytes()))
        work["owned_coverage_reference"] = e.base.reference(raw / "owned_coverage.json")


def publish(value: Json, output: Path, raw: Path) -> None:
    """Unchanged validators decide whether the checked bytes become the primary."""

    def validate(candidate: Path) -> Json:
        cold = check(
            dict(
                name="terminal_cold_replay",
                argv=cli() + ["--cold-replay", str(candidate)],
                deadline_s=120,
            ),
            raw / "terminal_logs",
        )
        found = findings.audit(candidate, raw / "terminal_findings", value["false_zero_control"])
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
            raw / "terminal_logs",
        )
        return dict(
            passed=all(r["passed"] for r in [cold, found["receipt"], rows]),
            checks=[cold, found["receipt"], rows],
            adversarial=found,
        )

    publication = publish_primary(output, value, validate)
    atomic_json(
        raw / "terminal_validation.json",
        dict(publication=publication, normal_process_completion=True),
    )
    e.progress("published", 1, 0)


def main(argv: list[str] | None = None) -> int:
    """Execute current qualification, private blocked controls or fresh cold replay."""
    e.progress("start")
    os.environ["PYTHONUNBUFFERED"] = "1"
    began = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261009"], default="20261009")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        return int(not e.replay(args.cold_replay))
    output = (args.fixture_output or args.output).absolute()
    if args.fixture_output and output.is_relative_to(e.ROOT / "results"):
        parser.error("private control output must stay outside results")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="carnot-8347-") as directory:
        private = Path(directory)
        plan = manifest(private, private / "candidate.json")
        atomic_json(raw / "execution_manifest.json", plan)
        work = e.measure(args.root, raw)
        receipts = []
        if not args.fixture_output:
            for index, spec in enumerate(plan["commands"]):
                if spec["name"] == "coverage_combine":
                    active = coverage.Coverage.current()
                    if active is not None:
                        for _ in range(2):
                            active.save()
                            shutil.copyfile(
                                active.get_data().data_filename(), private / ".coverage.invocation"
                            )
                receipts.append(check(spec, raw / "logs"))
                e.progress("validation", index + 1, len(plan["commands"]) - index - 1)
        retain_coverage(private, raw, work)
        work.update(
            invocation_argv=list(sys.argv if argv is None else [e.CLI, *argv]),
            execution_manifest_reference=e.base.reference(raw / "execution_manifest.json"),
            duration_s=time.monotonic() - began,
            phase_spans=[
                dict(
                    phase="historical_operand_and_consumer_qualification",
                    start_s=0,
                    duration_s=time.monotonic() - began,
                )
            ],
        )
        atomic_json(raw / "measurement.json", work)
        provisional = e.build(work, raw, receipts)
        for name, candidate, expected in [
            ("valid", provisional, 0),
            ("negative", {}, 1),
            ("rehashed", dict(provisional, local_kernel_ready_score=99), 1),
        ]:
            candidate.pop("reproducibility_checksum", None)
            candidate["reproducibility_checksum"] = canonical_hash(candidate)
            path = private / (name + ".json")
            atomic_json(path, candidate)
            atomic_json(raw / "measurement.json", work)
            receipts.append(
                check(
                    dict(
                        name=name + "_replay",
                        argv=cli() + ["--cold-replay", str(path)],
                        deadline_s=120,
                        expected_exit=expected,
                    ),
                    raw / "logs",
                )
            )
        value = e.build(work, raw, receipts)
        candidate = private / "candidate.json"
        atomic_json(candidate, value)
        found = findings.audit(candidate, raw / "finding_controls", value["false_zero_control"])
        work["finding_dispositions"] = found["dispositions"]
        receipts.append(found["receipt"])
        atomic_json(raw / "measurement.json", work)
        value = e.build(work, raw, receipts)
        publish(value, output, raw)
    return 0
