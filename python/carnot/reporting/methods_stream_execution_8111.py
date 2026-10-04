"""REQ-REPORT-8111: validate normally exited work before exposing one primary.

Private validation children cannot alter historical results. Long child waits
use the existing bounded supervisor with real heartbeat and log receipts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import methods_stream_custody_8111 as e

Json = dict[str, Any]
OWNED = [e.MODULE, e.CLI, e.RUNNER]


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze tools, include paths and deadlines before any measurement opens."""
    py, cov, pytest, ruff, mypy = [
        str(e.ROOT / ".venv/bin" / n) for n in ["python", "coverage", "pytest", "ruff", "mypy"]
    ]
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in OWNED)
    )
    rc = "--rcfile=" + str(config)
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    commands = [
        (
            "owned_unit_and_private_CLI",
            [
                "/usr/bin/env",
                "COVERAGE_RCFILE=" + str(config),
                cov,
                "run",
                rc,
                "-m",
                "pytest",
                *common,
                "-s",
                e.TEST,
            ],
            180,
        ),
        (
            "consumer_and_E2E015_019",
            [
                pytest,
                *common,
                "tests/python/test_development_methods_8098.py",
                "tests/python/test_learning_stream_capture_8102.py",
                "tests/python/test_primary_publication_7928.py",
                "tests/python/test_source_boundary_7852.py",
                "tests/python/test_experiment_7942_v689_sentence_labels.py",
            ],
            180,
        ),
        ("coverage_combine", [cov, "combine", rc], 30),
        ("coverage_report", [cov, "report", rc, "--show-missing", "--fail-under=100"], 30),
        ("coverage_json", [cov, "json", rc, "-o", str(private / "coverage.json")], 30),
        ("ruff_check", [ruff, "check", *OWNED, e.TEST], 30),
        ("ruff_format", [ruff, "format", "--check", *OWNED, e.TEST], 30),
        ("strict_mypy", [mypy, "--strict", "--follow-imports=silent", *OWNED], 60),
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", e.TEST], 30),
    ]
    terminal = [
        ("cold_replay", [py, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(candidate)], 60),
        ("adversarial_verify", [py, "scripts/adversarial_verify.py", "--json", str(candidate)], 60),
        (
            "strict_row_lint",
            [py, "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            60,
        ),
    ]

    def pack(rows: list[tuple[str, list[str], int]]) -> list[Json]:
        return [
            dict(name=n, argv=a, deadline_s=t, expected_exit=0, classification="required")
            for n, a, t in rows
        ]

    return dict(
        commands=pack(commands),
        terminal_commands=pack(terminal),
        repository_health=dict(
            name="repository_full_suite",
            argv=[pytest, "tests/python", "-q"],
            deadline_s=120,
            expected_exit=0,
            classification="diagnostic",
        ),
    )


def publish(
    value: Json, output: Path, private: Path, raw: Path, terminal: list[Json], fixture: bool
) -> None:
    """Publication validators bind exact candidate bytes after each normal exit."""
    reports: list[Json] = []

    def validate(candidate: Path) -> Json:
        if fixture:
            return dict(passed=e.replay(candidate), checks=[dict(name="private_cold_replay")])
        for spec in terminal:
            e.progress("before_subprocess_" + spec["name"])
            reports.append(run_check(e.ROOT, spec, private, raw / "terminal_logs", heartbeat_s=20))
            e.progress(
                "after_subprocess_" + spec["name"], len(reports), len(terminal) - len(reports)
            )
        return dict(passed=all(r["passed"] for r in reports), checks=reports)

    try:
        publication = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        atomic_json(raw / "failed_terminal_candidate.json", value)
        atomic_json(raw / "failed_terminal_report.json", dict(checks=reports))
        receipts = value["validation_receipts"] + reports
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        work = json.loads((raw / "measurement.json").read_text())
        value = e.build(work, raw, receipts, fixture=fixture)
        publication = publish_primary(
            output,
            value,
            lambda p: dict(
                passed=e.replay(p),
                checks=[dict(name="disqualified_custody_replay")],
                failed_owned_checks=reports,
            ),
        )
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            normal_process_exit=True,
            owned_checks_passed=value["required_checks_passed"],
        ),
    )
    e.progress("publication_complete", 1)


def main(argv: list[str] | None = None) -> int:
    """Execute a no-model measurement worker, private fixtures or cold replay."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261004"], default="20261004")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--stream-path", type=Path)
    parser.add_argument(
        "--mutation", choices=["", "labels", "roles", "slots", "source"], default=""
    )
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "reduction_drift")
        return 0 if passed else 1
    if args.worker_output:
        e.measure(args.root, args.worker_output.parent, stream_path=args.stream_path)
        return 0
    fixture = args.fixture_output is not None
    output = (args.fixture_output or args.output).absolute()
    if fixture and output.is_relative_to(e.ROOT / "results"):
        parser.error("fixture outputs must be private")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="carnot-8111-") as directory:
        private = Path(directory)
        candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
        specs = manifest(private, candidate)
        specs["measurement"] = dict(
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
            deadline_s=180,
            expected_exit=0,
            classification="required",
        )
        atomic_json(raw / "validation_commands.json", specs)
        if fixture:
            work = e.measure(
                args.root, raw, fixture=True, stream_path=args.stream_path, mutation=args.mutation
            )
            receipts = [dict(name="private_measurement_normal_exit", passed=True)]
        else:
            command = specs["measurement"]
            e.progress("before_measurement_subprocess")
            receipts = [run_check(e.ROOT, command, private, raw / "logs", heartbeat_s=20)]
            e.progress("after_measurement_subprocess")
            work = json.loads((raw / "measurement.json").read_text())
            for spec in specs["commands"]:
                e.progress("before_subprocess_" + spec["name"])
                receipts.append(run_check(e.ROOT, spec, private, raw / "logs", heartbeat_s=20))
                e.progress(
                    "after_subprocess_" + spec["name"],
                    len(receipts) - 1,
                    len(specs["commands"]) - len(receipts) + 1,
                )
            e.progress("before_repository_health_subprocess")
            work["global_health"] = run_check(
                e.ROOT, specs["repository_health"], private, raw / "global_health", heartbeat_s=20
            )
            e.progress("after_repository_health_subprocess")
            atomic_json(raw / "measurement.json", work)
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        value = e.build(work, raw, receipts, fixture=fixture)
        publish(value, output, private, raw, specs["terminal_commands"], fixture)
    return 0
