"""REQ-REPORT-8116: publish only normally exited, owned-validated memory evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v686_contract_validation import run_check
from carnot.reporting import methods_stream_execution_8111 as previous
from carnot.verify import independent_online_memory_8116 as e

Json = dict[str, Any]
OWNED = [e.MODULE, e.RUNNER, e.CLI]


def manifest(private: Path, candidate: Path) -> Json:
    """Reuse qualified bounded validation with this task's exact code and tests."""
    result = previous.manifest(private, candidate)
    replacements = dict(zip(previous.OWNED, OWNED, strict=True))
    replacements[previous.e.TEST] = e.TEST
    replacements[str(e.ROOT / previous.e.CLI)] = str(e.ROOT / e.CLI)
    for spec in result["commands"] + result["terminal_commands"]:
        spec["argv"] = [replacements.get(a, a) for a in spec["argv"]]
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["deadline_s"] = 360
        if spec["name"] == "cold_replay":
            spec["deadline_s"] = 120
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in OWNED)
    )
    return result


def publish(
    value: Json, output: Path, private: Path, raw: Path, specs: list[Json], fixture: bool
) -> None:
    """Terminal validators bind exact bytes; owned failure preserves its candidate."""
    reports: list[Json] = []

    def validate(candidate: Path) -> Json:
        if fixture:
            return dict(passed=e.replay(candidate))
        for spec in specs:
            e.progress("before_subprocess_" + spec["name"])
            reports.append(run_check(e.ROOT, spec, private, raw / "terminal_logs", heartbeat_s=20))
            reports[-1]["normal_exit"] = reports[-1]["actual_exit"] >= 0 and not reports[-1].get(
                "timed_out", False
            )
            e.progress("after_subprocess_" + spec["name"], len(reports), len(specs) - len(reports))
        return dict(passed=all(r["passed"] for r in reports), checks=reports)

    try:
        publication = publish_primary(output, value, validate)
    except ValueError as error:
        if str(error) != "candidate_rejected":
            raise
        atomic_json(raw / "failed_terminal_candidate.json", value)
        atomic_json(raw / "failed_terminal_report.json", dict(checks=reports))
        receipts = value["validation_receipts"] + reports
        for receipt in receipts:
            receipt["normal_exit"] = receipt.get("actual_exit", 0) >= 0 and not receipt.get(
                "timed_out", False
            )
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        work = json.loads((raw / "measurement.json").read_text())
        value = e.build(work, raw, receipts, fixture=fixture)
        publication = publish_primary(
            output, value, lambda p: dict(passed=e.replay(p), failed_owned_checks=reports)
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
    """Use a child for measurement so normal exit is evidence, not an assumption."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261004"], default="20261004")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--mutation", choices=["", "future-label"], default="")
    args = parser.parse_args(argv)
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
    with TemporaryDirectory(prefix="carnot-8116-") as directory:
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
            deadline_s=240,
            expected_exit=0,
            classification="required",
        )
        atomic_json(raw / "validation_commands.json", specs)
        if fixture:
            work = e.measure(args.root, raw, fixture=True, seeds=[101], mutation=args.mutation)
            receipts = [dict(name="private_measurement_normal_exit", passed=True)]
        else:
            e.progress("before_measurement_subprocess")
            receipts = [
                run_check(e.ROOT, specs["measurement"], private, raw / "logs", heartbeat_s=20)
            ]
            e.progress("after_measurement_subprocess")
            work = json.loads((raw / "measurement.json").read_text())
            for index, spec in enumerate(specs["commands"]):
                e.progress(
                    "before_subprocess_" + spec["name"], index, len(specs["commands"]) - index
                )
                receipts.append(run_check(e.ROOT, spec, private, raw / "logs", heartbeat_s=20))
                e.progress(
                    "after_subprocess_" + spec["name"],
                    index + 1,
                    len(specs["commands"]) - index - 1,
                )
            e.progress("before_repository_health_subprocess")
            work["global_health"] = run_check(
                e.ROOT, specs["repository_health"], private, raw / "global_health", heartbeat_s=20
            )
            e.progress("after_repository_health_subprocess")
            atomic_json(raw / "measurement.json", work)
        for receipt in receipts:
            receipt["normal_exit"] = receipt.get("actual_exit", 0) >= 0 and not receipt.get(
                "timed_out", False
            )
        atomic_json(raw / "validation_receipts.json", dict(rows=receipts))
        value = e.build(work, raw, receipts, fixture=fixture)
        publish(value, output, private, raw, specs["terminal_commands"], fixture)
    return 0
