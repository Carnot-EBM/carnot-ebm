"""REQ-REPORT-8138: freeze checks and publish only validated finite receipts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import mkdtemp
import time
from typing import Any

from carnot.reporting import methods_stream_execution_8111 as previous
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import learning_protocol_8138 as e

Json = dict[str, Any]
OWNED = [e.MODULE, e.CLI, e.RUNNER]


def manifest(private: Path, candidate: Path) -> Json:
    """Qualified supervisors retain exact commands and bound each terminal validator."""
    specs = previous.manifest(private, candidate)
    replacements = dict(zip(previous.OWNED, OWNED, strict=True))
    replacements[previous.e.TEST] = e.TEST
    replacements[str(e.ROOT / previous.e.CLI)] = str(e.ROOT / e.CLI)
    for spec in specs["commands"] + specs["terminal_commands"]:
        spec["argv"] = [replacements.get(a, a) for a in spec["argv"]]
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["deadline_s"] = 600
        if spec["name"] in ["cold_replay", "adversarial_verify", "strict_row_lint"]:
            spec["deadline_s"] = 300
    consumer = next(s for s in specs["commands"] if s["name"] == "consumer_and_E2E015_019")
    # The old methods tests read a retired V701 roadmap heading; keep their health result separate.
    consumer["argv"].remove("tests/python/test_development_methods_8098.py")
    specs["historical_consumer_health"] = dict(
        consumer,
        name="historical_methods_repository_health",
        classification="diagnostic",
        argv=consumer["argv"][
            : consumer["argv"].index("tests/python/test_learning_stream_capture_8102.py")
        ]
        + ["tests/python/test_development_methods_8098.py"],
    )
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in OWNED)
    )
    return specs


def check(spec: Json, private: Path) -> Json:
    """Keep validation logs private and record normal exits, hashes and durations."""
    e.progress("before_subprocess_" + spec["name"])
    row = run_check(e.ROOT, spec, private, private / "sealed_logs", heartbeat_s=30)
    row["normal_exit"] = row["actual_exit"] >= 0 and not row.get("timed_out", False)
    e.progress("after_subprocess_" + spec["name"], int(row["passed"]))
    return row


def main(argv: list[str] | None = None) -> int:
    """Script-path workers and cold replay work outside the checkout without PYTHONPATH."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261004"], default="20261004")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--mutation", choices=["", "transcript"], default="")
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "reduction_drift")
        return 0 if passed else 1
    if args.worker_output:
        e.measure(args.root, args.worker_output.parent)
        return 0
    output = (args.fixture_output or args.output).absolute()
    fixture = args.fixture_output is not None
    if fixture and output.is_relative_to(e.ROOT / "results"):
        parser.error("fixture outputs must be private")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    private = Path(mkdtemp(prefix="carnot-8138-validation-"))
    candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
    specs = manifest(private, candidate)
    atomic_json(raw / "validation_commands.json", specs)
    if fixture:
        work = e.measure(args.root, raw, fixture_mode=args.root == e.ROOT)
        receipts = [dict(name="private_fixture_execution", passed=True)]
    else:
        spec = dict(
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
        specs["measurement"] = spec
        atomic_json(raw / "validation_commands.json", specs)
        receipts = [check(spec, private)]
        work = json.loads((raw / "measurement.json").read_text())
        receipts.extend(check(s, private) for s in specs["commands"])
        work["repository_health"] = check(specs["repository_health"], private)
    command_ref = dict(
        path=str(raw / "validation_commands.json"),
        sha256=e.sha256_file(raw / "validation_commands.json"),
    )
    work["raw_shard_hashes"] = [*work["raw_shard_hashes"], command_ref]
    work["validation_command_manifest"] = command_ref
    if args.mutation:
        work["rows"][0]["numerator"] += 0.1
    value = e.build(work, raw, receipts)
    atomic_json(candidate, value)
    reports = [] if fixture else [check(s, private) for s in specs["terminal_commands"]]
    if fixture:
        reports = [
            dict(
                name="private_event_conformance",
                passed=work["full_size_validation_receipts"]["passed"],
            )
        ]
    if not all(r["passed"] for r in reports):
        atomic_json(raw / "failed_terminal_candidate.json", value)
    receipts.extend(reports)
    value = e.build(work, raw, receipts)
    publication = publish_primary(output, value, lambda p: dict(passed=e.replay(p), checks=reports))
    atomic_json(
        raw / "terminal_validation.json",
        dict(
            publication=publication,
            normal_process_exit=True,
            owned_checks_passed=value["required_checks_passed"],
        ),
    )
    e.progress("publication_complete", 1)
    return 0
