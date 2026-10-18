"""REQ-REPORT-8165: freeze owned checks before measuring zero-model methods."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import mkdtemp
import time
from typing import Any

from carnot.reporting import admission_horizon_execution_8152 as previous
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import learning_qualification_8165 as e

Json = dict[str, Any]
OWNED = [e.MODULE, e.RUNNER, e.CLI]


def manifest(private: Path, candidate: Path) -> Json:
    """Reuse qualified command supervision; measure coverage of new statements only."""
    specs = previous.manifest(private, candidate)
    replacements = dict(zip(previous.OWNED, OWNED, strict=True))
    replacements[previous.e.TEST] = e.TEST
    replacements[str(e.ROOT / previous.e.CLI)] = str(e.ROOT / e.CLI)
    for spec in specs["commands"] + specs["terminal_commands"]:
        spec["argv"] = [replacements.get(a, a) for a in spec["argv"]]
        if e.TEST in spec["argv"]:
            spec["argv"].append(
                e.legacy.TEST + "::test_exact_checksum_rejection"
                if spec["name"] == "owned_unit_and_private_CLI"
                else e.legacy.TEST
            )
        spec["deadline_s"] = max(spec["deadline_s"], 120)
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["deadline_s"] = 900
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in OWNED)
    )
    specs["measurement"] = dict(
        name="measurement",
        argv=[
            str(e.ROOT / ".venv/bin/python"),
            "-u",
            str(e.ROOT / e.CLI),
            "--worker-output",
            str(candidate.parent / "measurement.json"),
        ],
        expected_exit=0,
        deadline_s=240,
        classification="required",
    )
    return specs


def check(spec: Json, private: Path, raw: Path) -> Json:
    """Keep exact exits and durable log hashes, with visible child heartbeats."""
    e.progress("before_subprocess_" + spec["name"])
    row = run_check(e.ROOT, spec, private, raw / "validation_logs", heartbeat_s=30)
    row["normal_exit"] = row["actual_exit"] >= 0 and not row.get("timed_out", False)
    e.progress("after_subprocess_" + spec["name"], int(row["passed"]))
    return row


def main(argv: list[str] | None = None) -> int:
    """A thin script supports private success, external blocks and cold replay."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261005"], default="20261005")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
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
    private = Path(mkdtemp(prefix="carnot-8165-validation-"))
    candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
    specs = manifest(private, candidate)
    specs["measurement"]["argv"] += ["--root", str(args.root)]
    specs["measurement"]["argv"][specs["measurement"]["argv"].index("--worker-output") + 1] = str(
        raw / "measurement.json"
    )
    atomic_json(raw / "validation_commands.json", specs)
    if fixture:
        work = e.measure(args.root, raw, fixture_mode=args.root == e.ROOT)
        receipts = [dict(name="private_fixture_normal_exit", passed=True, normal_exit=True)]
    else:
        receipts = [check(specs["measurement"], private, raw)]
        work = json.loads((raw / "measurement.json").read_text())
        receipts.extend(check(s, private, raw) for s in specs["commands"])
        work["repository_health"] = check(specs["repository_health"], private, raw)
    work["validation_command_manifest"] = dict(
        path=str(raw / "validation_commands.json"),
        sha256=sha256_file(raw / "validation_commands.json"),
    )
    work["raw_shard_hashes"].append(work["validation_command_manifest"])
    value = e.build(work, raw, receipts)
    atomic_json(candidate, value)
    reports = [] if fixture else [check(s, private, raw) for s in specs["terminal_commands"]]
    if any(not r["passed"] for r in reports):
        atomic_json(raw / "failed_terminal_candidate.json", value)
    receipts.extend(reports)
    value = e.build(work, raw, receipts)

    def validate(path: Path) -> Json:
        checked = [] if fixture else [check(s, private, raw) for s in specs["terminal_commands"]]
        return dict(
            passed=e.replay(path) and all(r["passed"] and r["normal_exit"] for r in checked),
            checks=checked,
        )

    publication = publish_primary(output, value, validate)
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
