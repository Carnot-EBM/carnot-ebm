"""REQ-REPORT-8144: frozen private validation precedes atomic publication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import mkdtemp
import time
from typing import Any

from carnot.reporting import delayed_energy_execution_8143 as previous
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary
from carnot.verify import learning_audit_8144 as e

Json = dict[str, Any]
OWNED = [e.MODULE, e.CLI, e.RUNNER]


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze qualified tools and exact coverage includes before opening inputs."""
    specs = previous.manifest(private, candidate)
    replacements = dict(zip(previous.OWNED, OWNED, strict=True))
    replacements[previous.e.TEST] = e.TEST
    replacements[str(e.ROOT / previous.e.CLI)] = str(e.ROOT / e.CLI)
    for spec in specs["commands"] + specs["terminal_commands"]:
        spec["argv"] = [replacements.get(a, a) for a in spec["argv"]]
        spec["deadline_s"] = (
            600 if spec["name"] == "owned_unit_and_private_CLI" else max(spec["deadline_s"], 60)
        )
    (private / "coverage.ini").write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in OWNED)
    )
    return specs


def check(spec: Json, private: Path) -> Json:
    """Real bounded child exits bind durations and immutable private log hashes."""
    e.progress("before_subprocess_" + spec["name"])
    row: Json = previous.previous.run_check(
        e.ROOT, spec, private, private / "sealed_logs", heartbeat_s=30
    )
    row["normal_exit"] = row["actual_exit"] >= 0 and not row.get("timed_out", False)
    e.progress("after_subprocess_" + spec["name"], int(row["passed"]))
    return row


def main(argv: list[str] | None = None) -> int:
    """The thin script runs from any directory without inherited package paths."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261005"], default="20261005")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--mutation", action="store_true")
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
    private = Path(mkdtemp(prefix="carnot-8144-validation-"))
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
        deadline_s=1200,
        expected_exit=0,
        classification="required",
    )
    atomic_json(raw / "validation_commands.json", specs)
    if fixture:
        work = e.measure(args.root, raw, fixture=True)
        receipts = [dict(name="private_fixture_execution", passed=True)]
    else:
        receipts = [check(specs["measurement"], private)]
        work = json.loads((raw / "measurement.json").read_text())
        receipts.extend(check(s, private) for s in specs["commands"])
        work["repository_health"] = check(specs["repository_health"], private)
    command_ref = e.reference(raw / "validation_commands.json")
    work["raw_shard_hashes"].append(command_ref)
    work["validation_command_manifest"] = command_ref
    if args.mutation and work["rows"]:
        work["rows"][0]["numerator"] += 0.1
    value = e.build(work, raw, receipts)
    atomic_json(candidate, value)
    reports = [] if fixture else [check(s, private) for s in specs["terminal_commands"]]
    if not all(r["passed"] for r in reports):
        atomic_json(raw / "failed_terminal_candidate.json", value)
    receipts.extend(reports)
    value = e.build(work, raw, receipts)
    if output.exists():
        atomic_json(raw / "historical_primary.json", json.loads(output.read_text()))
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
