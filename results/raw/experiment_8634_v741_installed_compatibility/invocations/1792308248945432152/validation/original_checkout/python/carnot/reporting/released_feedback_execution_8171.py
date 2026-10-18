"""REQ-REPORT-8171: freeze owned checks before measuring historical learning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import mkdtemp
import time
from typing import Any

from carnot.reporting import delayed_energy_execution_8143 as previous
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import released_feedback_learning_8171 as e

Json = dict[str, Any]
OWNED = [e.MODULE, e.CLI, e.RUNNER]


def manifest(private: Path, candidate: Path) -> Json:
    """Retain qualified commands with real paths and coverage of added statements."""
    specs = previous.manifest(private, candidate)
    replacements = dict(zip(previous.OWNED, OWNED, strict=True))
    replacements[previous.e.TEST] = e.TEST
    replacements[str(e.ROOT / previous.e.CLI)] = str(e.ROOT / e.CLI)
    for spec in specs["commands"] + specs["terminal_commands"]:
        spec["argv"] = [replacements.get(a, a) for a in spec["argv"]]
        spec["deadline_s"] = max(120, spec["deadline_s"])
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["deadline_s"] = 1200
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
        deadline_s=1200,
        classification="required",
    )
    return specs


def check(spec: Json, private: Path, raw: Path) -> Json:
    """Progress and log hashes distinguish real child exits from forced closure."""
    e.progress("before_subprocess_" + spec["name"])
    row = run_check(e.ROOT, spec, private, raw / "validation_logs", heartbeat_s=30)
    row["normal_exit"] = row["actual_exit"] >= 0 and not row.get("timed_out", False)
    e.progress("after_subprocess_" + spec["name"], int(row["passed"]))
    return row


def main(argv: list[str] | None = None) -> int:
    """Private CLI fixtures preserve historical primaries while exercising real paths."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261005"], default="20261005")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--seed-input", type=Path)
    parser.add_argument("--seed-output", type=Path)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--crash-slot", type=int, default=0)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "reduction_drift")
        return 0 if passed else 1
    if args.seed_input:
        inputs = json.loads(args.seed_input.read_text())
        state = json.loads(args.resume_state.read_text()) if args.resume_state else None
        e.run_seed(
            inputs["rows"],
            Path(inputs["label_path"]),
            inputs["seed"],
            args.seed_output,
            state=state,
            crash_slot=args.crash_slot,
        )
        return 0
    if args.worker_output:
        e.measure(args.root, args.worker_output.parent)
        return 0
    output = (args.fixture_output or args.output).absolute()
    fixture = args.fixture_output is not None
    if fixture and output.is_relative_to(e.ROOT / "results"):
        parser.error("fixture outputs must be private")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    private = Path(mkdtemp(prefix="carnot-8171-validation-"))
    candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
    specs = manifest(private, candidate)
    specs["restart_commands"] = e.restart_specs(raw)
    specs["measurement"]["argv"] += ["--root", str(args.root)]
    specs["measurement"]["argv"][specs["measurement"]["argv"].index("--worker-output") + 1] = str(
        raw / "measurement.json"
    )
    atomic_json(raw / "validation_commands.json", specs)
    if fixture:
        work = e.measure(args.root, raw, fixture=args.root == e.ROOT)
        receipts = [dict(name="private_fixture_normal_exit", passed=True, normal_exit=True)]
    else:
        receipts = [check(specs["measurement"], private, raw)]
        work = json.loads((raw / "measurement.json").read_text())
        receipts.extend(check(s, private, raw) for s in specs["commands"])
        work["repository_health"] = check(specs["repository_health"], private, raw)
    command_ref = dict(
        path=str(raw / "validation_commands.json"),
        sha256=sha256_file(raw / "validation_commands.json"),
    )
    work["raw_shard_hashes"].append(command_ref)
    work["validation_command_manifest"] = command_ref
    value = e.build(work, raw, receipts)
    atomic_json(candidate, value)
    reports = [] if fixture else [check(s, private, raw) for s in specs["terminal_commands"]]
    if any(not r["passed"] for r in reports):
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
