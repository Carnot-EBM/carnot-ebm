"""REQ-REPORT-8143: validate finite causal evidence before primary publication."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from tempfile import mkdtemp
import time
from typing import Any

from carnot.reporting import learning_protocol_execution_8138 as previous
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary
from carnot.verify import delayed_energy_memory_8143 as e

Json = dict[str, Any]
OWNED = [e.MODULE, e.CLI, e.RUNNER]


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze exact owned commands using the qualified bounded supervisor."""
    specs = previous.manifest(private, candidate)
    replacements = dict(zip(previous.OWNED, OWNED, strict=True))
    replacements[previous.e.TEST] = e.TEST
    replacements[str(e.ROOT / previous.e.CLI)] = str(e.ROOT / e.CLI)
    for spec in specs["commands"] + specs["terminal_commands"]:
        spec["argv"] = [replacements.get(a, a) for a in spec["argv"]]
    (private / "coverage.ini").write_text(
        "[run]\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in OWNED)
    )
    return specs


def check(spec: Json, private: Path) -> Json:
    """Normal child exits and private log hashes make validation reviewable."""
    e.progress("before_subprocess_" + spec["name"])
    row: Json = previous.run_check(e.ROOT, spec, private, private / "sealed_logs", heartbeat_s=30)
    row["normal_exit"] = row["actual_exit"] >= 0 and not row.get("timed_out", False)
    e.progress("after_subprocess_" + spec["name"], int(row["passed"]))
    return row


def main(argv: list[str] | None = None) -> int:
    """Outside-checkout workers preserve source identity across a real restart."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261004"], default="20261004")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--fixture-mode", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--mutation", action="store_true")
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
        e.measure(args.root, args.worker_output.parent, fixture=args.fixture_mode)
        return 0
    output = (args.fixture_output or args.output).absolute()
    fixture = args.fixture_output is not None
    if fixture and output.is_relative_to(e.ROOT / "results"):
        parser.error("fixture outputs must be private")
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    private = Path(mkdtemp(prefix="carnot-8143-validation-"))
    candidate = output.parent / "raw" / output.stem / "terminal_candidate.json"
    specs = manifest(private, candidate)
    specs["restart_commands"] = e.restart_specs(raw)
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
        work = e.measure(args.root, raw, fixture=args.root == e.ROOT)
        receipts = [dict(name="private_fixture_execution", passed=True)]
    else:
        receipts = [check(specs["measurement"], private)]
        work = json.loads((raw / "measurement.json").read_text())
        receipts.extend(check(s, private) for s in specs["commands"])
        work["repository_health"] = check(specs["repository_health"], private)
    command_ref = dict(
        path=str(raw / "validation_commands.json"),
        sha256=e.sha256_file(raw / "validation_commands.json"),
    )
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
        e.engine.shard(raw / "historical", json.loads(output.read_text()))
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
