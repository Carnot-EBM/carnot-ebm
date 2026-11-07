"""REQ-VERIFY-8839: use scoped checks and publish only a checked terminal candidate."""

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from coverage import Coverage

from carnot.reporting import v755_branch_runner as scoped
from carnot.reporting import v756_capstone as c
from carnot.reporting.current_work_receipt import build_current_work_receipt
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v748_collection_contract import namespace

Json = dict[str, Any]
checks = namespace(scoped, "capstone8839_checks")
checks.c = c
original_check = checks.run_check


def run_check(
    name: str, argv: list[str], raw: Path, deadline: float = 120, scope: str = "owned"
) -> Json:
    """Freeze a larger test budget for repeated custody checks; heartbeats never renew it."""
    result: Json = original_check(
        name, argv, raw, 600 if name == "owned_tests" else deadline, scope
    )
    return result


checks.run_check = run_check


def run(output: Path, raw: Path, root: Path, run_date: str) -> int:
    """Owned failures clear readiness; repository health retains its independent outcome."""
    start = time.monotonic_ns()
    raw.mkdir(parents=True, exist_ok=True)
    tracer = Coverage(
        data_file=str(raw / ".coverage.parent"),
        include=[str(c.ROOT / p) for p in c.OWNED],
        config_file=False,
    )
    tracer.set_option("report:exclude_lines", [])
    tracer.start()
    work = c.freeze(root, raw, run_date)
    c.progress("validation_before")
    validation = checks.validate(raw, tracer)
    historical = checks.run_check(
        "historical_publication_gates",
        [sys.executable, "scripts/publication_gate.py", "--json"],
        raw / "checks",
        120,
    )
    validation["receipts"].append(historical)
    validation["historical_publication_gates"] = json.loads(
        Path(historical["stdout_path"]).read_bytes()
    )
    health = Path("/tmp/exp8839-global/receipt.json")
    if health.is_file():
        validation["repository_health"] = json.loads(health.read_bytes())
        validation["repository_health"]["log_reference"] = c.retain(
            Path(validation["repository_health"]["log_path"]), raw / "repository_health"
        )
    end = time.monotonic_ns()
    validation["current_work_receipt"] = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(owned_science_calls=0),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[dict(phase="custody_and_validation", start_s=0, end_s=(end - start) / 1e9)],
    )
    c.atomic_json(raw / "validation.json", validation)
    c.progress("validation_after")
    value = c.build(work, validation)
    c.progress("publication_before")
    try:
        receipt = publish_primary(
            output, value, lambda candidate: checks.terminal(candidate, raw / "terminal")
        )
    except ValueError as error:
        c.progress("publication_rejected_" + str(error))
        return 1
    c.atomic_json(raw / "publication.json", receipt)
    readers = reader_receipt(
        c.TASK,
        output.parent,
        field="reconciliation_ready_score",
        expected=value["reconciliation_ready_score"],
    )
    c.atomic_json(raw / "reader_receipt.json", readers)
    c.progress("publication_after", 13, 0)
    print(
        json.dumps(
            dict(
                path=str(output),
                verdict=value["honest_verdict"],
                reconciliation_ready_score=value["reconciliation_ready_score"],
            )
        ),
        flush=True,
    )
    return 0 if value["required_checks_passed"] and readers["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Private output and replay keep research execution separate from milestone activation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261105")
    parser.add_argument(
        "--output", "--private-output", type=Path, default=c.ROOT / "results" / (c.NAME + ".json")
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--root", type=Path, default=c.ROOT)
    parser.add_argument("--cold-replay", "--replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        result = c.replay(args.cold_replay)
        print(json.dumps(result), flush=True)
        return 0 if result["passed"] else 1
    if not re.fullmatch(r"\d{8}", args.date):
        print("invalid_date", flush=True)
        return 2
    raw = args.raw_dir or args.output.parent / "raw" / c.NAME / "invocations" / str(time.time_ns())
    return run(args.output.absolute(), raw.absolute(), args.root, args.date)
