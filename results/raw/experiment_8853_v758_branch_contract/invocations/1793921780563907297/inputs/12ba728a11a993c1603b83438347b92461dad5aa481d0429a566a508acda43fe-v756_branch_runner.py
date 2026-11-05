"""REQ-VERIFY-8827: qualify measured original execution and current owned checks."""

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any
from unittest.mock import patch

from coverage import Coverage

from carnot.reporting import v755_branch_runner as shared
from carnot.reporting import v756_branch_contract as c
from carnot.reporting import v756_branch_qualification as qualification
from carnot.reporting.primary_publication import publish_primary, reader_receipt

Json = dict[str, Any]


def validate(raw: Path, tracer: Coverage, original: Json) -> Json:
    """Reuse scoped tests, linters and private E2E; original evidence stays separate."""
    with patch.object(shared, "c", c):
        result = shared.validate(raw, tracer)
    result["original"] = original
    result["receipts"] += original["receipts"]
    result["references"] = (
        [original[k] for k in ["report_reference", "data_reference"]]
        + original["fixture_references"]
        + original["code_references"]
        + [c.reference(raw / "original/qualification.json")]
    )
    return result


def terminal(candidate: Path, raw: Path) -> Json:
    """Unchanged scanners and an outside-checkout interpreter inspect the same bytes."""
    with patch.object(shared, "c", c):
        return shared.terminal(candidate, raw)


def run(output: Path, raw: Path, root: Path, run_date: str) -> int:
    """Only accepted bytes are published; failed checks retain disqualified evidence."""
    raw.mkdir(parents=True, exist_ok=True)
    tracer = Coverage(
        data_file=str(raw / ".coverage.parent"),
        include=[str(c.ROOT / p) for p in c.OWNED],
        config_file=False,
    )
    tracer.set_option("report:exclude_lines", [])
    tracer.start()
    work = c.freeze(root, raw, run_date)
    original = qualification.qualify(raw / "original")
    c.progress("owned_validation_before")
    validation = validate(raw, tracer, original)
    c.atomic_json(raw / "validation.json", validation)
    c.progress("owned_validation_after")
    work["phase_spans"] = [
        dict(phase="operand_custody", start_s=0, end_s=(work["end_ns"] - work["start_ns"]) / 1e9),
        dict(
            phase="original_and_owned_validation",
            start_s=(work["end_ns"] - work["start_ns"]) / 1e9,
            end_s=(time.monotonic_ns() - work["start_ns"]) / 1e9,
        ),
    ]
    work["end_ns"] = work["start_ns"] + int(work["phase_spans"][-1]["end_s"] * 1e9)
    c.atomic_json(raw / "sealed_work.json", {k: v for k, v in work.items() if k != "raw_reference"})
    work["raw_reference"] = c.reference(raw / "sealed_work.json")
    value = c.build(work, validation)
    c.progress("publication_before")
    try:
        receipt = publish_primary(output, value, lambda p: terminal(p, raw / "terminal"))
    except ValueError as error:
        c.progress("publication_rejected_" + str(error))
        return 1
    c.atomic_json(raw / "publication.json", receipt)
    readers = reader_receipt(
        c.TASK, output.parent, field="contract_ready_score", expected=value["contract_ready_score"]
    )
    c.atomic_json(raw / "reader_receipt.json", readers)
    c.progress("publication_after", value["completed_count"])
    print(
        json.dumps(
            dict(
                path=str(output),
                verdict=value["honest_verdict"],
                contract_ready_score=value["contract_ready_score"],
            )
        ),
        flush=True,
    )
    return 0 if value["required_checks_passed"] and readers["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Private output and replay options never change the active roadmap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261105")
    parser.add_argument("--output", type=Path, default=c.ROOT / "results" / (c.NAME + ".json"))
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
