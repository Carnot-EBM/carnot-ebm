"""REQ-VERIFY-8614: bounded real checks precede reader-visible publication."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import lean_runner_8608 as shipped
from carnot.reporting import v739_capstone_evidence as e
from carnot.reporting import v739_capstone_report as report
from carnot.reporting import v739_capstone_samples as samples
from carnot.reporting.gatemate_continuity_8400 import budget
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def validate(raw: Path) -> list[Json]:
    """Use the shipped disjoint-node validator without running historical controllers."""
    with patch.object(shipped, "e", e):
        return list(shipped.validate(raw))


def terminal(path: Path, raw: Path) -> Json:
    """Retain all original validator exits, including warnings and rejected claims."""
    with patch.object(shipped, "e", e):
        return dict(shipped.terminal(path, raw))


def run(output: Path, run_date: str, private: bool, failed: bool) -> int:
    """Qualify private candidate bytes before atomically publishing the final primary."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    work = e.measure(raw, run_date)
    receipts = (
        [
            child(
                "private_child",
                [
                    sys.executable,
                    "-c",
                    "raise SystemExit(7)" if failed else "print('private CPU control')",
                ],
                raw / "checks",
                deadline=10,
            )
        ]
        if private
        else validate(raw)
    )
    e.atomic_json(raw / "validation.json", dict(receipts=receipts))
    value = report.build(work, receipts, output)
    private_output = raw / "private" / output.name
    publication = publish_primary(
        private_output, value, lambda p: terminal(p, raw / "private_checks")
    )
    reader = reader_receipt(
        e.TASK,
        private_output.parent,
        field="required_checks_passed",
        expected=value["required_checks_passed"],
    )
    receipts.append(
        dict(
            name="private_publication_readers",
            passed=reader["passed"],
            publication=publication,
            reader=reader,
        )
    )
    e.atomic_json(raw / "validation.json", dict(receipts=receipts))
    value = report.build(work, receipts, output)
    published = publish_primary(output, value, lambda p: terminal(p, raw / "terminal_checks"))
    e.atomic_json(
        raw / "terminal_validation.json",
        dict(
            passed=True,
            publication=published,
            consumer=reader_receipt(
                e.TASK,
                output.parent,
                field="required_checks_passed",
                expected=value["required_checks_passed"],
            ),
        ),
    )
    e.progress("complete", 13, 0)
    return 0 if value["required_checks_passed"] else 1


def main(argv: list[str] | None = None) -> int:
    """Replay preserves the invocation date; new runs accept valid calendar dates."""
    e.progress("start_no_model_load_current_LLM_calls_0", 0, 13)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now().strftime("%Y%m%d"))
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--sample-output", type=Path)
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--failed-child", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.sample_output:
            e.atomic_json(args.sample_output, samples.reduce())
            return 0
        if args.cold_replay:
            report.replay(args.cold_replay)
            e.progress("replay_passed", 13, 0)
            return 0
        if len(args.date) != 8:
            raise ValueError("calendar_date")
        datetime.strptime(args.date, "%Y%m%d")
        if (args.private_control or args.failed_child) and args.output.absolute().is_relative_to(
            e.ROOT
        ):
            raise ValueError("private controls require an output outside checkout")
        with budget(4800):
            return run(args.output.absolute(), args.date, args.private_control, args.failed_child)
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
        print(f"[exp8614] error={error}", file=sys.stderr, flush=True)
        return 1
