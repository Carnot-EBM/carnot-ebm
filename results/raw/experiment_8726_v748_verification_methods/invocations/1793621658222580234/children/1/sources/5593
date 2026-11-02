"""REQ-VERIFY-8626: bounded original validators qualify exact candidate bytes."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import lean_runner_8608 as shipped
from carnot.reporting import v739_capstone as previous
from carnot.reporting import v740_capstone_evidence as e
from carnot.reporting import v740_capstone_report as report
from carnot.reporting import v740_capstone_samples as samples
from carnot.reporting.gatemate_continuity_8400 import budget

Json = dict[str, Any]


def validate(raw: Path) -> list[Json]:
    """The shipped validator collects exact nodes before disjoint bounded children."""
    with patch.object(shipped, "e", e):
        return list(shipped.validate(raw))


def terminal(path: Path, raw: Path) -> Json:
    """Retain unchanged cold replay, adversarial and strict-row validator findings."""
    with patch.object(shipped, "e", e):
        return dict(shipped.terminal(path, raw))


def main(argv: list[str] | None = None) -> int:
    """Accept a real calendar date and isolate private failure controls from checkout."""
    e.progress("start_no_model_load_current_LLM_calls_0", 0, 12)
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
            e.progress("replay_passed", 12, 0)
            return 0
        if len(args.date) != 8:
            raise ValueError("calendar_date")
        datetime.strptime(args.date, "%Y%m%d")
        if (args.private_control or args.failed_child) and args.output.absolute().is_relative_to(
            e.ROOT
        ):
            raise ValueError("private controls require an output outside checkout")
        with (
            budget(4800),
            patch.multiple(previous, e=e, report=report, validate=validate, terminal=terminal),
        ):
            return int(
                previous.run(
                    args.output.absolute(), args.date, args.private_control, args.failed_child
                )
            )
    except (OSError, ValueError, KeyError, TypeError, TimeoutError) as error:
        print(f"[exp8626] error={error}", file=sys.stderr, flush=True)
        return 1
