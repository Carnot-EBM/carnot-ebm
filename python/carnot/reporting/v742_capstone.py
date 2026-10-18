"""REQ-VERIFY-8652: bounded original checks precede exact-byte publication."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shutil
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import lean_runner_8608 as shipped
from carnot.reporting import v739_capstone as previous
from carnot.reporting import v742_capstone_evidence as e
from carnot.reporting import v742_capstone_report as report
from carnot.reporting import v742_capstone_samples as samples
from carnot.reporting.gatemate_continuity_8400 import budget
from carnot.reporting.v709_execution import child

Json = dict[str, Any]


def validate(raw: Path) -> list[Json]:
    """Preserve original child SQLite files before combining copies of those files."""
    shards: list[Json] = []

    def bounded(name: str, argv: list[str], logs: Path, *, deadline: float) -> Json:
        """Keep every original node and retain the real exit of each supervised child."""
        if name == "coverage_combine":
            config = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--rcfile=")))
            copies = config.parent / "combine_copies"
            copies.mkdir()
            for source in sorted(config.parent.glob(".coverage.*")):
                target = raw / "original_child_coverage" / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                shutil.copyfile(target, copies / target.name)
                shards.append(e.reference(target))
            argv = [*argv, str(copies)]
        return dict(
            child(name, argv, logs, deadline=600 if name.startswith("tests_") else deadline)
        )

    with patch.object(shipped, "e", e), patch.object(shipped, "child", bounded):
        receipts = list(shipped.validate(raw))
    receipts.append(dict(name="preserved_child_coverage", passed=bool(shards), shards=shards))
    return receipts


def terminal(path: Path, raw: Path) -> Json:
    """Retain actual cold replay, adversarial findings and strict row scanner exits."""
    with patch.object(shipped, "e", e):
        return dict(shipped.terminal(path, raw))


def main(argv: list[str] | None = None) -> int:
    """Private controls exercise failures without exposing a research primary."""
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
        print(f"[exp8652] error={error}", file=sys.stderr, flush=True)
        return 1
