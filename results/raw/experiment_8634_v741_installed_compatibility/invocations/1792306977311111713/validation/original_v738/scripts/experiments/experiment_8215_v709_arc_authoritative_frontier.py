#!/usr/bin/env python3
"""Read producer-bound ARC receipts without games or models. REQ-REPORT-8215.

An explicit versioned locator seals authority. The default is discovered from
real producer definitions and their ledger; no live-state substitute is written.
"""

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting import arc_authoritative_execution_8215 as task  # noqa: E402
from carnot.reporting import arc_authoritative_frontier_8215 as reader  # noqa: E402
from carnot.reporting.current_work_receipt import atomic_json  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Keep private qualification separate from the published live denominator."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261006")
    parser.add_argument("--locator", type=Path)
    parser.add_argument("--output", type=Path, default=task.OUTPUT)
    parser.add_argument("--fixture-e2e", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20261006":
        parser.error("date must be 20261006")
    if args.fixture_e2e and args.output.resolve().is_relative_to(
        (reader.ROOT / "results").resolve()
    ):
        parser.error("fixture output requires private scratch")
    if args.cold_replay:
        errors = task.replay(json.loads(args.cold_replay.read_text()))
        print(json.dumps(dict(replay_passed=not errors, errors=errors)), flush=True)
        return int(bool(errors))
    with tempfile.TemporaryDirectory(prefix="carnot-8215-", dir="/tmp") as scratch:
        locator = args.locator
        if locator is None:
            locator = args.output.parent / "raw" / args.output.stem / "authority_locator.v1.json"
            atomic_json(locator, reader.discover())
        return task.execute(locator, args.output, Path(scratch), fixture=args.fixture_e2e)


if __name__ == "__main__":
    raise SystemExit(main())
