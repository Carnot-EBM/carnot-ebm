#!/usr/bin/env python3
"""Recover authenticated same-day receipts. REQ-REPORT-7936-TERMINAL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

from carnot.reporting import arc_supervisor_v688_receipts as reader
from carnot.reporting import arc_supervisor_v688_refinement as task
from carnot.reporting.current_work_receipt import atomic_json


def main(argv: list[str] | None = None) -> int:
    """Private CLI routes exercise receipt mechanics without publishing fixtures."""
    print("[exp7936] phase=start completed_units=0", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260930", choices=["20260930"])
    parser.add_argument("--reduce-ledger", type=Path)
    parser.add_argument("--producer", type=Path, action="append")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--terminal-recheck", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    replay = args.cold_replay or args.terminal_recheck
    if replay:
        errors = reader.replay(json.loads(replay.read_text()))
        print(json.dumps({"cold_replay_errors": errors}), flush=True)
        return int(bool(errors))
    if args.reduce_ledger:
        if not args.producer or args.output is None:
            parser.error("--reduce-ledger requires --producer and --output")
        result = reader.inspect(args.reduce_ledger, args.producer, {}, args.date, args.date, {})
        atomic_json(args.output, result)
        print(
            f"[exp7936] phase=private_reduced completed_units={result['identity_filter_count']}",
            flush=True,
        )
        return 0
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7936-", dir="/tmp"))
    return task.execute(args.output or task.OUTPUT, private)


if __name__ == "__main__":
    sys.exit(main())
