#!/usr/bin/env python3
"""Audit live receipts without replaying games. REQ-REPORT-7924-V687."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

from carnot.reporting import arc_supervisor_v687_delta as task
from carnot.reporting.current_work_receipt import atomic_json


def main(argv: list[str] | None = None) -> int:
    """Keep private fixture and replay routes separate from the current receipt run."""
    started = time.monotonic()
    task.progress(started, "cli_start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260930", choices=["20260930"])
    parser.add_argument("--reduce-ledger", type=Path)
    parser.add_argument("--producer", type=Path, action="append")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--terminal-recheck", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args(argv)
    path = args.cold_replay or args.terminal_recheck
    if path:
        errors = task.replay(json.loads(path.read_text()))
        print(json.dumps({"cold_replay_errors": errors}), flush=True)
        return int(bool(errors))
    if args.reduce_ledger:
        if not args.producer or args.output is None:
            parser.error("--reduce-ledger requires --producer and --output")
        task.progress(started, "before_private_reduce")
        delta = task.previous.reduce_receipts(args.reduce_ledger, args.producer, {}, {})
        atomic_json(args.output, delta)
        task.progress(started, "after_private_reduce", delta["new_outcome_count"])
        return 0
    private = Path(tempfile.mkdtemp(prefix="carnot-exp7924-", dir="/tmp"))
    options = {"resume_from": args.resume_from} if args.resume_from else {}
    return task.execute(
        args.output or task.ROOT / "results/experiment_7924_v687_arc_supervisor_delta.json",
        private,
        **options,
    )


if __name__ == "__main__":
    sys.exit(main())
