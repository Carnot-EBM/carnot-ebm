"""REQ-REPORT-8494: reuse bounded validation and unchanged atomic publication."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from unittest.mock import patch

from carnot.reporting import append_service_cost_8494 as r
from carnot.reporting import append_store_runner_8493 as shipped


def main(argv: list[str] | None = None) -> int:
    """Private controls exercise genuine CLI exits without recursive suite execution."""
    r.progress("start_no_model_load_MODEL_SPECS_empty_current_llm_calls_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261013"], default="20261013")
    parser.add_argument("--root", type=Path, default=r.ROOT)
    parser.add_argument("--output", type=Path, default=r.ROOT / "results" / (r.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--arm", choices=["snapshot", "append"])
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.worker:
        if not args.store or not args.arm:
            parser.error("worker store and arm required")
        return r.worker(args.worker, args.store, args.arm)
    if args.cold_replay:
        passed = r.replay(args.cold_replay)
        r.progress("replay_passed" if passed else "replay_rejected")
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(r.ROOT):
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8494-owned-", dir="/var/tmp") as folder:
            with patch.object(shipped, "r", r):
                return int(shipped.run(args.root, output, Path(folder), args.private_control))
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
