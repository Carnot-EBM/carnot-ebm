#!/usr/bin/env python3
"""REQ-REPORT-8358: run immutable replay qualification from any directory."""

import argparse
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ.setdefault("TMPDIR", "/tmp/carnot8358-private-20261009")

from carnot.reporting import v720_terminal_replay as e  # noqa: E402
from carnot.reporting.v720_replay_execution import run  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Dispatch bounded workers separately so they cannot launch another experiment."""
    e.progress("start_no_model_load_current_model_calls_0")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261009"], default="20261009")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-e2e", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker_request:
        if args.worker_output is None:
            parser.error("worker output is required")
        return e.worker(args.worker_request, args.worker_output)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        print(f"[exp8358] replay_passed={passed}", flush=True)
        return int(not passed)
    if args.private_e2e and not str(args.output.absolute()).startswith("/tmp/"):
        parser.error("private E2E requires /tmp output")
    with TemporaryDirectory(prefix="exp8358-") as directory:
        private = Path(directory)
        private.chmod(0o700)
        return run(args.output, private, fixture=args.private_e2e)


if __name__ == "__main__":
    raise SystemExit(main())
