#!/usr/bin/env python3
"""REQ-VERIFY-8469: evidence-only three-board continuity, with no model load."""

import argparse
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from carnot.reporting import board_operation_boundary_8469 as e  # noqa: E402
from carnot.reporting import board_operation_runner_8469 as r  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Private paths and an owned timer preserve old publications and the task cap."""
    e.progress("start_no_model_load_MODEL_SPECS_empty_current_LLM_invocations_zero")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261013"], default="20261013")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-e2e", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "reduction_drift", int(passed), 0)
        return int(not passed)
    if args.private_e2e and args.output.resolve().is_relative_to(ROOT):
        parser.error("private E2E output requires scratch outside the checkout")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8469-private-", dir="/var/tmp") as directory:
            return r.run(
                args.root, args.output.absolute(), Path(directory), control=args.private_e2e
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)


if __name__ == "__main__":
    raise SystemExit(main())
