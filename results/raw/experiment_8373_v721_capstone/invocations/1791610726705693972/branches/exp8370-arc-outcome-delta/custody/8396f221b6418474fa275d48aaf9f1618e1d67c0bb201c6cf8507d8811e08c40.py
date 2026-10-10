#!/usr/bin/env python3
"""REQ-VERIFY-8370: private controls cannot replace authenticated live evidence."""

import argparse
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting import arc_outcome_delta_8370 as e  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []
SCRATCH = Path.home() / ".cache/carnot-exp8370-private"


def main(argv: list[str] | None = None) -> int:
    """Use private disk scratch and a task deadline so silent children cannot linger."""
    print("[exp8370] start no_model_load current_game_calls=0 current_model_calls=0", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261010"], default="20261010")
    parser.add_argument("--output", type=Path, default=e.OUTPUT)
    parser.add_argument("--private-e2e", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.private_e2e and args.output.resolve().is_relative_to(e.ROOT):
        parser.error("private E2E output requires scratch outside the checkout")
    if args.cold_replay:
        passed = e.replay(e.json_document(args.cold_replay))
        print(f"[exp8370] replay_passed={passed}", flush=True)
        return int(not passed)
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
        SCRATCH.chmod(0o700)
        with TemporaryDirectory(prefix="invocation-", dir=SCRATCH) as scratch:
            return e.run(args.output, Path(scratch), fixture=args.private_e2e)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)


if __name__ == "__main__":
    raise SystemExit(main())
