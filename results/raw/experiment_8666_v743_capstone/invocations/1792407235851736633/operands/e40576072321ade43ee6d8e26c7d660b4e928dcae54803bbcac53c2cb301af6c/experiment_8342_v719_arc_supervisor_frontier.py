#!/usr/bin/env python3
"""REQ-REPORT-8342: inspect new supervisor receipts after qualified execution."""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting import arc_supervisor_frontier_8342 as e  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Private controls stay in scratch so they cannot replace live evidence."""
    print("[exp8342] start no_model_load current_game_calls=0 current_model_calls=0", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", choices=["20261009"], default="20261009")
    parser.add_argument("--output", type=Path, default=e.OUTPUT)
    parser.add_argument("--private-e2e", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.private_e2e and not args.output.resolve().is_relative_to(Path("/tmp")):
        parser.error("private E2E output requires /tmp scratch")
    if args.cold_replay:
        passed = e.replay(e.json_document(args.cold_replay))
        print(f"[exp8342] replay_passed={passed}", flush=True)
        return int(not passed)
    with TemporaryDirectory(prefix="carnot-8342-", dir="/tmp") as scratch:
        return e.run(args.output, Path(scratch), fixture=args.private_e2e)


if __name__ == "__main__":
    raise SystemExit(main())
