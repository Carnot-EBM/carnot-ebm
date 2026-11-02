#!/usr/bin/env python3
"""REQ-REPORT-8328: inspect sealed live outcomes with zero game and model calls."""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2] / "python"),
    str(Path(__file__).resolve().parents[2]),
]

from carnot.reporting import arc_supervisor_execution_8328 as e  # noqa: E402
from carnot.reporting import arc_supervisor_frontier_8328 as m  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Private CLI controls cannot publish into the live result directory."""
    print("[exp8328] start no_model_load current_game_calls=0 current_model_calls=0", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261009")
    parser.add_argument("--output", type=Path, default=e.OUTPUT)
    parser.add_argument("--private-e2e", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20261009":
        parser.error("date must be 20261009")
    if args.private_e2e and not args.output.resolve().is_relative_to(Path("/tmp")):
        parser.error("private E2E output requires /tmp scratch")
    if args.cold_replay:
        passed = e.replay(m.json_document(args.cold_replay))
        print(f"[exp8328] replay_passed={passed}", flush=True)
        return int(not passed)
    with TemporaryDirectory(prefix="carnot-8328-", dir="/tmp") as scratch:
        return e.run(args.output, Path(scratch), fixture=args.private_e2e)


if __name__ == "__main__":
    raise SystemExit(main())
