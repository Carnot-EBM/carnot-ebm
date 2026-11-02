#!/usr/bin/env python3
"""REQ-REPORT-8272: inspect sealed environment outcomes without games or models.

Private fixture inputs stay outside live results so test evidence cannot be
mistaken for hidden-game discoveries. The default locator comes from producers.
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

from carnot.reporting import arc_outcome_frontier_8272 as reader  # noqa: E402
from carnot.reporting import arc_outcome_execution_8272 as task  # noqa: E402
from carnot.reporting.current_work_receipt import atomic_json  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Keep development controls and the live primary in separate output locations."""
    print("[exp8272] start no_model_load current_llm_calls=0", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261008")
    parser.add_argument("--locator", type=Path)
    parser.add_argument("--frontier", type=Path, default=reader.FRONTIER)
    parser.add_argument("--output", type=Path, default=task.OUTPUT)
    parser.add_argument("--fixture-e2e", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.date != "20261008":
        parser.error("date must be 20261008")
    if args.fixture_e2e and args.output.resolve().is_relative_to(
        (reader.ROOT / "results").resolve()
    ):
        parser.error("fixture output requires private scratch")
    if args.frontier != reader.FRONTIER and not args.fixture_e2e:
        parser.error("alternate frontier requires private fixture mode")
    if args.cold_replay:
        errors = task.replay(json.loads(args.cold_replay.read_text()))
        print(json.dumps(dict(replay_passed=not errors, errors=errors)), flush=True)
        return int(bool(errors))
    with tempfile.TemporaryDirectory(prefix="carnot-8272-", dir="/tmp") as scratch:
        locator = args.locator
        if locator is None:
            locator = args.output.parent / "raw" / args.output.stem / "authority_locator.v1.json"
            atomic_json(locator, reader.authority.discover())
        return task.execute(
            locator, args.frontier, args.output, Path(scratch), fixture=args.fixture_e2e
        )


if __name__ == "__main__":
    raise SystemExit(main())
