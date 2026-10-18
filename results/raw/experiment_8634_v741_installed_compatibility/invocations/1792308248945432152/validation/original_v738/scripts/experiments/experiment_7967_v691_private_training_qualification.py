"""Run private fitting qualification without natural training (REQ-REPORT-7967)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.verify import private_training_7967 as core  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Expose explicit private checks and cold replay with bounded exception exits."""
    core.progress("start_flushed")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, choices=["20261001"])
    parser.add_argument(
        "--output",
        type=Path,
        default=core.ROOT / "results/experiment_7967_v691_private_training_qualification.json",
    )
    parser.add_argument("--publication", type=Path, default=core.fit.PUBLICATION)
    parser.add_argument("--runtime", type=Path, default=core.fit.RUNTIME)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--check-private-root", type=Path)
    parser.add_argument("--guard-fixture", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            core.replay(args.cold_replay)
            return 0
        if args.check_private_root:
            core.check_roots(args.check_private_root, args.output.parent / "raw" / args.output.stem)
            core.progress("private_root_accepted")
            return 0
        if args.guard_fixture:
            receipt = core.guard_probe(args.guard_fixture)
            print(json.dumps(receipt), flush=True)
            return int(not receipt["passed"]) * 2
        return core.qualify(args.output, args.publication, args.runtime)
    except (ValueError, OSError, KeyError) as exc:
        core.progress("failure: " + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
