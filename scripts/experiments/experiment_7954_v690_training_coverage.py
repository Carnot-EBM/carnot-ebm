"""Qualify real fitting exception routes with no natural fit (REQ-REPORT-7954)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.verify import training_coverage_7954 as core  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """A small explicit wrapper exposes private paths for current callable qualification."""
    core.progress("start_flushed")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, choices=["20260930"])
    parser.add_argument(
        "--output",
        type=Path,
        default=core.ROOT / "results/experiment_7954_v690_training_coverage.json",
    )
    parser.add_argument("--publication", type=Path, default=core.fit.PUBLICATION)
    parser.add_argument("--runtime", type=Path, default=core.fit.RUNTIME)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            core.replay(args.cold_replay)
            return 0
        return core.qualify(args.output, args.publication, args.runtime)
    except (ValueError, OSError, KeyError) as exc:
        core.progress("failure: " + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
