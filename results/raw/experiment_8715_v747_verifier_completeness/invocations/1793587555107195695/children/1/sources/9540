"""Fit fresh heads through the qualified callable runtime (REQ-VERIFY-7930-V688)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.verify import energy_fit_7930 as core  # noqa: E402
from carnot.verify import energy_fit_7930_run as run  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Explicit private paths exercise the same producer without touching history."""
    core.progress("start", "flushed")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, choices=["20260930"])
    parser.add_argument("--runtime", type=Path, default=core.RUNTIME)
    parser.add_argument("--upstream", type=Path, default=core.UPSTREAM)
    parser.add_argument(
        "--output", type=Path, default=core.ROOT / "results/experiment_7930_v688_energy_fit.json"
    )
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--assert-ready", action="store_true")
    args = parser.parse_args(argv)
    if args.cold_replay:
        try:
            core.replay(args.cold_replay)
        except (ValueError, OSError, KeyError):
            core.progress("replay", "mismatch")
            return 2
        return 0
    result = run.produce(args.upstream, args.runtime, args.output)
    if args.assert_ready and json.loads(args.output.read_text())["energy_fit_ready_score"] != 1:
        return 2
    return result


if __name__ == "__main__":
    raise SystemExit(main())
