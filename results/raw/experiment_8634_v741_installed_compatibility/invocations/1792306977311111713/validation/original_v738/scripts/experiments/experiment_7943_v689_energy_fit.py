"""Fit current heads after isolated publication qualification (REQ-VERIFY-7943)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from carnot.verify import energy_fit_7943 as core  # noqa: E402

MODEL_SPECS: list[dict[str, str]] = []


def main(argv: list[str] | None = None) -> int:
    """Private paths exercise success and failures through the same producer and publisher."""
    core.progress("start", "flushed")
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, choices=["20260930"])
    parser.add_argument("--publication", type=Path, default=core.PUBLICATION)
    parser.add_argument("--runtime", type=Path, default=core.RUNTIME)
    parser.add_argument("--upstream", type=Path, default=core.UPSTREAM)
    parser.add_argument(
        "--output", type=Path, default=core.ROOT / "results/experiment_7943_v689_energy_fit.json"
    )
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--terminal-recheck", type=Path)
    parser.add_argument("--assert-ready", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.cold_replay or args.terminal_recheck:
            core.replay(args.cold_replay or args.terminal_recheck, bool(args.terminal_recheck))
            return 0
        if args.fixture:
            result = core.fixture(args.publication, args.runtime, args.output)
        else:
            result = core.driver(args.publication).produce(args.upstream, args.runtime, args.output)
        value = json.loads(args.output.read_text())
        if (
            args.assert_ready
            and value.get("fixture_ready_score" if args.fixture else "energy_fit_ready_score", 0)
            != 1
        ):
            return 2
        return int(result)
    except (ValueError, OSError, KeyError) as exc:
        core.progress("failure", str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
