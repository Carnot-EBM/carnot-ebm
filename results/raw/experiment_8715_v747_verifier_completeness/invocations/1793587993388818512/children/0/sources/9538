#!/usr/bin/env python3
"""Small parameterized CLI for REQ-REPORT-7928 publication qualification."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]
from carnot.reporting.publication_qualification_7928 import fixture, qualify, replay  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Expose fixture and cold routes while keeping all work in callable libraries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20260930")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fixture-e2e", type=Path)
    modes.add_argument("--cold-replay", type=Path)
    modes.add_argument("--terminal-recheck", type=Path)
    args = parser.parse_args(argv)
    print("[exp7928] start publication CLI", flush=True)
    if args.date != "20260930":
        raise ValueError("run_date_mismatch")
    if args.fixture_e2e:
        result = fixture(args.fixture_e2e)
    elif args.cold_replay or args.terminal_recheck:
        result = replay(args.cold_replay or args.terminal_recheck)
    else:
        return qualify(args.date)
    print(json.dumps(result), flush=True)
    return int(not result["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
