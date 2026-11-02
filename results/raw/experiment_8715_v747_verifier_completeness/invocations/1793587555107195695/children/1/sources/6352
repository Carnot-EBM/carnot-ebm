#!/usr/bin/env python3
"""Print local Claude/Codex plan-usage snapshots.

Spec: REQ-REPORT-024, SCENARIO-REPORT-021, SCENARIO-REPORT-022,
SCENARIO-REPORT-023, SCENARIO-REPORT-025.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PYTHON_ROOT = _REPO_ROOT / "python"
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))

from carnot.reporting.agent_usage import build_usage_snapshot, format_usage_table


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect local Claude/Codex logs and print a plan-usage snapshot."
    )
    parser.add_argument(
        "--home",
        default=str(Path.home()),
        help="Home directory containing .codex/ and .claude/ (default: current user's home).",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json"),
        default="table",
        help="Output format (default: table).",
    )
    parser.add_argument(
        "--claude-live",
        action="store_true",
        help="Fetch exact live Claude usage from the authenticated OAuth usage endpoint.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    snapshot = build_usage_snapshot(home=Path(args.home), claude_live=args.claude_live)
    if args.format == "json":
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    else:
        print(format_usage_table(snapshot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
