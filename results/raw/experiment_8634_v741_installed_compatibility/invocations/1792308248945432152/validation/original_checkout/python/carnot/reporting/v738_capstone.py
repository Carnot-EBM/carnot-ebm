"""REQ-VERIFY-8601: retain bounded original consumer validation and atomic publication."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone, UTC
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import v736_capstone as base
from carnot.reporting import v738_capstone_evidence as e
from carnot.reporting import v738_capstone_report as report

SUITES = base.SUITES


def manifest(private: Path) -> list[dict[str, Any]]:
    """Preserve original collected node IDs and instrument only newly owned statements."""
    with patch.object(base, "e", e):
        return list(base.manifest(private))


def main(argv: list[str] | None = None) -> int:
    """Validate the invocation calendar date independently from the legacy parser adapter."""
    e.progress("start_no_model_load_current_LLM_calls_0", 0, 11)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--date", default=datetime.now(UTC).strftime("%Y%m%d"))
    args, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    try:
        if len(args.date) != 8:
            raise ValueError("eight digit date")
        datetime.strptime(args.date, "%Y%m%d")
    except ValueError:
        parser.error("--date requires a valid YYYYMMDD calendar date")
    with patch.object(e, "RUN_DATE", args.date), patch.multiple(base, e=e, report=report):
        return int(base.main(["--date", "20261016", *rest]))
