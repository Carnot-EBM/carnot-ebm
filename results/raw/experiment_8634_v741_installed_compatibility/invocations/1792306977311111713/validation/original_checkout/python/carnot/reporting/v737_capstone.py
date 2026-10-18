"""REQ-VERIFY-8590: small adapters preserve bounded validation and publication."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import v736_capstone as base
from carnot.reporting import v737_capstone_evidence as e
from carnot.reporting import v737_capstone_report as report

SUITES = base.SUITES


def manifest(private: Path) -> list[dict[str, Any]]:
    """Keep original consumer test IDs and instrument only this invocation's new statements."""
    with patch.object(base, "e", e):
        return list(base.manifest(private))


def main(argv: list[str] | None = None) -> int:
    """Bind the requested date while retaining the shipped deadline and private controls."""
    e.progress("start_no_model_load_current_LLM_calls_0", 0, 12)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--date", choices=["20261017"], default="20261017")
    _, rest = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    with patch.multiple(base, e=e, report=report):
        return int(base.main(["--date", "20261016", *rest]))
