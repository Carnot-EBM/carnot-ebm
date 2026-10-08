"""REQ-REPORT-8315: reuse bounded children and unchanged atomic publication checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import kv260_local_cost_boundary_8315 as e
from carnot.reporting import local_update_execution_8306 as qualified

Json = dict[str, Any]
BASE_MANIFEST = qualified.manifest


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze private cost tests, coverage, consumers and unchanged terminal auditors."""
    with patch.object(qualified, "e", e):
        specs = BASE_MANIFEST(private, candidate)
    specs["commands"][1]["argv"] = [
        str(e.ROOT / ".venv/bin/pytest"),
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "--basetemp=" + str(private / "consumer-tests"),
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_primary_publication_7928.py",
    ]
    specs["commands"][1]["deadline_s"] = 180
    return specs


def main(argv: list[str] | None = None) -> int:
    """Use tested process cleanup and failure recovery without loading a model."""
    e.progress("start_no_model_load")
    with patch.object(qualified, "e", e), patch.object(qualified, "manifest", manifest):
        return int(qualified.main(argv))
