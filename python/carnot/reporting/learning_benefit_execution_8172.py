"""REQ-REPORT-8172: reuse bounded validation before checked publication."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import learning_audit_execution_8144 as previous
from carnot.verify import learning_benefit_audit_8172 as e

Json = dict[str, Any]
OWNED = [e.MODULE, e.CLI, e.RUNNER]
BASE_MANIFEST = previous.manifest


def manifest(private: Path, candidate: Path) -> Json:
    """Real file paths and identical includes freeze coverage before measurement."""
    with patch.object(previous, "e", e), patch.object(previous, "OWNED", OWNED):
        specs = BASE_MANIFEST(private, candidate)
    specs["repository_health"]["deadline_s"] = 1200
    for spec in specs["terminal_commands"]:
        spec["deadline_s"] = 300
    return specs


def main(argv: list[str] | None = None) -> int:
    """The qualified CLI supports private success, blocking and cold replay."""
    with patch.object(previous, "e", e), patch.object(previous, "manifest", manifest):
        return previous.main(argv)
