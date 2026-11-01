"""REQ-REPORT-8212: freeze owned validation through the qualified supervisor."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as previous
from carnot.verify import memory_benefit_audit_8212 as e

Json = dict[str, Any]
OWNED = [e.MODULE, e.RUNNER, e.CLI]
BASE_MANIFEST = previous.manifest


def manifest(private: Path, candidate: Path) -> Json:
    """Coverage includes the real script path, while repository health stays separate."""
    with patch.object(previous, "e", e), patch.object(previous, "OWNED", OWNED):
        specs = BASE_MANIFEST(private, candidate)
    config = private / "coverage.ini"
    config.write_text(config.read_text() + "[report]\nexclude_lines =\n")
    specs["commands"][0]["deadline_s"] = 600
    for spec in specs["commands"]:
        if spec["name"].startswith("coverage_"):
            spec["argv"].append("--data-file=" + str(private / ".coverage"))
        if spec["name"] == "spec_coverage":
            spec["argv"].insert(-1, "--files")
    for spec in specs["terminal_commands"]:
        spec["deadline_s"] = 180
    return specs


def manifest_adapter(private: Path, candidate: Path) -> Json:
    """The saved builder avoids recursion when the legacy CLI calls its manifest."""
    return manifest(private, candidate)


def main(argv: list[str] | None = None) -> int:
    """Qualified private publication and cold replay share the same audit reducer."""
    with (
        patch.object(previous, "e", e),
        patch.object(previous, "OWNED", OWNED),
        patch.object(previous, "manifest", manifest_adapter),
        patch.object(previous, "run_check", e.run_check),
    ):
        return previous.main(argv)
