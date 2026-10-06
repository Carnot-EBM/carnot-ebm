"""REQ-REPORT-8206: freeze scoped validation and reuse atomic checked publication.

The existing producer handles candidate validation and separate repository health.
The qualified group supervisor retains complete streams and kills timed-out children.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import methods_stream_execution_8111 as previous
from carnot.verify import hard_exit_learning_qualification_8206 as e

Json = dict[str, Any]
OWNED = [e.MODULE, e.RUNNER, e.CLI]
BASE_MANIFEST = previous.manifest


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze owned includes and all validation argv before opening fixture outcomes."""
    with patch.object(previous, "e", e), patch.object(previous, "OWNED", OWNED):
        specs = BASE_MANIFEST(private, candidate)
    config = private / "coverage.ini"
    config.write_text(
        "[run]\npatch = _exit\nparallel = true\ndata_file = "
        + str(private / ".coverage")
        + "\ninclude =\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in [*OWNED, e.legacy.MODULE])
        + "[report]\nexclude_lines =\ninclude =\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in OWNED)
    )
    for spec in specs["commands"]:
        spec["deadline_s"] = 900 if spec["name"] == "owned_unit_and_private_CLI" else 180
        if spec["name"] == "owned_unit_and_private_CLI":
            spec["argv"].insert(1, "CARNOT_8206_COVERAGE_PARENT=" + str(private))
            spec["argv"].extend(
                e.legacy.TEST + "::" + name
                for name in [
                    "test_convex_scalar_reference_and_bounds",
                    "test_schedule_admission_and_overflow",
                    "test_no_signal_and_ineligible",
                ]
            )
        if spec["name"].startswith("coverage_"):
            spec["argv"].append("--data-file=" + str(private / ".coverage"))
        if spec["name"] == "spec_coverage":
            spec["argv"].insert(-1, "--files")
    for spec in specs["terminal_commands"]:
        spec["deadline_s"] = 300
    specs["repository_health"]["deadline_s"] = 1200
    return specs


def main(argv: list[str] | None = None) -> int:
    """Use the normal-exit publisher without installing any profile callback."""
    with (
        patch.object(previous, "e", e),
        patch.object(previous, "OWNED", OWNED),
        patch.object(previous, "manifest", manifest),
        patch.object(previous, "run_check", e.run_check),
    ):
        return previous.main(argv)
