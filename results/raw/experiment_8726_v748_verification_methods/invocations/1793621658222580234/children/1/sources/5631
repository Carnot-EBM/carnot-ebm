"""REQ-VERIFY-8667: reuse bounded children and preserve original historical readers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v743_contract_runner as previous
from carnot.reporting import v744_contract_methods as e
from carnot.reporting import v744_contract_report as report

Json = dict[str, Any]
_MANIFEST = previous.manifest
collection = previous.collection
verify_collection = previous.verify_collection
retain_coverage = previous.retain_coverage
execute = previous.execute


def manifest(private: Path) -> list[Json]:
    """Collect original disjoint nodes once and retain actual SQLite before combining."""
    with patch.multiple(previous, e=e, report=report):
        plan = list(_MANIFEST(private))
    config = private / "consumer-coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n    "
        + str(e.ROOT / "python/carnot/*")
        + "\n    "
        + str(e.ROOT / "scripts/*")
        + "\n[report]\nexclude_lines=\n"
    )
    for spec in plan:
        if spec["name"].startswith("test_"):
            spec["argv"] = [
                str(e.ROOT / ".venv/bin/coverage"),
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *spec["argv"][1:],
            ]
        if spec["name"] in {"coverage_report", "coverage_json"}:
            spec["argv"] += ["--include=" + ",".join(str(e.ROOT / path) for path in e.OWNED)]
    return plan


def main(argv: list[str] | None = None) -> int:
    """Temporary bindings restore all original producer identities on return."""
    with patch.multiple(previous, e=e, report=report, manifest=manifest):
        return int(previous.main(argv))
