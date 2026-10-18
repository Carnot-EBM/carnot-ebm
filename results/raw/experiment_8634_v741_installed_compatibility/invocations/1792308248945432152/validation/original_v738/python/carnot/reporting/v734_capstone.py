"""REQ-VERIFY-8551: thin adapters retain shipped bounded checks and atomic publication."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v733_capstone as shipped
from carnot.reporting import v734_capstone_evidence as e
from carnot.reporting.v734_capstone_report import build

Json = dict[str, Any]
SUITES = shipped.SUITES


def manifest(private: Path) -> list[Json]:
    """Reuse exact original consumer unions and real subprocess coverage."""
    with patch.object(shipped, "e", e):
        plan = list(shipped.manifest(private))
    for row in plan:
        if row["name"] == "owned_tests" or "shard_node_ids" in row:
            row["argv"].append("-vv")
    return plan


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, failed: bool = False
) -> int:
    """Only identity and evidence adapters change; the shipped validation gates remain intact."""
    with patch.multiple(
        shipped.shipped.old,
        e=e,
        build=build,
        manifest=shipped.shipped._MANIFEST if control else manifest,
    ):
        return int(shipped.shipped._RUN(root, output, private, control=control, failed=failed))


def main(argv: list[str] | None = None) -> int:
    """Reuse the dated standalone CLI and its private disk scratch and task deadline."""
    with patch.multiple(shipped, e=e, run=run):
        return int(shipped.main(argv))
