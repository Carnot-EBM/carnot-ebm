"""REQ-VERIFY-8525: reuse bounded shipped validation and exact-byte publication."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v730_capstone as shipped
from carnot.reporting import v731_capstone as previous
from carnot.reporting import v732_capstone_evidence as e
from carnot.reporting.v732_capstone_report import build

Json = dict[str, Any]
SUITES = shipped.SUITES


def manifest(private: Path) -> list[Json]:
    """Collect first, then retain the exact historical union in bounded disjoint shards."""
    with patch.object(shipped, "e", e):
        return list(shipped.manifest(private))


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, failed: bool = False
) -> int:
    """External upstream blocks and failed owned validation retain different terminal classes."""
    with (
        patch.multiple(
            shipped.old,
            e=e,
            build=build,
            manifest=shipped._MANIFEST if control else manifest,
            controls=previous.controls,
            child=shipped.complete_child,
            terminal_child=shipped.complete_child,
        ),
        patch.object(previous, "e", e),
    ):
        return int(shipped._RUN(root, output, private, control=control, failed=failed))


def main(argv: list[str] | None = None) -> int:
    """The original CLI owns dated private scratch and the 4800-second deadline."""
    with patch.multiple(shipped, e=e, run=run):
        return int(shipped.main(argv))
