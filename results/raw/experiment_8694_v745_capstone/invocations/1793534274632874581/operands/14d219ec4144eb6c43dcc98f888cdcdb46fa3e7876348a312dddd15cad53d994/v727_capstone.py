"""REQ-VERIFY-8456: real bounded validation precedes checked atomic publication."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_capstone as old
from carnot.reporting import v727_capstone_evidence as e
from carnot.reporting.v727_capstone_report import build

Json = dict[str, Any]
_RUN, _MAIN, _MANIFEST = old.run, old.main, old.manifest


def manifest(private: Path) -> list[Json]:
    """Inherited serial tests collect real subprocess coverage with unchanged E2E checks."""
    with patch.object(old, "e", e):
        return list(_MANIFEST(private))


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, failed: bool = False
) -> int:
    """Reuse the qualified pipeline so failure publication sees the same validators."""
    with patch.multiple(old, e=e, build=build, manifest=manifest):
        return int(_RUN(root, output, private, control=control, failed=failed))


def main(argv: list[str] | None = None) -> int:
    """The dated CLI preserves standalone imports and owns only disk-backed scratch."""
    with patch.multiple(old, e=e, build=build, run=run, manifest=manifest):
        return int(_MAIN(argv))
