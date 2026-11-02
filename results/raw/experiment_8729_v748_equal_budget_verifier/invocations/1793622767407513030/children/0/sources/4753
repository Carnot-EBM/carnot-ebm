"""REQ-VERIFY-8414: qualify current continuity through shipped bounded children."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import gatemate_continuity_runner_8400 as old
from carnot.reporting import gatemate_obligation_8414 as e
from carnot.reporting.v709_execution import execute

Json = dict[str, Any]
ORIGINAL_MANIFEST, ORIGINAL_RUN, ORIGINAL_MAIN = old.manifest, old.run, old.main


def manifest(private: Path) -> list[Json]:
    """Keep original source and private E2E controls while covering only current code."""
    with patch.object(old, "e", e):
        plan: list[Json] = ORIGINAL_MANIFEST(private)
    return plan


def run(
    root: Path,
    output: Path,
    private: Path,
    supplied: list[Path] | None = None,
    *,
    control: bool = False,
) -> int:
    """Reuse disk, memory, deadline, replay and unchanged atomic publication checks."""
    with patch.multiple(old, e=e, manifest=manifest, execute=execute):
        return int(ORIGINAL_RUN(root, output, private, supplied, control=control))


def main(argv: list[str] | None = None) -> int:
    """Accept the current date and enforce private control outputs before work begins."""
    with patch.multiple(old, e=e, run=run):
        return int(ORIGINAL_MAIN(argv))
