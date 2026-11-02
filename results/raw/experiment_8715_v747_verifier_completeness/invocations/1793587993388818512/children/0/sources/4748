"""REQ-VERIFY-8441: qualify continuity through shipped bounded child processes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import gatemate_continuity_runner_8400 as old
from carnot.reporting import gatemate_obligation_8441 as e
from carnot.reporting.v709_execution import execute

Json = dict[str, Any]
ORIGINAL_MANIFEST = old.manifest


def manifest(private: Path) -> list[Json]:
    """Keep original private source and E2E controls; cover only current statements."""
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
    """Reuse disk, memory, deadlines, cold controls and unchanged atomic publication."""
    with patch.multiple(old, e=e, manifest=manifest, execute=execute):
        return int(old.run(root, output, private, supplied, control=control))


def main(argv: list[str] | None = None) -> int:
    """Allow current date and private controls while keeping historical bytes intact."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261012"], default="20261012")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--supplied-receipt", type=Path, action="append")
    parser.add_argument("--private-e2e", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_e2e and output.is_relative_to(e.ROOT):
        parser.error("private controls require output outside the repository")
    with e.budget(4800), TemporaryDirectory(prefix="exp8441-owned-", dir="/var/tmp") as directory:
        return run(
            args.root, output, Path(directory), args.supplied_receipt, control=args.private_e2e
        )
