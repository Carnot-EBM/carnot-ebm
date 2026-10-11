"""REQ-VERIFY-8400: reuse bounded validation and atomic publication for documentation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import gatemate_continuity_8400 as e
from carnot.reporting import gatemate_obligation_runner_8386 as old
from carnot.reporting import v717_contract_runner as qualified
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Freeze only owned commands; retain inherited exact-source and private authority controls."""
    with patch.object(qualified, "m", e):
        plan: list[Json] = qualified.manifest(private)
    plan[0]["deadline"] = 600
    for test in [
        "tests/python/test_gatemate_missing_evidence_8372.py::test_private_receipt_consumers",
        "tests/python/test_gatemate_missing_evidence_8372.py::test_authenticated_failure_rejections",
        "tests/python/test_gatemate_obligation_delta_8386.py::test_private_source_reader_and_reopening_tamper",
    ]:
        plan[1]["argv"].insert(-1, test)
    return plan


def emit(path: Path, value: Any) -> None:
    """The inherited manifest must describe the actual shorter evidence scan cap."""
    if path.name == "execution_manifest.json":
        value = dict(value, evidence_scan_cap_s=60)
    atomic_json(path, value)


def run(
    root: Path,
    output: Path,
    private: Path,
    supplied: list[Path] | None = None,
    *,
    control: bool = False,
) -> int:
    """Reuse real resource checks and child deadlines while preserving all failed receipts."""
    with patch.multiple(old, e=e, manifest=manifest, atomic_json=emit, execute=execute):
        return int(old.run(root, output, private, supplied, control=control))


def main(argv: list[str] | None = None) -> int:
    """Require private control outputs so tests cannot replace the research record."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    e.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261011"], default="20261011")
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
    with e.budget(4800), TemporaryDirectory(prefix="exp8400-owned-", dir="/var/tmp") as directory:
        return run(
            args.root, output, Path(directory), args.supplied_receipt, control=args.private_e2e
        )
