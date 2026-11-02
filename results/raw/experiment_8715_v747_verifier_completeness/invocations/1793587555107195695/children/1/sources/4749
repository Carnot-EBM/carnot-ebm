"""REQ-VERIFY-8455: reuse bounded resource checks and unchanged atomic consumers."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import gatemate_obligation_8455 as e
from carnot.reporting import gatemate_obligation_runner_8441 as old
from carnot.reporting import v726_feature_panel_runner as qualified
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Keep genuine child coverage and both private authority and frozen replay tests."""
    with patch.object(qualified, "e", e):
        plan: list[Json] = qualified.manifest(private)
    for test in [
        "tests/python/test_gatemate_missing_evidence_8372.py::test_private_receipt_consumers",
        "tests/python/test_gatemate_missing_evidence_8372.py::test_authenticated_failure_rejections",
        "tests/python/test_gatemate_obligation_delta_8386.py::test_private_source_reader_and_reopening_tamper",
    ]:
        plan[1]["argv"].insert(-1, test)
    return plan


def run(
    root: Path,
    output: Path,
    private: Path,
    supplied: list[Path] | None = None,
    *,
    control: bool = False,
) -> int:
    """Reuse disk, memory, deadlines and cold controls; preserve real coverage shards."""
    with (
        patch.multiple(old, e=e, manifest=manifest, execute=execute),
        patch.object(old.old, "emit", emit),
    ):
        status = int(old.run(root, output, private, supplied, control=control))
    shards = output.parent / "raw" / output.stem / "coverage_shards"
    shards.mkdir(parents=True, exist_ok=True)
    for source in private.glob(".coverage*"):
        shutil.copyfile(source, shards / source.name)
    atomic_json(
        shards / "inventory.json", [e.base.reference(p) for p in sorted(shards.glob(".coverage*"))]
    )
    return status


def emit(path: Path, value: Any) -> None:
    """Freeze the actual measurement ceiling and validation reserve before any work."""
    if path.name == "execution_manifest.json":
        value = dict(
            value, evidence_scan_cap_s=60, measurement_cap_s=3600, validation_reserve_s=900
        )
    atomic_json(path, value)


def main(argv: list[str] | None = None) -> int:
    """Private controls exercise the CLI without replacing any historical publication."""
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
    with e.budget(4800), TemporaryDirectory(prefix="exp8455-owned-", dir="/var/tmp") as directory:
        return run(
            args.root, output, Path(directory), args.supplied_receipt, control=args.private_e2e
        )
