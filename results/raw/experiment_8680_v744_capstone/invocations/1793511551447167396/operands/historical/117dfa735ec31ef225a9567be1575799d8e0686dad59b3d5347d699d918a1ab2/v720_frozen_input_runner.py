"""REQ-VERIFY-8346: reuse qualified child ownership and typed atomic publication."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v718_replay_runner as qualified
from carnot.reporting import v717_contract_runner as base
from carnot.reporting import v720_frozen_input_contract as e
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
_manifest = qualified.manifest


def manifest(private: Path) -> list[Json]:
    """Owned coverage and immutable low-level consumers replace fitting and sealing."""
    with patch.object(qualified, "e", e):
        plan = list(_manifest(private))
    plan[0]["deadline"] = 600
    plan[-1] = dict(
        name="frozen_input_consumers",
        argv=[
            str(e.ROOT / ".venv/bin/pytest"),
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "-q",
            "tests/python/test_v710_contract_replay_8218.py::test_contract_and_mutations",
            "tests/python/test_v710_contract_replay_8218.py::test_snapshots_and_missing_bytes",
            "tests/python/test_v718_contract_replay_8318.py::test_finding_policy",
            "--basetemp=" + str(private / "frozen-consumers"),
        ],
        expected=0,
        deadline=180,
        scope="owned",
    )
    return plan


def inspect(args: list[str]) -> int:
    """The real CLI can inspect a preserved milestone without requesting activation."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument(
        "--inspect-milestone",
        choices=["2026.10.716", "2026.10.717", "2026.10.718", "2026.10.719", e.MILESTONE],
    )
    parser.add_argument("--check-authority", action="store_true")
    options = parser.parse_args(args)
    if options.check_authority:
        with TemporaryDirectory(prefix="exp8346-authority-") as directory:
            try:
                result = e.authority(options.root, Path(directory), options.inspect_milestone)
            except (OSError, ValueError, KeyError, TypeError):
                e.progress("authority_rejected")
                return 1
        e.progress("authority_passed" if result["activated"] else "authority_rejected")
        return int(not result["activated"])
    path = e.design(options.root, options.inspect_milestone)
    tasks = parse_design(path.read_text(), milestone=options.inspect_milestone)[1]
    print(
        json.dumps(
            dict(
                path=str(path),
                sha256=sha256_file(path),
                milestone=options.inspect_milestone,
                tasks_sha256=canonical_hash(tasks),
                count=len(tasks),
            )
        ),
        flush=True,
    )
    return int(len(tasks) != 14)


def main(argv: list[str] | None = None) -> int:
    """Small invocation adapters preserve existing bounded children and validators."""
    args = list(sys.argv[1:] if argv is None else argv)
    e.progress("start")
    if "--inspect-milestone" in args:
        return inspect(args)
    if "--date" in args:
        index = args.index("--date") + 1
        if index == len(args) or args[index] != "20261009":
            raise SystemExit("date must be 20261009")
        args[index] = "20261008"
    with (
        patch.object(qualified, "e", e),
        patch.object(qualified, "manifest", manifest),
        patch.object(base, "m", e),
    ):
        return int(qualified.main(args))
