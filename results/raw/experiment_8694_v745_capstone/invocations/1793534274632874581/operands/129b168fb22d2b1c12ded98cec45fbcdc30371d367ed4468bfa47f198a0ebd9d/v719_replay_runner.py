"""REQ-VERIFY-8332: retain qualified bounded children and finding adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v719_contract_replay as e
from carnot.reporting import v718_replay_runner as qualified
from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
_manifest = qualified.manifest


def manifest(private: Path) -> list[Json]:
    """Freeze only owned coverage with enough time for instrumented cold children."""
    with patch.object(qualified, "e", e):
        plan = list(_manifest(private))
    plan[0]["deadline"] = 600
    return plan


def inspect(args: list[str]) -> int:
    """Historical design selection is independent of current activation readiness."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument(
        "--inspect-milestone", choices=["2026.10.716", "2026.10.717", "2026.10.718", e.MILESTONE]
    )
    parser.add_argument("--check-authority", action="store_true")
    options = parser.parse_args(args)
    if options.check_authority:
        with TemporaryDirectory(prefix="exp8332-authority-") as directory:
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
    """Reuse actual CLI execution while binding its invocation to the new authority."""
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
        e.bindings(),
        patch.object(qualified, "e", e),
        patch.object(qualified, "manifest", manifest),
    ):
        return int(qualified.main(args))
