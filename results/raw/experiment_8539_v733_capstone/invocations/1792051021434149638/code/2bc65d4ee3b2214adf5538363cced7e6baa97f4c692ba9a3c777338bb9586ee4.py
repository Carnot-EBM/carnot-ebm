"""REQ-VERIFY-8539: reuse bounded validation and exact-byte terminal publication."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v730_capstone as shipped
from carnot.reporting import v733_capstone_evidence as e
from carnot.reporting.v733_capstone_report import build
from carnot.reporting.gatemate_continuity_8400 import budget

Json = dict[str, Any]
SUITES = shipped.SUITES


def manifest(private: Path) -> list[Json]:
    """Collect first and preserve the complete consumer union in disjoint serial shards."""
    with patch.object(shipped, "e", e):
        return list(shipped.manifest(private))


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, failed: bool = False
) -> int:
    """Reuse the shipped pipeline without launching any historical validation runner."""
    with patch.multiple(
        shipped.old,
        e=e,
        build=build,
        manifest=shipped._MANIFEST if control else manifest,
        child=shipped.complete_child,
        terminal_child=shipped.complete_child,
    ):
        return int(shipped._RUN(root, output, private, control=control, failed=failed))


def main(argv: list[str] | None = None) -> int:
    """The dated CLI owns disk-backed scratch and a hard 4800-second task deadline."""
    e.progress("start_no_model_load_current_LLM_invocations_0", 0, 14)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261015"], default="20261015")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--failed-child", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--worker-request", type=Path)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args(argv)
    if args.worker_request:
        if not args.worker_output:
            parser.error("worker output required")
        return e.worker_process(args.worker_request, args.worker_output)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private controls must publish outside the repository")
    with budget(4800), TemporaryDirectory(prefix="exp8539-owned-", dir="/var/tmp") as directory:
        return run(
            args.root,
            output,
            Path(directory),
            control=args.private_control,
            failed=args.failed_child,
        )
