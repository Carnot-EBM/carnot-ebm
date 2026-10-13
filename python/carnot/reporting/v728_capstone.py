"""REQ-VERIFY-8470: dated narrow adapter preserves bounded private validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_capstone as old
from carnot.reporting import v728_capstone_evidence as e
from carnot.reporting.v728_capstone_report import build
from carnot.reporting.gatemate_continuity_8400 import budget

Json = dict[str, Any]
_MANIFEST = old.manifest


def manifest(private: Path) -> list[Json]:
    """The unchanged serial plan collects genuine child shards and required private E2E checks."""
    with patch.object(old, "e", e):
        return list(_MANIFEST(private))


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, failed: bool = False
) -> int:
    """Reuse the qualified pipeline so failed owned checks see the same publication validators."""
    with patch.multiple(old, e=e, build=build, manifest=manifest):
        return int(_RUN(root, output, private, control=control, failed=failed))


_RUN = old.run


def main(argv: list[str] | None = None) -> int:
    """Keep historical reader globals intact during worker dispatch and frozen replay."""
    e.progress("start_no_model_load_current_LLM_invocations_0", 0, 14)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261013"], default="20261013")
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
        return e.worker_process(args.worker_request, args.worker_output)
    if args.deliberate_error:
        e.progress("deliberate_error")
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private controls must publish outside the repository")
    with budget(4800), TemporaryDirectory(prefix="exp8470-owned-", dir="/var/tmp") as directory:
        return run(
            args.root,
            output,
            Path(directory),
            control=args.private_control,
            failed=args.failed_child,
        )
