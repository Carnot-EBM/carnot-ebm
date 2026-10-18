"""REQ-VERIFY-8458: narrow adapters reuse unchanged bounded validation and publication."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_feature_panel_runner as shipped
from carnot.reporting import v728_source_control_panel as e

Json = dict[str, Any]
_MANIFEST = shipped.manifest


def manifest(private: Path) -> list[Json]:
    """The inherited plan retains genuine serial child coverage and both private E2E suites."""
    with patch.object(shipped, "e", e):
        plan = list(_MANIFEST(private))
    next(s for s in plan if s["name"] == "owned_tests")["deadline"] = 600
    return plan


def main(argv: list[str] | None = None) -> int:
    """The dated invocation owns its cap and protects repository outputs in private control tests."""
    e.progress("start_no_model_load_LLM_loads_0_generations_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261012"], default="20261012")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = e.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8458-owned-", dir="/var/tmp") as directory:
            with patch.object(shipped, "e", e), patch.object(shipped, "manifest", manifest):
                return int(
                    shipped.run(args.root, output, Path(directory), control=args.private_control)
                )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
