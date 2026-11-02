"""REQ-VERIFY-8488: bounded genuine validation uses unchanged publication consumers."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_criterion_decision_runner as shipped
from carnot.reporting import v729_contract_runner as collected
from carnot.reporting import v730_source_information_audit as e

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Collect complete test rosters before disjoint bounded historical shards execute."""
    with patch.object(collected, "e", e):
        return list(collected.manifest(private))


def main(argv: list[str] | None = None) -> int:
    """The invocation owns its deadline and confines private control output to private paths."""
    e.progress("start_no_model_load_LLM_loads_0_generations_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261013"], default="20261013")
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
        parser.error("private control outputs must be outside the repository")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8488-owned-", dir="/var/tmp") as directory:
            with patch.object(shipped, "e", e), patch.object(shipped, "manifest", manifest):
                return int(
                    shipped.run(args.root, output, Path(directory), control=args.private_control)
                )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
