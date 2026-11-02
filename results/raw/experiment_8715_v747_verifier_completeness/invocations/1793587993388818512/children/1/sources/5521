"""REQ-VERIFY-8513: unchanged consumers qualify bounded private terminal bytes."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_criterion_decision_runner as shipped
from carnot.reporting import v729_contract_runner as collected
from carnot.reporting import v732_strongest_baseline_audit as e
from carnot.reporting.primary_publication import reader_receipt

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Collect exact historical node unions before disjoint bounded assertion-preserving shards."""
    with patch.object(collected, "e", e):
        return list(collected.manifest(private))


def main(argv: list[str] | None = None) -> int:
    """Private control mode prevents recursive validation while exercising real publication."""
    e.progress("start_no_model_load_LLM_loads_0_generations_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261014"], default="20261014")
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
        with TemporaryDirectory(prefix="exp8513-owned-", dir="/var/tmp") as directory:
            with (
                patch.object(shipped, "e", e),
                patch.object(shipped, "manifest", manifest),
                patch.object(
                    shipped,
                    "reader_receipt",
                    lambda task, results, **kwargs: reader_receipt(
                        task, results, **dict(kwargs, field="comparator_inputs_ready_score")
                    ),
                ),
            ):
                return int(
                    shipped.run(args.root, output, Path(directory), control=args.private_control)
                )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
