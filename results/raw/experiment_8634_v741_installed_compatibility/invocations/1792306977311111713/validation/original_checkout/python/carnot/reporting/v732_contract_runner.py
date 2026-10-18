"""REQ-VERIFY-8512: original assertions qualify exact terminal bytes in bounded children."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v732_contract_methods as e
from carnot.reporting import v729_contract_runner as shipped
from carnot.reporting import v732_historical_inputs as history
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
_MANIFEST = shipped.manifest


def manifest(private: Path) -> list[Json]:
    """Retain exact original node unions, assertions, private roots and historical CLI dates."""
    with patch.object(shipped, "e", e):
        return list(_MANIFEST(private))


def main(argv: list[str] | None = None) -> int:
    """Own only this invocation's deadline and private outputs; historical producers stay sealed."""
    e.progress("start_no_model_load_LLM_loads_0_generations_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261014"], default="20261014")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--history-request", type=Path)
    parser.add_argument("--history-output", type=Path)
    args = parser.parse_args(argv)
    if args.history_request:
        if not args.history_output:
            parser.error("history output required")
        return int(history.worker(args.history_request, args.history_output))
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
        with TemporaryDirectory(prefix="exp8512-owned-", dir="/var/tmp") as directory:
            with (
                patch.object(shipped.shipped, "e", e),
                patch.object(shipped.shipped, "manifest", manifest),
                patch.object(shipped.shipped, "child", bounded_child),
                patch.object(shipped.shipped.shipped, "child", bounded_child),
            ):
                return int(
                    shipped.shipped.run(
                        args.root, output, Path(directory), control=args.private_control
                    )
                )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)


def bounded_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """Eight isolated original readers need a finite complete replay with unchanged assertions."""
    if "--cold-replay" in argv:
        kwargs["deadline"] = 600
    return dict(child(name, argv, logs, **kwargs))
