"""REQ-VERIFY-8540: bounded serial validation preserves exact original assertions."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v734_contract_methods as e
from carnot.reporting import v729_contract_runner as shipped
from carnot.reporting import v734_historical_inputs as history
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
_MANIFEST = shipped.manifest


def manifest(private: Path) -> list[Json]:
    """Keep collected original unions in disjoint shards and include all new child coverage."""
    with patch.object(shipped, "e", e):
        plan = list(_MANIFEST(private))
    tests = next(p for p in plan if p["name"] == "owned_tests")
    tests["argv"].extend(e.TESTS[1:])
    tests["deadline"] = 1200
    for suite in e.TESTS[1:]:
        nodes, receipt = shipped.collect(private, suite)
        plan.insert(
            0,
            dict(
                name=receipt["name"],
                argv=receipt["argv"],
                expected=0,
                deadline=60,
                scope="owned",
                collected_node_ids=nodes,
            ),
        )
    tests["collected_node_ids"] = [
        node
        for p in plan
        if p["name"].startswith("collect_")
        for node in p["collected_node_ids"]
        if node.split("::")[0] in e.TESTS
    ]
    for check in plan:
        if check["name"] == "owned_tests" or check["name"].startswith("historical_"):
            check["argv"] = ["-vv" if arg == "-q" else arg for arg in check["argv"]]
        if check["name"] in {"ruff_check", "ruff_format", "spec_coverage"}:
            check["argv"].extend(e.TESTS[1:])
    return plan


def main(argv: list[str] | None = None) -> int:
    """A dated invocation owns its deadline; private controls cannot replace repository artifacts."""
    e.progress("start_no_model_load_LLM_loads_0_generations_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261015"], default="20261015")
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
        with TemporaryDirectory(prefix="exp8540-owned-", dir="/var/tmp") as directory:
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
    """Original replay readers receive a finite deadline while heartbeat supervision stays active."""
    if "--cold-replay" in argv:
        kwargs["deadline"] = 600
    return dict(child(name, argv, logs, **kwargs))
