"""REQ-VERIFY-8471: bounded disjoint historical suites retain every collected assertion."""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v725_contract_runner as shipped
from carnot.reporting import v729_contract_methods as e
from carnot.reporting import v729_historical_inputs as history
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
_MANIFEST = shipped.manifest


def collect(private: Path, suite: str) -> tuple[list[str], Json]:
    """Collect original node IDs before running them, so slow assertions cannot vanish."""
    receipt = child(
        "collect_" + Path(suite).stem,
        [
            str(e.ROOT / ".venv/bin/pytest"),
            "-n",
            "0",
            "-o",
            "addopts=",
            "--no-cov",
            "--collect-only",
            "-q",
            suite,
        ],
        private / "collection",
        deadline=60,
    )
    nodes = [
        line
        for line in Path(receipt["stdout_path"]).read_text().splitlines()
        if line.startswith(suite + "::")
    ]
    if not receipt["passed"] or not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("required_test_collection")
    return nodes, receipt


def manifest(private: Path) -> list[Json]:
    """Freeze real coverage and exact disjoint E2E shards; original CLI dates stay unchanged."""
    with patch.object(shipped, "e", e):
        plan = list(_MANIFEST(private))
    consumer = next(p for p in plan if p["name"] == "private_E2E018_consumers")
    suites = [p for p in consumer["argv"] if p.startswith("tests/")]
    suites += ["tests/python/test_restricted_decision_audit_8210.py"]
    plan.remove(consumer)
    collections, shards = [], []
    for suite in [e.TEST, *suites]:
        nodes, receipt = collect(private, suite)
        collections.append(
            dict(
                name=receipt["name"],
                argv=receipt["argv"],
                expected=0,
                deadline=60,
                scope="owned",
                collected_node_ids=nodes,
            )
        )
        if suite == e.TEST:
            continue
        size = 1 if suite.endswith("test_restricted_decision_audit_8210.py") else 8
        groups = [nodes[i : i + size] for i in range(0, len(nodes), size)]
        assert [n for g in groups for n in g] == nodes
        for i, group in enumerate(groups):
            shards.append(
                dict(
                    name=f"historical_{Path(suite).stem}_{i}",
                    argv=[
                        str(e.ROOT / ".venv/bin/pytest"),
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        "--no-cov",
                        "-q",
                        *group,
                        "--basetemp=" + str(private / f"{Path(suite).stem}-{i}"),
                    ],
                    expected=0,
                    deadline=300,
                    scope="owned",
                    collected_node_ids=group,
                )
            )
    next(p for p in plan if p["name"] == "owned_tests")["deadline"] = 1200
    return [*collections, plan[0], *shards, *plan[1:]]


def main(argv: list[str] | None = None) -> int:
    """Own only this invocation's deadline and private outputs; historical producers stay sealed."""
    e.progress("start_no_model_load_LLM_loads_0_generations_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261013"], default="20261013")
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
        with TemporaryDirectory(prefix="exp8471-owned-", dir="/var/tmp") as directory:
            with (
                patch.object(shipped, "e", e),
                patch.object(shipped, "manifest", manifest),
                patch.object(shipped, "child", bounded_child),
                patch.object(shipped.shipped, "child", bounded_child),
            ):
                return int(
                    shipped.run(args.root, output, Path(directory), control=args.private_control)
                )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)


def bounded_child(name: str, argv: list[str], logs: Path, **kwargs: Any) -> Json:
    """Eight isolated original readers need a finite complete replay with unchanged assertions."""
    if "--cold-replay" in argv:
        kwargs["deadline"] = 240
    return dict(child(name, argv, logs, **kwargs))
