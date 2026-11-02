"""REQ-VERIFY-8486: bounded new validation retains unchanged historical assertions."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import v726_feature_panel_runner as shipped
from carnot.reporting import v729_contract_runner as collected
from carnot.reporting import v730_source_control_qualification as e
from carnot.reporting.current_work_receipt import atomic_json

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Collect exact suites and retain their disjoint unions before any validation executes."""
    with patch.object(collected, "e", e):
        plan = list(collected.manifest(private))
        nodes, receipt = collected.collect(private, e.legacy.TEST)
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
    config = private / "coverage.ini"
    config.write_text(
        config.read_text().replace(
            "[report]", "    " + str(e.ROOT / e.legacy.OWNED[0]) + "\n[report]"
        )
    )
    cov = str(e.ROOT / ".venv/bin/coverage")
    original = []
    for i in range(0, len(nodes), 4):
        group = nodes[i : i + 4]
        original.append(
            dict(
                name=f"original_8458_{i // 4}",
                argv=[
                    cov,
                    "run",
                    "--rcfile=" + str(config),
                    str(e.ROOT / e.CLI),
                    "--original-tests",
                    *group,
                    "--basetemp=" + str(private / f"original-{i}"),
                ],
                expected=0,
                deadline=360,
                scope="owned",
                collected_node_ids=group,
            )
        )
    assert [n for s in original for n in s["collected_node_ids"]] == nodes
    position = next(i for i, spec in enumerate(plan) if spec["name"] == "coverage_combine")
    plan[position:position] = original
    return plan


def original_worker(argv: list[str], *, tests: bool) -> int:
    """Original CLI descendants receive the same sealed historical reads as their parent."""
    popen = subprocess.Popen

    def start(args: Any, *pos: Any, **kw: Any) -> Any:
        if isinstance(args, (list, tuple)) and str(e.ROOT / e.legacy.CLI) in args:
            index = args.index(str(e.ROOT / e.legacy.CLI))
            args = [
                str(e.ROOT / ".venv/bin/python"),
                "-u",
                str(e.ROOT / e.CLI),
                "--original-exec",
                *args[index:],
            ]
        return popen(args, *pos, **kw)

    with TemporaryDirectory(prefix="exp8486-original-", dir="/var/tmp") as directory:
        with e.original_binding(Path(directory)), patch.object(subprocess, "Popen", start):
            if tests:
                import pytest

                return int(pytest.main(["-n", "0", "-o", "addopts=", "--no-cov", "-q", *argv]))
            previous = sys.argv
            sys.argv = argv
            try:
                runpy.run_path(argv[0], run_name="__main__")
            except SystemExit as error:
                return int(error.code) if isinstance(error.code, int) else 0
            finally:
                sys.argv = previous
    return 0


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """The genuine negative producer fixture becomes a required fresh validation receipt."""
    with patch.object(shipped, "e", e), patch.object(shipped, "manifest", manifest):
        return int(shipped.run(root, output, private, control=control))


def main(argv: list[str] | None = None) -> int:
    """The dated task owns its deadline and publishes private controls only outside the checkout."""
    e.progress("start_no_model_load_LLM_loads_0_generations_0")
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in {"--original-tests", "--original-exec"}:
        return original_worker(arguments[1:], tests=arguments[0] == "--original-tests")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261013"], default="20261013")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--history-request", type=Path)
    parser.add_argument("--history-output", type=Path)
    args = parser.parse_args(arguments)
    if args.history_request:
        if not args.history_output:
            parser.error("history output required")
        return int(e.legacy.history_reader.worker(args.history_request, args.history_output))
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
        with TemporaryDirectory(prefix="exp8486-owned-", dir="/var/tmp") as directory:
            return run(args.root, output, Path(directory), control=args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
