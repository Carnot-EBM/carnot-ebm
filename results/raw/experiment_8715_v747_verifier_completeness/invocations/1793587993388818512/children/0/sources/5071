"""REQ-VERIFY-8489: dated bounded children retain every unchanged historical test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import objective_checkpoint_8489 as e
from carnot.reporting import objective_checkpoint_runner_8461 as shipped
from carnot.reporting import v729_contract_runner as collection
from carnot.reporting import v727_historical_inputs as history
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.historical_consumer_roots_8402 import assertion_sources, parse_outcomes
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
_MANIFEST = e.legacy.manifest
STORE_TESTS = [
    "tests/python/test_block_feedback_state_8433.py",
    "tests/python/test_capacity_online_learning_8434.py",
]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]


def manifest(private: Path) -> list[Json]:
    """Collect exact nodes before freezing a complete partition and genuine coverage commands."""
    with e.binding():
        plan = list(_MANIFEST(private))
    suites = [e.TEST, *e.ORIGINAL_TESTS, *STORE_TESTS]
    for spec in plan:
        if spec["name"].startswith("private_E2E"):
            suites += [a for a in spec["argv"] if a.startswith("tests/")]
    nodes: dict[str, list[str]] = {}
    for suite in suites:
        nodes[suite], receipt = collection.collect(private, suite)
        plan.insert(
            0,
            dict(
                name=receipt["name"],
                argv=receipt["argv"],
                expected=0,
                deadline=60,
                scope="owned",
                collected_node_ids=nodes[suite],
            ),
        )
    historical = [n for suite in STORE_TESTS for n in nodes[suite]]
    groups = [nodes[STORE_TESTS[0]], nodes[STORE_TESTS[1]][:12], nodes[STORE_TESTS[1]][12:]]
    assert [n for group in groups for n in group] == historical
    assert len(historical) == len(set(historical))
    atomic_json(
        private / "historical_collection.json",
        dict(
            node_ids=historical,
            shards=groups,
            union_equal=True,
            disjoint=True,
            total_deadline_s=1800,
            shard_deadline_s=600,
            heartbeat_s=30,
            assertion_sources=assertion_sources(historical),
        ),
    )
    cov = str(e.ROOT / ".venv/bin/coverage")
    rc = "--rcfile=" + str(private / "coverage.ini")
    owned = next(p for p in plan if p["name"] == "owned_tests")
    owned["argv"] = [
        cov,
        "run",
        rc,
        "-m",
        "pytest",
        *COMMON,
        *nodes[e.TEST],
        "--basetemp=" + str(private / "owned-tests"),
    ]
    owned["deadline"] = 900
    original = dict(
        name="original_objective_tests",
        expected=0,
        deadline=900,
        scope="owned",
        argv=[
            cov,
            "run",
            rc,
            str(e.ROOT / e.CLI),
            "--original-tests",
            *COMMON,
            *[n for suite in e.ORIGINAL_TESTS for n in nodes[suite]],
            "--basetemp=" + str(private / "original-tests"),
        ],
    )
    old = next(p for p in plan if p["name"] == "historical_store_regressions")
    index = plan.index(old)
    shards = [
        dict(
            name=f"historical_store_shard_{i}",
            expected=0,
            deadline=600,
            scope="owned",
            collected_node_ids=group,
            argv=[
                cov,
                "run",
                rc,
                str(e.ROOT / e.CLI),
                "--store-tests",
                *COMMON,
                "-vv",
                *group,
                "--basetemp=" + str(private / f"historical-{i}"),
            ],
        )
        for i, group in enumerate(groups)
    ]
    old["argv"] = [
        cov,
        "run",
        rc,
        str(e.ROOT / e.CLI),
        "--shard-summary",
        str(private / "historical_collection.json"),
        "--shard-receipts",
        str(private / "shard_receipts"),
    ]
    old["deadline"] = 30
    plan[index:index] = [original, *shards]
    for row in plan:
        if row["name"].startswith("historical_store_shard_"):
            row["receipt_directory"] = str(private / "shard_receipts")
    return plan


def summarize(ledger_path: Path, receipts_path: Path) -> int:
    """The typed historical consumer checks passed identities independently of process exits."""
    try:
        ledger = json.loads(ledger_path.read_bytes())
        receipts = [
            json.loads((receipts_path / f"historical_store_shard_{i}.receipt.json").read_bytes())
            for i in range(len(ledger["shards"]))
        ]
        outcomes = [parse_outcomes(Path(r["stdout_path"])) for r in receipts]
        union = [n for group in ledger["shards"] for n in group]
        ready = union == ledger["node_ids"] and len(union) == len(set(union)) and bool(union)
        ready = ready and sum(r["duration_s"] for r in receipts) <= 1800
        ready = ready and all(
            r["passed"]
            and not r["timed_out"]
            and o["passed"] == group
            and not any(o[k] for k in ("failed", "error", "skipped"))
            for r, o, group in zip(receipts, outcomes, ledger["shards"], strict=True)
        )
        e.progress(
            "historical_shard_summary", len(union) if ready else 0, 0 if ready else len(union)
        )
        return int(not ready)
    except (OSError, ValueError, KeyError, TypeError):
        return 1


def execute(plan: list[Json], logs: Path) -> list[Json]:
    """Durable shard receipts let a separate bounded process reject incomplete historical work."""
    rows = []
    for i, spec in enumerate(plan):
        e.progress("validation", i, len(plan) - i)
        receipt = child(
            spec["name"],
            spec["argv"],
            logs,
            deadline=spec["deadline"],
            expected=spec["expected"],
            heartbeat=30,
        )
        rows.append(receipt)
        if "receipt_directory" in spec:
            atomic_json(Path(spec["receipt_directory"]) / (spec["name"] + ".receipt.json"), receipt)
    return rows


def original_worker(argv: list[str], *, store: bool = False, tests: bool = True) -> int:
    """Private historical authority wraps original CLI descendants without changing their assertions."""
    import pytest
    from carnot.reporting.v721_capstone_evidence import frozen_inputs

    popen = subprocess.Popen

    def start(args: Any, *pos: Any, **kw: Any) -> Any:
        original_cli = str(e.ROOT / e.ORIGINAL_OWNED[-1])
        if isinstance(args, (list, tuple)) and original_cli in args:
            index = args.index(original_cli)
            args = [*args[:index], str(e.ROOT / e.CLI), "--original-exec", *args[index:]]
        return popen(args, *pos, **kw)

    with TemporaryDirectory(prefix="exp8489-original-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        context = (
            frozen_inputs(history.custody(e.ROOT, scratch / "authority"), scratch / "consumer")
            if store
            else e.original_binding(scratch)
        )
        with context, patch.object(subprocess, "Popen", start):
            if tests:
                return int(pytest.main(argv if "--no-cov" in argv else [*COMMON, *argv]))
            previous = sys.argv
            sys.argv = argv
            try:
                runpy.run_path(argv[0], run_name="__main__")
            except SystemExit as error:
                return int(error.code) if isinstance(error.code, int) else 0
            finally:
                sys.argv = previous
    return 0


def run(root: Path, output: Path, private: Path, *, validate: bool = True) -> int:
    """Shipped publication retains owned failures and atomically validates the terminal candidate."""
    with (
        e.binding(),
        patch.object(e.legacy, "manifest", manifest if validate else lambda path: []),
        patch.object(shipped, "execute", execute),
    ):
        return int(shipped.run(root, output, private, validate=validate))


def main(argv: list[str] | None = None) -> int:
    """The dated task uses private disk scratch and delegates only unchanged worker mechanics."""
    e.progress("start_no_model_load_current_LLM_calls_0")
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in ("--original-tests", "--store-tests", "--original-exec"):
        return original_worker(
            arguments[1:],
            store=arguments[0] == "--store-tests",
            tests=arguments[0] != "--original-exec",
        )
    if any(a in arguments for a in ("--worker", "--state-replay")):
        return int(shipped.main(arguments))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261013"], default="20261013")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--shard-summary", type=Path)
    parser.add_argument("--shard-receipts", type=Path)
    args = parser.parse_args(arguments)
    if args.shard_summary:
        if not args.shard_receipts:
            parser.error("shard receipts required")
        return summarize(args.shard_summary, args.shard_receipts)
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
        with TemporaryDirectory(prefix="exp8489-owned-", dir="/var/tmp") as directory:
            return run(args.root, output, Path(directory), validate=not args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
