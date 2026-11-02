"""REQ-VERIFY-8517: use bounded shipped validators and publish only reconstructed terminal bytes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import brier_online_learning_8517 as r
from carnot.reporting import capacity_learning_runner_8434 as shipped
from carnot.reporting import objective_publication_runner_8516 as historical_runner
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import reader_receipt
from carnot.reporting.v709_execution import child
from carnot.verify import objective_stream_8517 as e

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Collect the exact original historical union while covering only the new reader, worker and CLI."""
    private.mkdir(parents=True, exist_ok=True)
    with patch.multiple(historical_runner.e, OWNED=r.OWNED, TEST=r.TEST, CLI=r.CLI):
        plan = list(historical_runner.manifest(private))
    next(p for p in plan if p["name"] == "owned_tests")["deadline"] = 1500
    next(p for p in plan if p["name"] == "coverage_combine")["argv"].append("--keep")
    return plan


def qualify(plan: list[Json], logs: Path) -> list[Json]:
    """Archive each disjoint historical shard before the exact union consumer runs."""
    rows = []
    summary = next((p for p in plan if p["name"] == "historical_union"), None)
    shards = [p for p in plan if p["name"].startswith("historical_shard_")]

    def launch(spec: Json) -> Json:
        """Each child uses its own isolated root and retains the unchanged serial pytest command."""
        return dict(
            child(
                spec["name"],
                spec["argv"],
                logs,
                expected=spec["expected"],
                deadline=spec["deadline"],
                scope=spec["scope"],
            )
        )

    completed = False
    for spec in plan:
        if spec in shards:
            if completed:
                continue
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = [pool.submit(launch, shard) for shard in shards]
                measured = [future.result() for future in futures]
            rows.extend(measured)
            for receipt in measured:
                if summary:
                    destination = Path(summary["argv"][-1]) / (receipt["name"] + ".receipt.json")
                    atomic_json(destination, receipt)
            completed = True
        else:
            rows.append(launch(spec))
    if summary and "--historical-summary" in summary["argv"]:
        ledger = Path(summary["argv"][summary["argv"].index("--historical-summary") + 1])
        archived = logs.parent / "historical_collection"
        shutil.copytree(ledger.parent / "collection", archived, dirs_exist_ok=True)
        shutil.copyfile(ledger, logs.parent / "historical_collection.json")
        atomic_json(
            logs.parent / "collection_archive.json",
            [r.reference(p) for p in sorted(archived.iterdir())],
        )
    return rows


def run(
    root: Path,
    output: Path,
    private: Path,
    *,
    control: bool = False,
    measurement: Path | None = None,
) -> int:
    """Reuse unchanged private publication, scanners, negative controls and actual readiness readers."""
    with patch.multiple(shipped, r=r, manifest=manifest, qualify=qualify):
        status = int(shipped.run(root, output, private, control=control, measurement=measurement))
    value = json.loads(output.read_bytes())
    terminal = Path(value["terminal_validation_sidecar_path"])
    report = json.loads(terminal.read_bytes())
    report["objective_consumer"] = reader_receipt(
        r.TASK,
        output.parent,
        field="objective_stream_ready_score",
        expected=value["objective_stream_ready_score"],
    )
    atomic_json(terminal, report)
    return status


def main(argv: list[str] | None = None) -> int:
    """Separate evaluator and learner modes prevent private tests from launching recursive validation."""
    r.progress("start_no_model_load_current_llm_calls_0")
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in ("--historical-tests", "--historical-exec"):
        return int(
            historical_runner.historical(arguments[1:], script=arguments[0] == "--historical-exec")
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261014"], default="20261014")
    parser.add_argument("--root", type=Path, default=r.ROOT)
    parser.add_argument("--output", type=Path, default=r.ROOT / "results" / (r.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--resume-measurement", type=Path)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--crash", action="store_true")
    parser.add_argument("--release", type=Path)
    parser.add_argument("--retention", type=Path)
    parser.add_argument("--operand-output", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--historical-tests", action="store_true")
    parser.add_argument("--historical-summary", type=Path)
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--historical-exec", action="store_true")
    args, rest = parser.parse_known_args(argv)
    if args.historical_tests or args.historical_exec:
        return int(historical_runner.historical(rest, script=args.historical_exec))
    if args.historical_summary:
        return int(historical_runner.summarize(args.historical_summary, args.receipts))
    if rest:
        parser.error("unrecognized arguments: " + " ".join(rest))
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = r.replay(args.cold_replay)
        r.progress("replay_passed" if passed else "reduction_drift")
        return int(not passed)
    if args.worker or args.release or args.retention:
        try:
            cfg = json.loads(args.configuration.read_bytes())
            if args.release:
                issue = json.loads(args.release.read_bytes())
                before = dict(issue["state"], version=issue["state"]["version"] - 1)
                if issue["predictions"] != e.shipped.predictions(
                    cfg, before, before["cursor"] // 20
                ):
                    raise ValueError("issued_prediction_drift")
                atomic_json(
                    args.operand_output,
                    dict(
                        labels=e.shipped.released(cfg, issue["state"]),
                        released_wall_ns=time.time_ns(),
                    ),
                )
            elif args.retention:
                atomic_json(
                    args.operand_output, e.shipped.retention(cfg, e.shipped.steps(args.retention))
                )
            else:
                e.worker(cfg, args.worker, args.crash)
            return 0
        except (OSError, ValueError, KeyError, TypeError, AttributeError, TimeoutError):
            return 1
    if args.private_control and args.output.resolve().is_relative_to(r.ROOT):
        return 2
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8517-private-", dir="/var/tmp") as directory:
            return run(
                args.root,
                args.output.absolute(),
                Path(directory),
                control=args.private_control,
                measurement=args.resume_measurement,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
