"""REQ-VERIFY-8490: validate a stopped causal study through unchanged public readers."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import signal
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import brier_online_learning_8490 as e
from carnot.reporting import v717_contract_runner as checks
from carnot.reporting import v726_feature_panel_runner as consumers
from carnot.reporting import v729_contract_runner as collection
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.historical_consumer_roots_8402 import assertion_sources
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Use shipped scoped checks; coverage measures only this reader and its real CLI."""
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(checks, "m", e):
        plan = checks.manifest(private)
    next(r for r in plan if r["name"] == "owned_tests")["deadline"] = 600
    next(r for r in plan if r["name"] == "coverage_combine")["argv"].append("--keep")
    plan.insert(
        2,
        dict(
            name="private_E2E021",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(private / "e2e021"),
            ],
            expected=0,
            deadline=300,
            scope="owned",
        ),
    )
    historical = [r for r in plan if r["name"].startswith("private_E2E")]
    plan = [r for r in plan if r not in historical]
    suites = [e.TEST, *[a for r in historical for a in r["argv"] if a.startswith("tests/")]]
    nodes, shards = [], []
    for suite in suites:
        collected, receipt = collection.collect(private, suite)
        nodes.extend(collected)
        if suite == e.TEST:
            next(r for r in plan if r["name"] == "owned_tests")["collected_node_ids"] = collected
            continue
        size = 1 if "8210" in suite else 8
        for i in range(0, len(collected), size):
            group = collected[i : i + size]
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
                        "--basetemp=" + str(private / (Path(suite).stem + str(i))),
                    ],
                    collected_node_ids=group,
                    expected=0,
                    deadline=600,
                    scope="owned",
                )
            )
    union = [
        *next(r for r in plan if r["name"] == "owned_tests")["collected_node_ids"],
        *[n for r in shards for n in r["collected_node_ids"]],
    ]
    if union != nodes or len(union) != len(set(union)):
        raise ValueError("historical_collection_union")
    atomic_json(
        private / "collection.json",
        dict(
            node_ids=nodes,
            shards=[r["collected_node_ids"] for r in [plan[0], *shards]],
            assertion_sources=assertion_sources(nodes),
            union_equal=True,
            disjoint=True,
        ),
    )
    return [plan[0], *shards, *plan[1:]]


def run(root: Path, output: Path, private: Path, control: bool = False) -> int:
    """Retain failed qualification and actual validation without permitting any scientific positive."""
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    work = e.measure(root, raw, private)
    plan = (
        [
            dict(
                name="private_control",
                argv=[sys.executable, "-u", "-c", "print('actual private CLI validation')"],
                expected=0,
                deadline=30,
                scope="owned",
            )
        ]
        if control
        else manifest(private)
    )
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=plan,
            owned=e.OWNED,
            task_cap_s=4800,
            validation_reserve_s=900,
            full_repository_suite="not_run",
        ),
    )
    receipts = execute(plan, raw / "validation")
    if (private / "collection.json").is_file():
        shutil.copyfile(private / "collection.json", raw / "collection.json")
        shutil.copytree(private / "collection", raw / "collection")
    if (private / "coverage.json").is_file():
        copies = raw / "coverage_shards"
        copies.mkdir()
        for source in [*private.glob(".coverage*"), private / "coverage.json"]:
            shutil.copyfile(source, copies / source.name)
        work["coverage_refs"] = [e.reference(p) for p in sorted(copies.iterdir())]
    atomic_json(raw / "measurement.json", work)
    candidate = raw / "preterminal.json"
    atomic_json(candidate, e.build(work, receipts, output))
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(
            child(name, [*argv, *args], raw / "controls", expected=expected, deadline=120)
        )
    changed = deepcopy(e.read(e.reference(candidate)))
    changed["objective_stream_ready_score"] = 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    tamper = raw / "rehashed_tamper.json"
    atomic_json(tamper, changed)
    receipts.append(
        child(
            "rehashed_tamper",
            [*argv, "--cold-replay", str(tamper)],
            raw / "controls",
            expected=1,
            deadline=120,
        )
    )
    atomic_json(candidate, e.build(work, receipts, output))
    scanned, findings = consumers.scan(candidate, raw / "preterminal_validators")
    receipts.extend(scanned)
    work.update(adversarial_findings=findings, ended_monotonic_ns=time.monotonic_ns())
    atomic_json(raw / "measurement.json", work)
    value = e.build(work, receipts, output)

    def validator(path: Path) -> Json:
        """Actual scanners and cold reconstruction must approve the exact final bytes."""
        scans, flags = consumers.scan(path, raw / "terminal_validators")
        return dict(
            passed=e.replay(path) and all(r["passed"] for r in scans),
            findings=flags,
            receipts=scans,
        )

    publication = publish_primary(output, value, validator)
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            consumer=reader_receipt(
                e.TASK, output.parent, field="objective_stream_ready_score", expected=0
            ),
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    e.progress("finished", 0, 6000)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Private controls cannot recurse into their own tests or replace public output."""
    e.progress("start_no_model_load_current_llm_calls_0")
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
        e.progress("replay_passed" if passed else "reduction_drift")
        return int(not passed)
    if args.private_control and args.output.resolve().is_relative_to(
        (e.ROOT / "results").resolve()
    ):
        parser.error("private control requires output outside repository results")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        with TemporaryDirectory(prefix="exp8490-", dir="/var/tmp") as directory:
            return run(args.root, args.output, Path(directory), args.private_control)
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
