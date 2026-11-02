"""REQ-VERIFY-8681: bounded original nodes and genuine retained child coverage."""

from __future__ import annotations

from pathlib import Path
import argparse
from datetime import datetime
import json
import signal
import shutil
from tempfile import mkdtemp
import time
from coverage import CoverageData
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import v717_contract_runner as commands
from carnot.reporting import v736_contract_runner as publication
from carnot.reporting import v738_contract_runner as lifecycle
from carnot.reporting import v743_contract_runner as custody
from carnot.reporting import v745_contract_methods as e
from carnot.reporting import v745_contract_report as report
from carnot.reporting.v709_execution import child, execute as run_children

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Collect once and freeze disjoint original and new node sets before execution."""
    if private.absolute().is_relative_to(e.ROOT):
        raise ValueError("validation scratch must be outside the checkout")
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(commands, "m", e):
        plan = list(commands.manifest(private))
    plan = [r for r in plan if r["name"] not in {"owned_tests", "private_E2E018_consumers"}]
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n    "
        + str(e.ROOT / "python/carnot/*")
        + "\n    "
        + str(e.ROOT / "scripts/*")
        + "\n[report]\nexclude_lines=\n"
    )
    union, shards = {}, []
    suites = (
        e.ORIGINAL_TESTS
        + e.TESTS
        + [
            "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            "tests/python/test_primary_publication_7928.py",
            "tests/python/test_restricted_decision_audit_8210.py",
        ]
    )
    for suite in suites:
        nodes, receipt = custody.collection.collect(private, suite)
        union[suite] = nodes
        shards.append(
            dict(
                name=receipt["name"],
                argv=[
                    sys.executable,
                    "-c",
                    "from pathlib import Path; import sys; from carnot.reporting.v743_contract_runner import verify_collection; raise SystemExit(not verify_collection(Path(sys.argv[1]),sys.argv[2]))",
                    str(private / "collection" / (receipt["name"] + ".receipt.json")),
                    e.sha256_file(private / "collection" / (receipt["name"] + ".receipt.json")),
                ],
                expected=0,
                deadline=30,
                scope="owned",
                collected_node_ids=nodes,
            )
        )
        size = len(nodes) if suite in e.ORIGINAL_TESTS + e.TESTS else 1 if "8210" in suite else 8
        for i in range(0, len(nodes), size):
            group = nodes[i : i + size]
            name = "test_" + Path(suite).stem + "_" + str(i)
            junit = private / (name + ".xml")
            shards.append(
                dict(
                    name=name,
                    argv=[
                        str(e.ROOT / ".venv/bin/coverage"),
                        "run",
                        "--rcfile=" + str(config),
                        "-m",
                        "pytest",
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        "--no-cov",
                        "-p",
                        "carnot.reporting.v745_private_inputs",
                        "--capture=tee-sys",
                        "-q",
                        *group,
                        "--basetemp=" + str(private / name),
                        "--junitxml=" + str(junit),
                    ],
                    expected=0,
                    deadline=900 if suite in e.ORIGINAL_TESTS else 600,
                    scope="owned",
                    collected_node_ids=group,
                    junit_path=str(junit),
                )
            )
        if suite == e.TESTS[-1]:
            shards += [
                dict(
                    name="producer_coverage_combine",
                    argv=[
                        str(e.ROOT / ".venv/bin/coverage"),
                        "combine",
                        "--rcfile=" + str(config),
                        "--keep",
                        str(private / "coverage_copies"),
                    ],
                    expected=0,
                    deadline=30,
                    scope="owned",
                ),
                dict(
                    name="producer_coverage_report",
                    argv=[
                        str(e.ROOT / ".venv/bin/coverage"),
                        "report",
                        "--rcfile=" + str(config),
                        "--show-missing",
                        "--fail-under=100",
                        "--include=" + ",".join(str(e.ROOT / p) for p in e.OWNED),
                    ],
                    expected=0,
                    deadline=30,
                    scope="owned",
                ),
            ]
    for row in plan:
        if row["name"] == "coverage_combine":
            row["argv"] += ["--keep", str(private / "coverage_copies")]
        if row["name"] in {"coverage_report", "coverage_json"}:
            row["argv"] += ["--include=" + ",".join(str(e.ROOT / p) for p in e.OWNED)]
    e.atomic_json(private / "original_node_union.json", dict(suites=union))
    return shards + plan


def execute(plan: list[Json], logs: Path) -> list[Json]:
    """Retain each finished child's untouched SQLite and JUnit before combining copies."""
    receipts = []
    if not any(s["name"] == "coverage_combine" for s in plan):
        return list(run_children(plan, logs))
    private = Path(next(s for s in plan if s["name"] == "coverage_combine")["argv"][-1]).parent
    for i, spec in enumerate(plan):
        e.progress("validation", i, len(plan) - i)
        receipts.append(
            child(
                spec["name"],
                spec["argv"],
                logs,
                deadline=spec["deadline"],
                expected=spec["expected"],
                scope=spec["scope"],
            )
        )
        if spec["name"].startswith("test_"):
            receipt = custody.retain_coverage(private, logs.parent / "coverage_shards" / str(i))
            junit = Path(spec.get("junit_path", "/absent"))
            if junit.is_file():
                retained_junit = logs.parent / "junit" / junit.name
                retained_junit.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(junit, retained_junit)
                receipts[-1]["junit_reference"] = e.reference(retained_junit)
            receipts[-1]["source_hashes"] = [e.reference(e.ROOT / p) for p in e.OWNED + e.TESTS]
            e.atomic_json(logs / (spec["name"] + ".receipt.json"), receipts[-1])
            # Preserve child evidence beside its command before any later combine.
            e.atomic_json(logs / (spec["name"] + ".custody.json"), receipt)
    retained = custody.retain_coverage(private, logs.parent / "coverage_shards" / "final")
    receipts.append(retained)
    return receipts


def original_authority(plan: list[Json], private: Path) -> Json:
    """Original nodes and the retained rejection line qualify their producer separately."""
    nodes = [
        n
        for s in plan
        if s["name"].startswith("collect_")
        for n in s["collected_node_ids"]
        if n.split("::")[0] in e.ORIGINAL_TESTS
    ]
    data = CoverageData(basename=str(private / ".coverage"))
    data.read()
    lines = data.lines(str(e.ROOT / "python/carnot/reporting/v744_contract_report.py")) or []
    return dict(
        name="original_contract_node_authority",
        passed=len(nodes) == len(set(nodes)) == 77 and 113 in lines,
        original_node_ids=nodes,
        original_rejection_line=113,
        rejection_line_executed=113 in lines,
        original_test_source_hashes=[e.reference(e.ROOT / p) for p in e.ORIGINAL_TESTS],
    )


def main(argv: list[str] | None = None) -> int:
    """Freeze configuration before measurements and use only bounded private children."""
    e.progress("start_no_model_load_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=lifecycle.date_value, default="20261101")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = report.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if args.private_control and output.is_relative_to(e.ROOT):
        parser.error("private controls require an output outside the checkout")
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        private = Path(mkdtemp(prefix="exp8681-validation-", dir="/var/tmp"))
        raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
        plan = (
            [
                dict(
                    name="private_child",
                    argv=[sys.executable, "-c", "print('private CPU control')"],
                    deadline=10,
                    expected=0,
                    scope="owned",
                )
            ]
            if args.private_control
            else manifest(private)
        )
        e.atomic_json(
            raw / "execution_manifest.json",
            dict(commands=plan, private_control=args.private_control, private_scratch=str(private)),
        )
        work = e.measure(args.root, raw, args.date)
        work["execution_manifest_reference"] = e.reference(raw / "execution_manifest.json")
        e.atomic_json(raw / "measurement.json", work)
        receipts = execute(plan, raw / "checks")
        if not args.private_control:
            receipts += [lifecycle.exact_union(plan), original_authority(plan, private)]
            for name in ["coverage.json", "original_node_union.json"]:
                e.atomic_json(raw / name, json.loads((private / name).read_bytes()))
        validation = raw / "validation.json"
        e.atomic_json(validation, dict(receipts=receipts))
        value = report.build(work, e.reference(validation), raw, output)
        with patch.multiple(publication, e=e, report=report):
            receipts += publication.controls(value, raw)
            e.atomic_json(validation, dict(receipts=receipts))
            value = report.build(work, e.reference(validation), raw, output)
            publication.publish(value, output, raw)
        return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
