"""REQ-VERIFY-8565: bounded validation owns only new code and private consumers.

Historical qualification is reused by hashes. This runner never invokes a full
repository suite or a nested historical controller, and never opens a model.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import signal
import sys
from tempfile import mkdtemp
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v736_contract_methods as e
from carnot.reporting import v736_contract_report as report
from carnot.reporting import v717_contract_runner as baseline
from carnot.reporting import v729_contract_runner as collection
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Collect original node IDs before disjoint serial children; preserve their exact union."""
    if private.absolute().is_relative_to(e.ROOT):
        raise ValueError("validation scratch must be private and outside the checkout")
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(baseline, "m", e):
        plan = list(baseline.manifest(private))
    consumer = next(r for r in plan if r["name"] == "private_E2E018_consumers")
    plan.remove(consumer)
    for spec in plan:
        if spec["name"] in {"owned_tests", "ruff_check", "ruff_format", "spec_coverage"}:
            spec["argv"] += e.TESTS[1:]
        if spec["name"] == "coverage_combine":
            spec["argv"] += ["--keep"]
    suites = [
        e.TESTS[0],
        e.TESTS[1],
        "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_restricted_decision_audit_8210.py",
    ]
    shards = []
    union = {}
    for suite in suites:
        nodes, receipt = collection.collect(private, suite)
        union[suite] = nodes
        shards.append(
            dict(
                name=receipt["name"],
                argv=receipt["argv"],
                deadline=60,
                expected=0,
                scope="owned",
                collected_node_ids=nodes,
            )
        )
        if suite in e.TESTS:
            continue
        size = 1 if "8210" in suite else 8
        for i in range(0, len(nodes), size):
            group = nodes[i : i + size]
            shards.append(
                dict(
                    name=f"{Path(suite).stem}_{i}",
                    argv=[
                        str(e.ROOT / ".venv/bin/pytest"),
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        "--no-cov",
                        "-vv",
                        *group,
                        "--basetemp=" + str(private / f"{Path(suite).stem}_{i}"),
                    ],
                    deadline=300,
                    expected=0,
                    scope="owned",
                    collected_node_ids=group,
                )
            )
    e.atomic_json(private / "original_node_union.json", dict(suites=union))
    return [*shards, *plan]


def controls(value: Json, raw: Path) -> list[Json]:
    """Fresh children reject missing primitives and rehashed summaries rather than trusting hashes."""
    receipts = []
    for mode in ["valid", "missing", "rehashed"]:
        changed = deepcopy(value)
        if mode == "missing":
            changed["primitive_reference"]["path"] = str(raw / "absent.json")
        if mode == "rehashed":
            changed["methods_ready_score"] = 1 - changed["methods_ready_score"]
            changed["reproducibility_checksum"] = e.checksum(changed)
        path = raw / (mode + ".json")
        e.atomic_json(path, changed)
        receipts.append(
            child(
                "cold_" + mode,
                [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
                raw / "controls",
                expected=int(mode != "valid"),
                deadline=120,
            )
        )
    return receipts


def publish(value: Json, output: Path, raw: Path) -> Json:
    """Validate exact candidate bytes before atomic exposure and retain every scanner finding."""
    rows: list[Json] = []

    def validate(candidate: Path) -> Json:
        commands = [
            ("terminal_cold", [str(e.ROOT / e.CLI), "--cold-replay", str(candidate)]),
            ("adversarial", ["scripts/adversarial_verify.py", "--json", str(candidate)]),
            (
                "strict_rows",
                ["scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)],
            ),
        ]
        for name, args in commands:
            receipt = child(
                name, [sys.executable, "-u", *args], raw / "terminal" / str(len(rows)), deadline=120
            )
            if name == "adversarial":
                document = json.loads(Path(receipt["stdout_path"]).read_bytes())
                receipt["adversarial_findings"] = document.get("reports", [])
            rows.append(receipt)
        return dict(passed=all(r["passed"] for r in rows[-3:]), checks=rows[-3:])

    publication = publish_primary(output, value, validate)
    downstream = report.consumer(output)
    e.atomic_json(
        raw / "terminal_validation.json",
        dict(
            **publication,
            publication=publication,
            checks=rows,
            downstream=downstream,
            normal_process_completion=True,
        ),
    )
    read_bound_sidecar(output, raw / "terminal_validation.json")
    e.progress("published", 1, 0)
    return dict(publication)


def main(argv: list[str] | None = None) -> int:
    """The CLI owns a finite task clock and keeps fixture writes outside historical authority."""
    e.progress("start_no_model_load_calls_0")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261016"], default="20261016")
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
        raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
        work = e.measure(args.root, raw)
        private = Path(mkdtemp(prefix="exp8565-validation-", dir="/var/tmp"))
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
        work["execution_manifest_reference"] = e.reference(raw / "execution_manifest.json")
        e.atomic_json(raw / "measurement.json", work)
        receipts = execute(plan, raw / "checks")
        validation = raw / "validation.json"
        e.atomic_json(validation, dict(receipts=receipts))
        value = report.build(work, e.reference(validation), raw, output)
        receipts += controls(value, raw)
        e.atomic_json(validation, dict(receipts=receipts))
        value = report.build(work, e.reference(validation), raw, output)
        publish(value, output, raw)
        return 0
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
