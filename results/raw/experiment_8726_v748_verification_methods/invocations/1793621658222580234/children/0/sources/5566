"""REQ-VERIFY-8578: bounded original consumer unions and unchanged atomic publication."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from tempfile import mkdtemp
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import v736_capstone_evidence as e
from carnot.reporting import v736_capstone_report as report
from carnot.reporting import v736_contract_runner as shipped
from carnot.reporting.gatemate_continuity_8400 import budget
from carnot.reporting.v709_execution import child, execute

Json = dict[str, Any]
SUITES = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def manifest(private: Path) -> list[Json]:
    """Collect original IDs first and preserve disjoint bounded children under original authority."""
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(shipped.baseline, "m", e):
        plan = list(shipped.baseline.manifest(private))
    config = private / "coverage.ini"
    config.write_text(config.read_text().replace("[run]\n", "[run]\ncore=sysmon\n", 1))
    plan = [r for r in plan if r["name"] != "private_E2E018_consumers"]
    collections, shards = [], []
    for suite in [e.TEST, *SUITES]:
        nodes, receipt = shipped.collection.collect(private, suite)
        collections.append(
            dict(
                name=receipt["name"],
                argv=receipt["argv"],
                deadline=60,
                expected=0,
                scope="owned",
                collected_node_ids=nodes,
            )
        )
        if suite == e.TEST:
            next(r for r in plan if r["name"] == "owned_tests").update(
                suite=suite, original_node_ids=nodes, shard_node_ids=nodes
            )
            continue
        size = 1 if "8210" in suite else 8
        for index in range(0, len(nodes), size):
            group = nodes[index : index + size]
            shards.append(
                dict(
                    name=f"{Path(suite).stem}_{index}",
                    argv=[
                        str(e.ROOT / ".venv/bin/pytest"),
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        "--no-cov",
                        "-q",
                        *group,
                        "--basetemp=" + str(private / f"{Path(suite).stem}_{index}"),
                    ],
                    deadline=300,
                    expected=0,
                    scope="owned",
                    suite=suite,
                    shard_node_ids=group,
                )
            )
    next(r for r in plan if r["name"] == "owned_tests")["deadline"] = 600
    next(r for r in plan if r["name"] == "coverage_combine")["argv"] += ["--keep"]
    return [*collections, plan[0], *shards, *plan[1:]]


def controls(value: Json, raw: Path) -> list[Json]:
    """Actual fresh processes distinguish valid, missing and rehashed claims."""
    receipts = []
    for mode in ["valid", "missing", "rehashed"]:
        changed = deepcopy(value)
        if mode == "missing":
            changed["primitive_reference"]["path"] = str(raw / "absent.json")
        if mode == "rehashed":
            changed["task_count"] = 13
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


def main(argv: list[str] | None = None) -> int:
    """Own one finite invocation, with private failure controls that cannot replace history."""
    e.progress("start_no_model_load_current_LLM_calls_0", 0, 14)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261016"], default="20261016")
    parser.add_argument("--root", type=Path, default=e.ROOT)
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--private-control", action="store_true")
    parser.add_argument("--failed-child", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    parser.add_argument("--sample-output", type=Path)
    args = parser.parse_args(argv)
    if args.sample_output:
        return e.sample_worker(args.sample_output)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = report.replay(args.cold_replay)
        e.progress("replay_passed" if passed else "replay_rejected", int(passed), 0)
        return int(not passed)
    output = args.output.absolute()
    if (args.private_control or args.failed_child) and output.is_relative_to(e.ROOT):
        parser.error("private controls require an output outside the checkout")
    with budget(4800):
        raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
        work = e.measure(args.root, raw)
        private = Path(mkdtemp(prefix="exp8578-validation-", dir="/var/tmp"))
        plan = (
            [
                dict(
                    name="private_child",
                    argv=[
                        sys.executable,
                        "-c",
                        "raise SystemExit(7)"
                        if args.failed_child
                        else "print('private CPU control')",
                    ],
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
        work["publication_receipt"] = child(
            "publication_gate",
            [sys.executable, "-u", "scripts/publication_gate.py", "--json"],
            raw / "publication",
            deadline=60,
        )
        work["publication"] = json.loads(
            Path(work["publication_receipt"]["stdout_path"]).read_bytes()
        )
        e.atomic_json(raw / "measurement.json", work)
        receipts = execute(plan, raw / "checks")
        validation = raw / "validation.json"
        e.atomic_json(validation, dict(receipts=receipts))
        value = report.build(work, e.reference(validation), raw, output)
        receipts += controls(value, raw)
        e.atomic_json(validation, dict(receipts=receipts))
        value = report.build(work, e.reference(validation), raw, output)
        with patch.multiple(shipped, e=e, report=report):
            shipped.publish(value, output, raw)
        return 0
