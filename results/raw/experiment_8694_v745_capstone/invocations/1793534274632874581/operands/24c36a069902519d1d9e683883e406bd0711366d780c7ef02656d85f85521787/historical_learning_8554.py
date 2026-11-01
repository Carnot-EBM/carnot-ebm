"""REQ-REPORT-8554: qualify measured nodes through unchanged historical adapters."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import xml.etree.ElementTree as ET

from carnot.reporting import historical_learning_8541 as old
from carnot.reporting import historical_learning_report_8541 as report
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_8554_v735_historical_learning_qualification"
TASK = "exp8554-historical-learning-qualification"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_historical_learning_8554.py"
OWNED = [
    "python/carnot/reporting/historical_learning_8554.py",
    "python/carnot/reporting/historical_learning_runner_8554.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
PIN = dict(
    path=str(ROOT / "results" / (old.NAME + ".json")),
    sha256="sha256:9321fc009824b58b7345df2212ea7d5ebef4cea6df0db7acc0b67359edb89d75",
)
INTERRUPTED = "tests/python/test_capacity_online_learning_8434.py::test_retention_child_failure_and_restart_mismatch"


def original() -> tuple[Json, Json]:
    """The fixed V734 producer hash independently freezes the union and its authority."""
    value = old.read(PIN)
    work = old.read(value["primitive_reference"])
    for ref in work["sources"]:
        if old.reference(Path(ref["path"])) != ref:
            raise ValueError("original_source_hash")
    for row in work["receipts"]:
        for stream in ("stdout", "stderr"):
            if old.reference(Path(row[stream + "_path"]))["sha256"] != row[stream + "_sha256"]:
                raise ValueError("original_log_hash")
    if (
        len(work["nodes"]) != 158
        or work["nodes"] != old.original_union()[1]
        or work["authorities"] != old.authorities(work["nodes"])
    ):
        raise ValueError("original_union_authority")
    return value, work


def partition(nodes: list[str]) -> list[list[str]]:
    """Small capacity children isolate expensive exit and retention fixtures without dropping tests."""
    if len(nodes) != len(set(nodes)):
        raise ValueError("duplicate_nodes")
    groups: list[list[str]] = []
    for suite in dict.fromkeys(n.split("::")[0] for n in nodes):
        pending: list[str] = []
        for node in (n for n in nodes if n.split("::")[0] == suite):
            isolated = "8434" in suite and ("process_exit" in node or "retention" in node)
            if pending and (isolated or ("8434" in suite and len(pending) == 4)):
                groups.append(pending)
                pending = []
            if isolated:
                groups.append([node])
            else:
                pending.append(node)
        if pending:
            groups.append(pending)
    return groups


def node_child(
    name: str, nodes: list[str], scratch: Path, raw: Path, deadline: float = 480
) -> Json:
    """Retain actual per-node pytest times while the old adapter supplies sealed authority."""
    scratch.mkdir(parents=True, exist_ok=True)
    xml = raw / (name + ".xml")
    suite = nodes[0].split("::")[0]
    argv = node_argv(name, nodes, scratch, raw)
    row = child(name, argv, raw, deadline=deadline, heartbeat=30)
    times = node_times(xml, nodes)
    row.update(
        collected_node_ids=nodes,
        authority=old.authorities(nodes)[suite],
        node_elapsed_s=times,
        junit_reference=old.reference(xml) if xml.exists() else None,
    )
    return row


def node_argv(name: str, nodes: list[str], scratch: Path, raw: Path) -> list[str]:
    """One command constructor keeps measured workers and replay contracts identical."""
    return [
        sys.executable,
        "-u",
        str(ROOT / old.CLI),
        "--suite",
        nodes[0].split("::")[0],
        *old.COMMON,
        *nodes,
        "--junitxml=" + str(raw / (name + ".xml")),
        "--basetemp=" + str(scratch / "pytest"),
    ]


def node_times(xml: Path, nodes: list[str]) -> dict[str, float]:
    """Read measured pytest times directly so repaired summaries cannot invent faster nodes."""
    times = {}
    if xml.exists():
        for case in ET.parse(xml).iter("testcase"):
            for node in nodes:
                if case.attrib["name"] == node.split("::")[-1]:
                    times[node] = float(case.attrib["time"])
    return times


def exact_union(nodes: list[str], receipts: list[Json]) -> Json:
    """Canonical commands, original authority and genuine terminal outcomes must all agree."""
    normalized = []
    valid = True
    for row in receipts:
        clone = dict(row, argv=list(row["argv"]))
        xml = Path(row["stdout_path"]).with_suffix(".xml")
        expected = "--junitxml=" + str(xml)
        valid = valid and clone["argv"][-2:-1] == [expected]
        clone["argv"] = [a for a in clone["argv"] if a != expected]
        group = row["collected_node_ids"]
        suite = group[0].split("::")[0] if group else ""
        valid = valid and row.get("authority") == old.authorities(nodes).get(suite)
        valid = valid and set(row.get("node_elapsed_s", {})) == set(group)
        valid = valid and row.get("node_elapsed_s") == node_times(xml, group)
        normalized.append(clone)
    value = old.union(nodes, normalized)
    value["passed"] = bool(
        value["passed"] and valid and sum(r["duration_s"] for r in receipts) <= 2400
    )
    for outcome in value["terminal_node_outcomes"]:
        receipt = next(r for r in receipts if r["name"] == outcome["child"])
        outcome["elapsed_s"] = receipt["node_elapsed_s"].get(outcome["node_id"])
    return value


def contract(work: Json, raw: Path) -> bool:
    """A saved plan cannot authorize replacement commands, skipped checks or repaired exits."""
    from carnot.reporting.historical_learning_runner_8554 import commands

    expected = commands(work, raw)
    rows = work["checks"]
    from carnot.reporting.primary_publication import reader_receipt

    consumer = work.get("accounting_consumer", {})
    consumer_ok = bool(
        consumer.get("passed")
        and consumer
        == reader_receipt(
            TASK,
            raw / "consumer/results",
            field="historical_learning_validation_ready_score",
            expected=0,
        )
    )
    return bool(
        consumer_ok
        and set(expected) == {r["name"] for r in rows}
        and len(rows) == len(expected)
        and all(
            r["argv"] == expected[r["name"]][0]
            and r["expected_exit"] == r["actual_exit"] == r["exit_code"] == expected[r["name"]][1]
            and r["passed"]
            and not r["timed_out"]
            and 0 < r["deadline_s"] <= 480
            and work["start_ns"]
            <= r["started_monotonic_ns"]
            <= r["ended_monotonic_ns"]
            <= work["end_ns"]
            for r in rows
        )
    )


def build(work: Json, raw: Path, output: Path) -> Json:
    """Reuse the shipped reducer with explicit metadata and union adapters; science credit stays zero."""
    adapter = SimpleNamespace(
        **dict(vars(old), TASK=TASK, OWNED=OWNED, TEST=TEST, union=exact_union)
    )
    with patch.object(report, "e", adapter), patch.object(report, "check_contract", contract):
        value = report.build(work, raw, output)
    value["raw_shard_hashes"] = [
        old.reference(raw / "measurement.json"),
        *([work["coverage"]] if work["coverage"] else []),
        *[r["junit_reference"] for r in work["receipts"] if r.get("junit_reference")],
    ]
    value.update(
        experiment_id=8554,
        task_id=TASK,
        milestone="2026.10.735",
        random_seed=7358554,
        honest_verdict="complete_"
        + value["verdict_class"]
        + "_v735_historical_learning_qualification",
        preserved_quarantines=[8530, 8533, 8537, 8541, 8545, 8548],
        future_dependencies=[
            dict(experiment_id=n, status="future_dependency_not_input") for n in range(8555, 8565)
        ],
        failed_family_diagnostics=work.get("diagnostics", []),
        node_budget_s=480,
        union_budget_s=2400,
        original_assertion_sources=work.get("assertion_sources", {}),
        upstream_failed_family_diagnostics=work.get("upstream_failed_family_diagnostics", []),
    )
    value["cited_upstream_artifacts"].append(
        dict(
            PIN,
            imported_fields=[
                "collected_node_ids",
                "original_authority_by_suite",
                "failed_family_diagnostics",
            ],
            prior_quarantine_preserved=True,
        )
    )
    value["field_principles"] = {
        k: "Bind this invocation to immutable original authority and measured terminal evidence; grant no science credit."
        for k in [*value, "reproducibility_checksum"]
    }
    value["reproducibility_checksum"] = report.checksum(value)
    return value


def replay(path: Path) -> bool:
    """Rebuild from immutable primitive bytes and reject rehashed commands, timings or summaries."""
    try:
        value = json.loads(path.read_bytes())
        work = old.read(value["primitive_reference"])
        if work["nodes"] and (
            work["nodes"] != original()[1]["nodes"]
            or work["authorities"] != original()[1]["authorities"]
        ):
            return False
        for ref in [*value["source_artifact_hashes"], *value["code_config_hashes"]]:
            if old.reference(Path(ref["path"])) != ref:
                return False
        for row in [*work["receipts"], *work["checks"]]:
            for stream in ("stdout", "stderr"):
                if old.reference(Path(row[stream + "_path"]))["sha256"] != row[stream + "_sha256"]:
                    return False
            stored = json.loads(Path(row["stdout_path"]).with_suffix(".receipt.json").read_bytes())
            if any(stored[k] != row[k] for k in stored):
                return False
            if (
                row.get("junit_reference")
                and old.reference(Path(row["junit_reference"]["path"])) != row["junit_reference"]
            ):
                return False
        return bool(
            value
            == build(
                work,
                Path(value["primitive_reference"]["path"]).parent,
                Path(value["publication_output"]),
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
