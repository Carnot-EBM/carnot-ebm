"""Continue the exact receipt frontier without games. Spec: REQ-REPORT-7975.

This configuration reuses the qualified reader and validation supervisor.
An empty valid inventory describes available evidence, not agent improvement.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import yaml

from carnot.reporting import arc_supervisor_v690_delta as runner
from carnot.reporting.current_work_receipt import sha256_file

ROOT = runner.ROOT
PRIOR = ROOT / "results/experiment_7962_v690_arc_supervisor_delta.json"
INVENTORY = ROOT / "results/raw/experiment_7962_v690_arc_supervisor_delta/receipt_inventory.json"
REGISTRY = runner.REGISTRY
OUTPUT = ROOT / "results/experiment_7975_v691_arc_supervisor_delta.json"
PINNED = {
    str(PRIOR): "sha256:4999f7c1398c5998d8194aa6315767654e77f02e0aba138d4e826e07e3b943f6",
    str(INVENTORY): "sha256:9eed81ea6f70d410198013c58e26d3cc1e30cdc5072e550d3ba86e41264d46c1",
    str(REGISTRY): "sha256:071ecd51939117d9bc5b48491b0649e2ae126e3f848b5af8ff7e53c88e890947",
}
EXPERIMENT_ID = 7975
PRIOR_ID = 7962
MILESTONE = "2026.10.691"
RUN_DATE = "20261001"
MODULE = "python/carnot/reporting/arc_supervisor_v691_delta.py"
CLI = "scripts/experiments/experiment_7975_v691_arc_supervisor_delta.py"
TEST = "tests/python/test_arc_supervisor_delta_7975.py"
ADDED = [MODULE, CLI, runner.MODULE, runner.CLI]
INCLUDE = ",".join("*/" + path for path in ADDED)
CONSUMERS = [runner.TEST, *runner.CONSUMERS]
COVERAGE_TESTS = [runner.TEST]
previous = runner.previous
validation = runner.validation
replay = runner.replay


def inputs() -> dict[str, Any]:
    """Bind frontier bytes to their producer and reject retired prerequisites."""
    checked = runner.inputs(sys.modules[__name__])
    for field, expected in (
        ("experiment_id", PRIOR_ID),
        ("task_id", "exp7962-arc-supervisor-delta"),
        ("run_date", "20261001"),
        ("honest_verdict", "complete_null_no_new_supervisor_outcomes"),
    ):
        checked["checks"].append(
            previous.operand(
                PRIOR, field, expected, checked["prior"].get(field), PINNED[str(PRIOR)]
            )
        )
    exclusion = ROOT / "ops/exclusion_manifest.yaml"
    document = yaml.safe_load(exclusion.read_text()) if exclusion.is_file() else {}
    retired = any(
        row.get("experiment_id") in {PRIOR_ID, EXPERIMENT_ID}
        or any(
            str(identity).split("-")[0] in {"exp7962", "exp7975"}
            for identity in row.get("experiment_ids", [])
        )
        for entries in document.values()
        if isinstance(entries, list)
        for row in entries
        if isinstance(row, dict)
    )
    checked["checks"].append(
        previous.operand(
            exclusion,
            "retired",
            False,
            retired if exclusion.is_file() else "missing",
            sha256_file(exclusion) if exclusion.is_file() else None,
        )
    )
    checked["failures"] = [r for r in checked["checks"] if r["expected"] != r["observed"]]
    history = ROOT / "results/raw" / OUTPUT.stem / "prior_owned_attempt.json"
    if history.is_file():
        record = json.loads(history.read_text())
        checked["prior"].setdefault("historical_required_failures", []).append(
            dict(
                upstream_id="exp7975-prior-owned-attempt",
                path=str(history),
                sha256=sha256_file(history),
                **record,
            )
        )
        checked["owned_repository_health"] = record["repository_health"]
        checked["additional_source_hashes"] = {
            str(history): sha256_file(history),
            **record["frozen_snapshots"],
        }
    return checked


def commands(private: Path) -> list[dict[str, Any]]:
    """Use the existing checks with current identity and identical coverage scope."""
    specs = runner.commands(private, sys.modules[__name__])
    success = next(row for row in specs if row["name"] == "cli_success")
    legacy = dict(success, name="legacy_cli_success", argv=list(success["argv"]))
    legacy["argv"] = [
        runner.CLI if arg == CLI else arg.replace("coverage.success", "coverage.legacy")
        for arg in legacy["argv"]
    ]
    legacy["argv"][-1] = str(private / "legacy" / runner.OUTPUT.name)
    combine = next(row for row in specs if row["name"] == "coverage_combine")
    combine["argv"].append(str(private / "coverage.legacy"))
    specs.insert(specs.index(combine), legacy)
    history = ROOT / "results/raw" / OUTPUT.stem / "prior_owned_attempt.json"
    if history.is_file():
        record = json.loads(history.read_text())
        health = next(row for row in specs if row["name"] == "full_python_suite")
        specs = [row for row in specs if row["name"] != "full_python_suite"]
        specs.extend(
            dict(
                health,
                name=f"reused_{receipt['name']}_{index}",
                argv=receipt.get("command_argv", receipt.get("argv", health["argv"])),
                reuse_receipt=receipt,
                reuse_source_path=str(history),
                reuse_source_sha256=sha256_file(history),
            )
            for index, receipt in enumerate(record["repository_health"])
        )
    return specs


def execute(output: Path, private: Path) -> int:
    """Keep publication checks identical while advancing only the frozen frontier."""
    return runner.execute(output, private, sys.modules[__name__])
