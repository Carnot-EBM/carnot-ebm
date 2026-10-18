"""REQ-VERIFY-8482: exact bounded original test shards qualify the independent rejection."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from carnot.reporting import arc_panel_qualification_8482 as p
from carnot.reporting import v717_contract_runner as commands
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.v709_execution import child
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
HISTORICAL = "tests/python/test_arc_generalization_panel_8468.py"
SUITES = [
    p.TEST,
    HISTORICAL,
    "tests/python/test_arc_decision_telemetry.py",
    "tests/python/test_arc_supervisor_delta_7874.py",
    "tests/python/test_arc_submitted_agent_parity.py",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def collect(private: Path, suite: str) -> tuple[list[str], Json]:
    """Freeze actual node IDs before execution so a slow test cannot disappear from a shard."""
    receipt = child(
        "collect_" + Path(suite).stem,
        [
            str(p.ROOT / ".venv/bin/pytest"),
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
    if not receipt["passed"]:
        raise ValueError("test_collection_failed")
    nodes = [
        line.strip()
        for line in Path(receipt["stdout_path"]).read_text().splitlines()
        if line.startswith(suite + "::")
    ]
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("empty_or_duplicate_collection")
    return nodes, receipt


def manifest(private: Path) -> list[Json]:
    """Coverage includes only new code and the original232 statements; all assertions remain."""
    private.mkdir(parents=True, exist_ok=True)
    owned = [
        *p.OWNED,
        "python/carnot/reporting/arc_generalization_panel_8468.py",
        "scripts/experiments/experiment_8468_v728_arc_generalization_panel.py",
    ]
    with patch.object(commands, "m", SimpleNamespace(ROOT=p.ROOT, OWNED=owned, TEST=p.TEST)):
        specs = list(commands.manifest(private))
    specs = [s for s in specs if s["name"] != "private_E2E018_consumers"]
    config = private / "coverage.ini"
    rc = "--rcfile=" + str(config)
    next(s for s in specs if s["name"] == "owned_tests")["deadline"] = 900
    next(s for s in specs if s["name"] == "coverage_combine")["argv"].append("--keep")
    for name in ["ruff_check", "ruff_format", "strict_mypy"]:
        spec = next(s for s in specs if s["name"] == name)
        spec["argv"] = [a for a in spec["argv"] if a not in owned[len(p.OWNED) :]]
    shards, collections, proof = [], [], {}
    for suite in SUITES:
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
        if suite == p.TEST:
            continue
        size = 1 if suite in [HISTORICAL, SUITES[-1]] else 8
        groups = [nodes[i : i + size] for i in range(0, len(nodes), size)]
        assert [n for group in groups for n in group] == nodes
        proof[suite] = dict(original=nodes, shards=groups, exact_union=True)
        for i, group in enumerate(groups):
            name = (
                f"historical_8468_{i}" if suite == HISTORICAL else f"private_{Path(suite).stem}_{i}"
            )
            if suite == HISTORICAL:
                request = private / f"historical-{i}.json"
                atomic_json(request, group)
                argv = [
                    str(p.ROOT / ".venv/bin/coverage"),
                    "run",
                    rc,
                    str(p.ROOT / p.CLI),
                    "--historical-shard",
                    str(request),
                ]
            else:
                argv = [
                    str(p.ROOT / ".venv/bin/pytest"),
                    "-n",
                    "0",
                    "-o",
                    "addopts=",
                    "--no-cov",
                    "-q",
                    *group,
                    "--basetemp=" + str(private / name),
                ]
            shards.append(
                dict(
                    name=name,
                    argv=argv,
                    deadline=240,
                    expected=0,
                    scope="owned",
                    collected_node_ids=group,
                )
            )
    atomic_json(private / "coverage_shard_union.json", proof)
    return [*collections, specs[0], *shards, *specs[1:]]


def historical(request: Path) -> int:
    """Read original V728 snapshots under their sealed authority without writing historical files."""
    import pytest

    if sha256_file(p.ROOT / p.PRIOR) != p.PRIOR_PIN:
        raise ValueError("historical_primary_pin")
    primary = json.loads((p.ROOT / p.PRIOR).read_bytes())
    ref = primary["work_reference"]
    if sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("historical_work_pin")
    refs = json.loads(Path(ref["path"]).read_bytes())["source_input_receipts"]
    if any(sha256_file(Path(r["snapshot_path"])) != r["sha256"] for r in refs):
        raise ValueError("historical_snapshot_pin")
    nodes = json.loads(request.read_bytes())
    with TemporaryDirectory(prefix="exp8482-history-", dir="/var/tmp") as directory:
        with frozen_inputs(refs, Path(directory) / "writes"):
            return int(
                pytest.main(
                    [
                        "-n",
                        "0",
                        "-o",
                        "addopts=",
                        "--no-cov",
                        "-q",
                        *nodes,
                        "--basetemp=" + str(Path(directory) / "pytest"),
                    ]
                )
            )
