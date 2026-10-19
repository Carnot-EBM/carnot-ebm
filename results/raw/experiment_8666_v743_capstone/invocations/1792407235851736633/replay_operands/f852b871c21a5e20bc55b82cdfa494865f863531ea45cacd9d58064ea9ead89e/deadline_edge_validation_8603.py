"""REQ-VERIFY-8603: disjoint bounded children retain exact clock statement evidence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import deadline_edge_8603 as e
from carnot.reporting import learning_clock_validation_8592 as prior
from carnot.reporting.count_semantics_runner_8566 import terminal_nodes
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
clock = time.monotonic
RUNNER = prior.RUNNER


def execute(work: Json) -> None:
    """Collect original identities before independent children and keep genuine CLI shards."""
    raw = Path(work["raw"])
    if clock() > work["started_monotonic_ns"] / 1e9 + 3600:
        raise TimeoutError("measurement deadline")
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess,_exit\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in [*e.OWNED, RUNNER])
        + "[report]\nexclude_lines=\n"
    )
    options = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    collection = child(
        "collection",
        [
            sys.executable,
            "-m",
            "pytest",
            *options,
            "--collect-only",
            *e.stream.TESTS,
            *e.TESTS,
            *prior.E2E,
        ],
        raw / "logs",
        deadline=90,
    )
    work["receipts"].append(collection)
    nodes = [
        line
        for line in Path(collection["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    atomic_json(
        raw / "original_collection.json",
        dict(
            nodes=nodes, original_nodes=work["original_nodes"], authority=work["original_authority"]
        ),
    )
    original = [n for n in nodes if n in work["original_nodes"]]
    heavy = (
        "fixed_stream",
        "real_cli",
        "rehashed_primitive",
        "measurement_adapter",
        "production_dispatch",
        "owned_worker",
        "owned_failures",
    )
    groups = [[n for n in original if not any(word in n for word in heavy)]]
    groups += [[n] for n in original if any(word in n for word in heavy)]
    original_group_count = len(groups)
    groups += [[n] for n in nodes if n not in work["original_nodes"]]
    outcomes = []
    with patch.dict(
        os.environ,
        CARNOT_8582_MEASUREMENT=work["mechanics_fixture"]["path"],
        CARNOT_8603_CASE_ROOT=str(raw / "deadline_cases"),
    ):
        for index, group in enumerate(groups):
            progress("owned_nodes", index, len(groups) - index)
            junit = raw / f"nodes{index}.xml"
            receipt = child(
                f"nodes{index}",
                [
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "--rcfile=" + str(config),
                    "-m",
                    "pytest",
                    *options,
                    "--junitxml=" + str(junit),
                    *group,
                ],
                raw / "logs",
                deadline=600,
            )
            work["receipts"].append(receipt)
            if junit.is_file():
                outcomes.extend(terminal_nodes(junit))
    work["node_union"] = dict(
        collected=nodes,
        original_children=groups[:original_group_count],
        children=groups,
        outcomes=outcomes,
        original_authority=work["original_authority"],
        expected_original_count=27,
    )
    for index, elapsed in enumerate((3599.0, 3600.0, 3601.0)):
        progress("deadline_edge_measurement", index, 3 - index)
        work["deadline_case_rows"].append(
            json.loads((raw / "deadline_cases" / f"case-{elapsed:g}" / "case.json").read_bytes())
        )
    commands = [
        ("ruff_check", ["-m", "ruff", "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", ["-m", "ruff", "format", "--check", *e.OWNED, *e.TESTS]),
        ("mypy", ["-m", "mypy", "--strict", "--follow-imports=silent", *e.OWNED]),
        ("spec", ["scripts/check_spec_coverage.py", "--files", *e.TESTS]),
        ("combine", ["-m", "coverage", "combine", "--keep", "--rcfile=" + str(config), str(raw)]),
        (
            "coverage",
            [
                "-m",
                "coverage",
                "json",
                "--rcfile=" + str(config),
                "-o",
                str(raw / "coverage_full.json"),
            ],
        ),
    ]
    for name, argv in commands:
        work["receipts"].append(child(name, [sys.executable, *argv], raw / "logs", deadline=180))
    full = json.loads((raw / "coverage_full.json").read_bytes())
    owned = {p: full["files"][p] for p in e.OWNED if p in full["files"]}
    source = (e.ROOT / RUNNER).read_text()
    anchors = prior.owned_clock_statements(source)
    diagnostic = next(
        index
        for index, line in enumerate(source.splitlines(), 1)
        if 'raw, "measurement_deadline"' in line
    )
    anchors = sorted(set([*anchors, diagnostic]))
    executed = set(full["files"][RUNNER]["executed_lines"])
    missing = sorted(set(anchors) - executed)
    owned[RUNNER] = dict(
        executed_lines=sorted(set(anchors) & executed),
        missing_lines=missing,
        summary=dict(
            num_statements=len(anchors),
            missing_lines=len(missing),
            covered_lines=len(anchors) - len(missing),
        ),
    )
    atomic_json(
        raw / "coverage_owned.json",
        dict(
            files=owned,
            ownership="new modules and original four clock statements plus explicit diagnostic read",
        ),
    )
    work["coverage_reference"] = e.old.ref(raw / "coverage_owned.json")
    work["measured_statement_anchors"] = dict(
        path=RUNNER,
        original_statement_lines=[30, 74, 91, 94],
        current_statement_lines=anchors,
        diagnostic_line=diagnostic,
        executed_lines=owned[RUNNER]["executed_lines"],
        missing_lines=missing,
    )
    work["private_cli_receipts"] = [r for r in work["receipts"] if r["name"].startswith("nodes")]
    atomic_json(raw / "work.json", work)
