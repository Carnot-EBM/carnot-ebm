"""REQ-VERIFY-8568: use explicit coverage configuration and disjoint scoped children.

Keeping child shards avoids the old live-parent SQLite failure. Original test
identities and terminal outcomes prevent an empty successful child from passing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import causal_accounting_8568 as e
from carnot.reporting.count_semantics_runner_8566 import (
    coverage_complete,
    exact_union,
    terminal_nodes,
)
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
E2E = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py::test_cli_private_success_failure_and_replay",
    "tests/python/test_restricted_decision_audit_8210.py::test_private_cli_and_rehashed_mutations",
]
REQUIRED = {
    "collection",
    "owned",
    "e2e018",
    "e2e021",
    "ruff_check",
    "ruff_format",
    "mypy",
    "spec",
    "combine",
    "coverage",
}


def configuration(raw: Path) -> Path:
    """Create the configuration directly instead of discovering ambient parent coverage."""
    raw.mkdir(parents=True, exist_ok=True)
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess,_exit\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    return config


def plan(raw: Path) -> list[Json]:
    """Freeze command vectors so collection, tools and retained coverage use one manifest."""
    py = sys.executable
    config = configuration(raw)
    commands = [
        ("ruff_check", [py, "-m", "ruff", "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", [py, "-m", "ruff", "format", "--check", *e.OWNED, *e.TESTS]),
        ("mypy", [py, "-m", "mypy", "--strict", "--follow-imports=silent", *e.OWNED]),
        ("spec", [py, "scripts/check_spec_coverage.py", "--files", *e.TESTS]),
        (
            "combine",
            [py, "-m", "coverage", "combine", "--keep", "--rcfile=" + str(config), str(raw)],
        ),
        (
            "coverage",
            [
                py,
                "-c",
                "import coverage; c=coverage.Coverage(config_file="
                + repr(str(config))
                + "); c.load(); c.json_report(outfile="
                + repr(str(raw / "coverage.json"))
                + ")",
            ],
        ),
    ]
    return [dict(name=name, argv=argv, deadline=180) for name, argv in commands]


def ready(work: Json) -> bool:
    """Require all original outcomes and measured owned statements, never an asserted score alone."""
    return bool(
        work.get("coverage_complete")
        and work.get("coverage_reference")
        and coverage_complete(e.old.read(work["coverage_reference"]), e.OWNED)
        and work.get("union_passed")
        and REQUIRED <= {r["name"] for r in work["receipts"] if r["passed"]}
        and all(r["passed"] for r in work["receipts"])
    )


def validate(work: Json) -> None:
    """Collect IDs before bounded serial children and require their exact terminal union."""
    raw = Path(work["raw"])
    config = configuration(raw)
    py = sys.executable
    options = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    collection = child(
        "collection",
        [py, "-m", "pytest", *options, "--collect-only", *e.TESTS, *E2E],
        raw / "logs",
        deadline=90,
    )
    work["receipts"].append(collection)
    nodes = [
        line
        for line in Path(collection["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    groups = [
        [n for n in nodes if n.split("::")[0] in e.TESTS],
        *[[n for n in nodes if n == spec] for spec in E2E],
    ]
    outcomes = []
    for name, group in zip(("owned", "e2e018", "e2e021"), groups, strict=True):
        junit = raw / (name + ".xml")
        with patch.dict(os.environ, CARNOT_8568_COVERAGE_DIR=str(raw)):
            receipt = child(
                name,
                [
                    py,
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
        original_node_ids=nodes, child_node_ids=groups, terminal_node_outcomes=outcomes
    )
    work["union_passed"] = (
        bool(nodes)
        and exact_union(nodes, groups)
        and exact_union(nodes, [[r["node_id"] for r in outcomes]])
        and all(r["outcome"] == "passed" for r in outcomes)
    )
    for spec in plan(raw):
        work["receipts"].append(
            child(spec["name"], spec["argv"], raw / "logs", deadline=spec["deadline"])
        )
    work["coverage_complete"] = coverage_complete(
        json.loads((raw / "coverage.json").read_bytes()), e.OWNED
    )
    work["coverage_reference"] = e.old.reference(raw / "coverage.json")
    work["owned_ready"] = ready(work)
