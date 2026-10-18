"""REQ-VERIFY-8593: preserve original node identities in disjoint bounded children."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

from carnot.reporting import resident_provenance_8593 as e
from carnot.reporting.current_validator_8580 import xml_nodes
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]
SUITES = [
    e.TEST,
    "tests/python/test_resident_complete_cost_8584.py",
    "tests/python/test_resident_inference_8571.py",
    "tests/python/test_resident_qualification_8572.py",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def validate(work: Json, raw: Path) -> list[Json]:
    """Collect before execution; fresh children retain sealed fixtures and original dates."""
    scratch = Path("/var/tmp") / ("exp8593-validation-" + raw.name)
    scratch.mkdir(parents=True, exist_ok=True)
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(scratch / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    receipts, shards, original = [], [], []
    for i, suite in enumerate(SUITES):
        receipt = child(
            f"collect_{i}",
            [sys.executable, "-m", "pytest", *COMMON, "--collect-only", "-q", suite],
            raw / "validation",
            deadline=60,
        )
        receipts.append(receipt)
        nodes = [
            line.strip()
            for line in Path(receipt["stdout_path"]).read_text().splitlines()
            if line.startswith("tests/") and "::" in line
        ]
        original.extend(nodes)
        shards.append(nodes)
    atomic_json(raw / "collection.json", dict(nodes=original, shards=shards, suites=SUITES))
    outcomes: list[str] = []
    for i, nodes in enumerate(shards):
        progress("8593_validation_shards", i, len(shards) - i)
        junit = scratch / f"shard-{i}.xml"
        receipt = child(
            f"shard_{i}",
            [
                "env",
                "COVERAGE_RCFILE=" + str(config),
                sys.executable,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *COMMON,
                "-vv",
                *nodes,
                "--junitxml=" + str(junit),
                "--basetemp=" + str(scratch / f"suite-{i}"),
            ],
            raw / "validation",
            deadline=480,
        )
        receipts.append(receipt)
        if junit.is_file() and all(len(row) == 0 for row in ET.parse(junit).iter("testcase")):
            outcomes.extend(xml_nodes(junit))
    historical = work.get("original_node_union", {}).get("collected", [])
    union = dict(
        passed=bool(original)
        and len(original) == len(set(original))
        and sorted(original) == sorted(outcomes)
        and set(historical) <= set(original),
        collected=original,
        passed_node_ids=outcomes,
        shards=shards,
        original_authority_by_suite={
            s: "unchanged sealed suite fixtures and original CLI dates" for s in SUITES
        },
        historical_union_preserved=set(historical) <= set(original),
    )
    atomic_json(raw / "node_union.json", union)
    work.update(
        exact_union_receipt=union, coverage_config=str(config), coverage_scratch=str(scratch)
    )
    receipts.append(dict(name="exact_disjoint_node_union", passed=union["passed"]))
    return receipts


def finish(work: Json, raw: Path) -> list[Json]:
    """Only newly owned statements enter the coverage gate; genuine children count."""
    config = work["coverage_config"]
    report = Path(work["coverage_scratch"]) / "coverage.json"
    commands = [
        (
            "coverage_combine",
            [sys.executable, "-m", "coverage", "combine", "--rcfile=" + config, "--keep"],
        ),
        (
            "coverage_report",
            [
                sys.executable,
                "-m",
                "coverage",
                "report",
                "--rcfile=" + config,
                "--show-missing",
                "--fail-under=100",
            ],
        ),
        (
            "coverage_json",
            [sys.executable, "-m", "coverage", "json", "--rcfile=" + config, "-o", str(report)],
        ),
        ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, e.TEST]),
        ("ruff_format", [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, e.TEST]),
        (
            "strict_mypy",
            [
                str(e.ROOT / ".venv/bin/mypy"),
                "--config-file=/dev/null",
                "--strict",
                "--follow-imports=skip",
                "--ignore-missing-imports",
                *e.OWNED,
            ],
        ),
        ("spec_coverage", [sys.executable, "scripts/check_spec_coverage.py", "--files", e.TEST]),
    ]
    receipts = [child(name, argv, raw / "validation", deadline=90) for name, argv in commands]
    coverage = json.loads(report.read_bytes())
    work["python_coverage"] = coverage
    atomic_json(raw / "coverage.json", coverage)
    receipts.append(
        dict(name="owned_python_coverage_100", passed=coverage["totals"]["percent_covered"] == 100)
    )
    return receipts
