"""REQ-VERIFY-8559: collect once, run a disjoint union, and cover only owned code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting import complete_native_cost_8559 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.historical_consumer_roots_8402 import parse_outcomes
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]
SUITES = [
    e.TEST,
    "tests/python/test_nine_input_8533.py",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def validate(work: Json, raw: Path) -> list[Json]:
    """The exact original node union is frozen before any pytest child executes."""
    scratch = Path("/var/tmp") / ("exp8559-validation-" + raw.name)
    scratch.mkdir(parents=True, exist_ok=True)
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(scratch / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    py = str(e.ROOT / ".venv/bin/python")
    receipts, shards, nodes, outcomes = [], [], [], []
    for i, suite in enumerate(SUITES):
        receipt = child(
            f"collect_{i}",
            [py, "-m", "pytest", *COMMON, "--collect-only", "-q", suite],
            raw / "validation",
            deadline=90,
        )
        receipts.append(receipt)
        found = [
            s.strip()
            for s in Path(receipt["stdout_path"]).read_text().splitlines()
            if s.startswith("tests/") and "::" in s
        ]
        shards.append(found)
        nodes.extend(found)
    atomic_json(raw / "collection.json", dict(node_ids=nodes, suites=SUITES, shards=shards))
    authorities = {
        suite: e.qualified.original_authority()
        if "8533" in suite
        else "suite-owned original sealed private fixtures and original CLI dates"
        for suite in SUITES
    }
    for i, shard in enumerate(shards):
        parent = scratch / f"suite-{i}"
        parent.mkdir(parents=True, exist_ok=True)
        receipt = child(
            f"shard_{i}",
            [
                "env",
                "COVERAGE_RCFILE=" + str(config),
                "CARNOT_8533_EXTENSION=" + work["bundle"]["extension"]["path"],
                "LLVM_PROFILE_FILE=" + str(scratch / "rust-%p-%m.profraw"),
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                str(e.ROOT / e.CLI),
                "--sealed-tests",
                SUITES[i],
                *COMMON,
                "-vv",
                *shard,
                "--basetemp=" + str(parent / "pytest"),
            ],
            raw / "validation",
            deadline=480,
        )
        receipts.append(receipt)
        outcomes.extend(parse_outcomes(Path(receipt["stdout_path"]))["passed"])
    union = dict(
        passed=bool(nodes)
        and len(nodes) == len(set(nodes))
        and len(outcomes) == len(nodes)
        and set(nodes) == set(outcomes),
        collected=nodes,
        passed_node_ids=outcomes,
        shards=shards,
        original_authority_by_suite=authorities,
    )
    work.update(
        exact_union_receipt=union, coverage_config=str(config), coverage_scratch=str(scratch)
    )
    atomic_json(raw / "node_union.json", union)
    receipts.append(dict(name="exact_disjoint_node_union", passed=union["passed"]))
    return receipts


def finish(work: Json, raw: Path) -> list[Json]:
    """Coverage includes real CLI subprocesses; old modules do not enter its denominator."""
    py, config = str(e.ROOT / ".venv/bin/python"), work["coverage_config"]
    report = Path(work["coverage_scratch"]) / "coverage.json"
    commands = [
        ("coverage_combine", [py, "-m", "coverage", "combine", "--rcfile=" + config, "--keep"]),
        (
            "coverage_report",
            [
                py,
                "-m",
                "coverage",
                "report",
                "--rcfile=" + config,
                "--show-missing",
                "--fail-under=100",
            ],
        ),
        ("coverage_json", [py, "-m", "coverage", "json", "--rcfile=" + config, "-o", str(report)]),
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
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", "--files", e.TEST]),
    ]
    receipts = [child(name, argv, raw / "validation", deadline=90) for name, argv in commands]
    coverage = json.loads(report.read_bytes())
    work["python_coverage"] = coverage
    atomic_json(raw / "coverage.json", coverage)
    receipts.append(
        dict(name="owned_python_coverage_100", passed=coverage["totals"]["percent_covered"] == 100)
    )
    return receipts
