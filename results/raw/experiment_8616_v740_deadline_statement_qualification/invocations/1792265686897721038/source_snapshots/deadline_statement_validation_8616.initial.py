"""REQ-VERIFY-8616: retain exact node unions and independently check raw child shards."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from coverage import Coverage, CoverageData
from coverage.parser import PythonParser

from carnot.reporting import deadline_statement_8616 as e
from carnot.reporting.count_semantics_runner_8566 import terminal_nodes
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.learning_clock_validation_8592 import E2E
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
clock = time.monotonic
OPTIONS = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]


def attest(work: Json) -> None:
    """Recompute coverage from real shards so repaired report hashes cannot add lines."""
    full = e.read(work["full_coverage_reference"])
    union = work["node_union"]
    collected = json.loads(Path(work["raw"], "original_collection.json").read_bytes())
    if (
        collected["nodes"] != union["collected"]
        or collected["authority"] != work["original_authority"]
        or len(work["original_nodes"]) != 27
        or not e.stream.validation.exact_union(work["original_nodes"], union["original_children"])
        or not e.stream.validation.exact_union(union["collected"], union["children"])
        or not e.stream.validation.exact_union(
            union["collected"], [[r["node_id"] for r in union["outcomes"]]]
        )
        or any(r["outcome"] != "passed" for r in union["outcomes"])
    ):
        raise ValueError("original_assertion_union")
    expected_old = e.read(work["historical_work_reference"])["node_union"]["collected"]
    measured_outcomes = []
    for reference in work["junit_references"]:
        if e.ref(Path(reference["path"])) != reference:
            raise ValueError("junit_hash")
        measured_outcomes.extend(terminal_nodes(Path(reference["path"])))
    if (
        not set(expected_old) <= set(union["collected"])
        or measured_outcomes != union["outcomes"]
        or any(r.get("exit_code", 0) != 0 or r.get("timed_out") for r in work["receipts"])
    ):
        raise ValueError("actual_child_assertions")
    executed: dict[str, set[int]] = {p: set() for p in [*e.OWNED, e.RUNNER]}
    if not work["shards"]:
        raise ValueError("missing_shards")
    for reference in work["shards"]:
        if e.ref(Path(reference["path"])) != reference:
            raise ValueError("shard_hash")
        data = CoverageData(basename=reference["path"])
        data.read()
        for path in executed:
            executed[path].update(data.lines(str(e.ROOT / path)) or [])
    for path, lines in executed.items():
        parser = PythonParser(filename=str(e.ROOT / path))
        parser.parse_source()
        measured = full["files"][path]
        if (
            set(parser.statements) - lines
            or measured["missing_lines"]
            or measured["excluded_lines"]
            or set(measured["executed_lines"]) != set(parser.translate_lines(lines))
        ):
            raise ValueError("owned_statement_coverage")
    if work["current_mapping"] != e.map_statements(e.ref(e.ROOT / e.RUNNER), full):
        raise ValueError("source_mapping")
    mutations = work["semantic_mutation_rows"]
    if not e.semantic_cases(work["deadline_case_rows"]) or {
        r["mutation"] for r in mutations if r["rejected"]
    } != {
        "omitted_expired",
        "changed_diagnostic",
        "removed_third_read",
    }:
        raise ValueError("semantic_mutations")


def execute(work: Json) -> None:
    """Measure once, collect once, then run bounded children without historical controllers."""
    raw = Path(work["raw"])
    if clock() > work["started_monotonic_ns"] / 1e9 + 3600:
        raise TimeoutError("measurement deadline")
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess,_exit\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in [*e.OWNED, e.RUNNER])
        + "[report]\nexclude_lines=\n"
    )
    meter = Coverage(config_file=str(config), data_suffix="parent")
    meter.start()
    targets = list(dict.fromkeys([*e.stream.TESTS, *e.edge.TESTS, *e.TESTS, *E2E]))
    receipt = child(
        "collection",
        [sys.executable, "-m", "pytest", *OPTIONS, "--collect-only", *targets],
        raw / "logs",
        deadline=90,
    )
    work["receipts"].append(receipt)
    nodes = [
        line
        for line in Path(receipt["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    atomic_json(
        raw / "original_collection.json",
        dict(
            nodes=nodes, original_nodes=work["original_nodes"], authority=work["original_authority"]
        ),
    )
    originals = [n for n in nodes if n in work["original_nodes"]]
    # Source qualification runs before the expensive historical union.
    focused = [n for n in nodes if n.startswith(e.TESTS[0]) and "historical_mapping" in n]
    heavy = (
        "fixed_stream",
        "real_cli",
        "rehashed_primitive",
        "measurement_adapter",
        "production_dispatch",
        "owned_worker",
        "owned_failures",
        "retention_comparison",
        "resumed_history",
    )
    fast = [n for n in originals if not any(word in n for word in heavy)]
    original_groups = [fast] + [[n] for n in originals if n not in fast]
    groups = [focused, *original_groups]
    rest = [n for n in nodes if n not in originals and n not in focused]
    groups += [[n] for n in rest]

    outcomes = []
    work["junit_references"] = []
    with patch.dict(
        os.environ,
        CARNOT_8582_MEASUREMENT=work["mechanics_fixture"]["path"],
        CARNOT_8603_CASE_ROOT=str(raw / "deadline_cases"),
        CARNOT_8616_CASE_ROOT=str(raw / "deadline_cases"),
    ):
        for index, group in enumerate(groups):
            progress("source_qualified_disjoint_nodes", index, len(groups) - index)
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
                    *OPTIONS,
                    "--junitxml=" + str(junit),
                    *group,
                ],
                raw / "logs",
                deadline=min(600, max(1, 3600 - (clock() - work["started_monotonic_ns"] / 1e9))),
            )
            work["receipts"].append(receipt)
            if junit.is_file():
                outcomes.extend(terminal_nodes(junit))
                work["junit_references"].append(e.ref(junit))
    work["node_union"] = dict(
        collected=nodes,
        children=groups,
        original_children=original_groups,
        outcomes=outcomes,
        original_authority=work["original_authority"],
    )
    work["deadline_case_rows"] = [
        json.loads((raw / "deadline_cases" / f"case-{v}" / "case.json").read_bytes())
        for v in (3599, 3600, 3601)
    ]
    rows = work["deadline_case_rows"]
    changed = json.loads(json.dumps(rows))
    changed[-1]["observed_elapsed_s"] = 0
    work["semantic_mutation_rows"] = [
        dict(mutation="omitted_expired", rejected=not e.semantic_cases(rows[:1])),
        dict(mutation="changed_diagnostic", rejected=not e.semantic_cases(changed)),
        dict(
            mutation="removed_third_read",
            rejected=any(
                "test_actual_removed_read_mutant" in r["node_id"] and r["outcome"] == "passed"
                for r in outcomes
            ),
        ),
    ]
    meter.stop()
    meter.save()
    progress("validation_with_900s_reserved")
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
    work["shards"] = [e.ref(p) for p in sorted(raw.glob(".coverage.*"))]
    work["full_coverage_reference"] = e.ref(raw / "coverage_full.json")
    full = e.read(work["full_coverage_reference"])
    work["current_mapping"] = e.map_statements(e.ref(e.ROOT / e.RUNNER), full)
    work["coverage_reference"] = work["full_coverage_reference"]
    atomic_json(raw / "work.json", work)
