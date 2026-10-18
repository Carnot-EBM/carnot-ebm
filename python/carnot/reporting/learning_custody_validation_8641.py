"""REQ-VERIFY-8641: execute disjoint children and cold-recompute independent unions."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import learning_execution_custody_8641 as e
from carnot.reporting import learning_custody_data_8641 as c
from carnot.reporting import v742_contract_methods as methods
from carnot.reporting.count_semantics_runner_8566 import terminal_nodes
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.learning_clock_validation_8592 import E2E
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
OPTIONS = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
SOURCE_AUTHORITY_PIN = "sha256:dde51b32078d5b9cef44045c7dea4e1d0e159e0231a331ecb95ed2e622b48bf2"


def original_sources() -> Json:
    """Use original publication source lists without reopening lost measurement bytes."""
    authority = methods.protocol()["test_inventory_reference"]
    inventory = e.read(authority)
    sources = {p: r["source"] for p, r in inventory["assertion_sources"].items()}
    historical = dict(
        path=str(e.ROOT / "results/experiment_8616_v740_deadline_statement_qualification.json"),
        sha256=SOURCE_AUTHORITY_PIN,
    )
    primary = e.read(historical)
    sources.update(
        {r["path"]: r for r in primary["code_config_hashes"] if "/tests/python/" in r["path"]}
    )
    return dict(
        inventory_reference=authority, historical_source_reference=historical, sources=sources
    )


def partition(nodes: list[str]) -> list[list[str]]:
    """Keep expensive publication and replay nodes alone without changing assertions."""
    heavy = (
        "fixed_stream",
        "real_cli",
        "rehashed",
        "private_cli",
        "real_cases",
        "deadline_edges",
        "owned_worker",
        "retention_comparison",
        "unresumed",
        "controller_plan",
    )
    groups: list[list[str]] = []
    for node in nodes:
        if (
            any(word in node for word in heavy)
            or not groups
            or len(groups[-1]) >= 6
            or any(word in groups[-1][0] for word in heavy)
        ):
            groups.append([node])
        else:
            groups[-1].append(node)
    return groups


def child_budget(started_ns: int, now_ns: int, scope: str) -> float:
    """Keep original execution within3600 seconds and verify adapters in the reserved time."""
    limit = 4500 if scope == "owned" else 3600
    return max(1.0, min(600.0, limit - (now_ns - started_ns) / 1e9))


def attest(work: Json) -> None:
    """Read sealed child bytes and original test authority instead of aggregate claims."""
    if not work["completed_execution"]:
        raise ValueError("unfinished_execution")
    authority = original_sources()
    if authority != work["test_authority"]:
        raise ValueError("original_authority")
    for path in work["original_test_files"]:
        if e.ref(e.ROOT / path) != authority["sources"][str(e.ROOT / path)]:
            raise ValueError("original_test_source")
    collection = e.read(work["collection_reference"])
    receipt = e.read(work["collection_receipt_reference"])
    collected_nodes = [
        line
        for line in Path(receipt["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    if collected_nodes != collection["nodes"] or receipt not in work["receipts"]:
        raise ValueError("collection_receipt")
    if not {"collection", "ruff_check", "ruff_format", "mypy", "spec"} <= {
        r["name"] for r in work["receipts"]
    }:
        raise ValueError("required_validation_receipts")
    for validation in work["receipts"]:
        if (
            not validation["passed"]
            or validation.get("timed_out")
            or validation["exit_code"] != 0
            or any(
                e.ref(Path(validation[k + "_path"]))["sha256"] != validation[k + "_sha256"]
                for k in ("stdout", "stderr")
            )
        ):
            raise ValueError("validation_exit_or_log")
    union = work["node_union"]
    manifests = work["child_shard_manifest"]
    expected = union["collected"]
    executed = [node for manifest in manifests for node in manifest["nodes"]]
    outcomes = [
        row for manifest in manifests for row in terminal_nodes(Path(manifest["junit"]["path"]))
    ]
    if (
        collection["nodes"] != expected
        or collection["authority"] != authority
        or len(executed) != len(set(executed))
        or set(executed) != set(expected)
        or outcomes != union["outcomes"]
        or len(outcomes) != len(expected)
        or {r["node_id"] for r in outcomes} != set(expected)
        or any(r["outcome"] != "passed" for r in outcomes)
    ):
        raise ValueError("original_node_union")
    if len(work["original_nodes"]) != 27 or not set(work["original_nodes"]) <= set(expected):
        raise ValueError("original27")
    for manifest in manifests:
        # The exit-time commitment also binds the separate JUnit reference.
        if not manifest["context"].startswith(manifest["scope"] + "-") or not c.verify(
            {k: v for k, v in manifest.items() if k != "scope"}
        ):
            raise ValueError("child_custody")
    for scope in ("original", "owned"):
        selected = [
            {k: v for k, v in m.items() if k != "scope"} for m in manifests if m["scope"] == scope
        ]
        report = c.aggregate(selected, Path(work["raw"]) / ("cold_" + scope))
        saved = work[scope + "_report"]
        if (
            report["files"] != saved["files"]
            or report["aggregate_reference"]["sha256"] != saved["aggregate_reference"]["sha256"]
            or e.ref(Path(saved["aggregate_reference"]["path"])) != saved["aggregate_reference"]
        ):
            raise ValueError("coverage_union")
    runner = work["original_report"]["files"][str(e.ROOT / e.original.RUNNER)]
    if len(runner["statements"]) != 83 or runner["missing_lines"] or runner["excluded_lines"]:
        raise ValueError("original83_statements")
    for path in e.OWNED:
        row = work["owned_report"]["files"].get(str(e.ROOT / path), {})
        if not row or row["missing_lines"] or row["excluded_lines"]:
            raise ValueError("owned_statement_coverage")
    if (
        not e.original.semantic_cases(work["deadline_case_rows"])
        or not all(
            r["rejected"] for r in work["semantic_mutation_rows"] + work["custody_mutation_rows"]
        )
        or len(work["semantic_mutation_rows"]) != 3
        or len(work["custody_mutation_rows"]) != 7
        or not work["second_combine_invariance"]["passed"]
    ):
        raise ValueError("semantic_or_custody_controls")
    report = dict(files={e.original.RUNNER: runner})
    if work["current_mapping"] != e.original.map_statements(
        e.ref(e.ROOT / e.original.RUNNER), report
    ):
        raise ValueError("source_bound_mapping")


def execute(work: Json) -> None:
    """Collect once, seal each child immediately, then validate within the real reserve."""
    if (time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9 >= 3600 and not work.get(
        "original_execution_complete"
    ):
        raise TimeoutError("measurement_deadline")
    raw = Path(work["raw"])
    work["test_authority"] = original_sources()
    targets = list(
        dict.fromkeys(
            [*e.original.stream.TESTS, *e.original.edge.TESTS, *e.original.TESTS, *E2E, *e.TESTS]
        )
    )
    receipt = child(
        "collection",
        [sys.executable, "-m", "pytest", *OPTIONS, "--collect-only", *targets],
        raw / "logs",
        deadline=90,
    )
    work["receipts"].append(receipt)
    work["collection_receipt_reference"] = e.ref(raw / "logs/collection.receipt.json")
    nodes = [
        line
        for line in Path(receipt["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    work["original_test_files"] = list(
        dict.fromkeys(n.split("::")[0] for n in nodes if n.split("::")[0] not in e.TESTS)
    )
    atomic_json(raw / "collection.json", dict(nodes=nodes, authority=work["test_authority"]))
    work["collection_reference"] = e.ref(raw / "collection.json")
    owned_nodes = [n for n in nodes if n.split("::")[0] in e.TESTS]
    original_nodes = [n for n in nodes if n not in owned_nodes]
    outcomes = []
    source_paths = list(
        dict.fromkeys([*e.OWNED, e.original.RUNNER, *(n.split("::")[0] for n in nodes)])
    )
    sources = [e.ref(e.ROOT / p) for p in source_paths]
    with patch.dict(
        os.environ,
        CARNOT_8582_MEASUREMENT=work["mechanics_fixture"]["path"],
        CARNOT_8603_CASE_ROOT=str(raw / "deadline_cases"),
        CARNOT_8616_CASE_ROOT=str(raw / "deadline_cases"),
    ):
        for scope, selected in (("original", original_nodes), ("owned", owned_nodes)):
            groups = partition(selected)
            for index, group in enumerate(groups):
                progress("8641_" + scope + "_children", index, len(groups) - index)
                root = raw / "children" / scope / str(index)
                root.mkdir(parents=True)
                config = root / "coverage.ini"
                context = scope + "-" + str(index) + "-" + raw.name
                config.write_text(
                    "[run]\nparallel=true\npatch=subprocess,_exit\ncontext="
                    + context
                    + "\ninclude=\n"
                    + "".join("    " + str(e.ROOT / p) + "\n" for p in source_paths)
                    + "[report]\nexclude_lines=\n"
                )
                junit = root / "nodes.xml"
                receipt = c.invoke(
                    scope + str(index),
                    [
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
                    root,
                    root / "coverage",
                    deadline=child_budget(work["started_monotonic_ns"], time.monotonic_ns(), scope),
                )
                work["receipts"].append(receipt)
                manifest = c.seal(root, receipt, group, sources, context)
                manifest.update(scope=scope, junit=e.ref(junit))
                work["child_shard_manifest"].append(manifest)
                outcomes.extend(terminal_nodes(junit))
                atomic_json(raw / "work.json", work)
            if scope == "original":
                work["original_execution_complete"] = True
    work["node_union"] = dict(
        collected=nodes, outcomes=outcomes, original_nodes=work["original_nodes"]
    )
    work["deadline_case_rows"] = [
        json.loads((raw / "deadline_cases" / f"case-{elapsed}" / "case.json").read_bytes())
        for elapsed in (3599, 3600, 3601)
    ]
    rows = work["deadline_case_rows"]
    changed = json.loads(json.dumps(rows))
    changed[-1]["observed_elapsed_s"] = 0
    work["semantic_mutation_rows"] = [
        dict(mutation="omitted_expired", rejected=not e.original.semantic_cases(rows[:1])),
        dict(mutation="changed_diagnostic", rejected=not e.original.semantic_cases(changed)),
        dict(
            mutation="removed_third_read",
            rejected=any(
                "test_actual_removed_read_mutant" in row["node_id"] and row["outcome"] == "passed"
                for row in outcomes
            ),
        ),
    ]
    work["custody_mutation_rows"] = c.controls(raw / "custody_controls")
    hashes_before = [
        c.ref(Path(row["reference"]["path"]))
        for m in work["child_shard_manifest"]
        for row in m["data"]
    ]
    for scope in ("original", "owned"):
        selected = [
            {k: v for k, v in m.items() if k != "scope"}
            for m in work["child_shard_manifest"]
            if m["scope"] == scope
        ]
        work[scope + "_report"] = c.aggregate(selected, raw / (scope + "_aggregate"))
        repeated = c.aggregate(selected, raw / (scope + "_aggregate"))
        work["second_combine_invariance"][scope] = repeated == work[scope + "_report"]
    work["second_combine_invariance"]["raw_unchanged"] = hashes_before == [
        c.ref(Path(r["path"])) for r in hashes_before
    ]
    work["second_combine_invariance"]["passed"] = all(work["second_combine_invariance"].values())
    report = dict(
        files={e.original.RUNNER: work["original_report"]["files"][str(e.ROOT / e.original.RUNNER)]}
    )
    work["current_mapping"] = e.original.map_statements(e.ref(e.ROOT / e.original.RUNNER), report)
    progress("8641_validation_with_900_seconds_reserved")
    commands = [
        ("ruff_check", ["-m", "ruff", "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", ["-m", "ruff", "format", "--check", *e.OWNED, *e.TESTS]),
        ("mypy", ["-m", "mypy", "--strict", "--follow-imports=silent", *e.OWNED]),
        ("spec", ["scripts/check_spec_coverage.py", "--files", *e.TESTS]),
    ]
    for name, argv in commands:
        work["receipts"].append(child(name, [sys.executable, *argv], raw / "logs", deadline=180))
    work["completed_execution"] = True
    atomic_json(raw / "work.json", work)
