"""REQ-VERIFY-8580: execute exact frozen test unions and genuine cold CLI checks."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET
from typing import Any

from carnot.reporting import current_validator_8580 as e
from carnot.reporting.count_semantics_runner_8566 import coverage_complete, exact_union
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
PYTEST = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]


def replay_controls(work: Json, scratch: Path) -> None:
    """Run the standalone file outside the checkout with PYTHONPATH removed."""
    raw = Path(work["raw"])
    scratch.mkdir(parents=True, exist_ok=True)
    corpus = work["authority"]["corpus"]
    for index, case in enumerate(corpus["cases"]):
        print(f"[exp8580] cli_controls completed={index} pending={17 - index}", flush=True)
        path = raw / ("control_" + case["id"] + ".json")
        atomic_json(path, e.old.candidate(corpus, case))
        imported = e.old.av.verify_artifact(path)["flags"]
        receipt = child(
            "cli_" + case["id"],
            [
                "/usr/bin/env",
                "-u",
                "PYTHONPATH",
                "-C",
                str(scratch),
                sys.executable,
                "-u",
                str(e.ROOT / "scripts/adversarial_verify.py"),
                "--json",
                str(path),
            ],
            raw / "logs",
            expected=int(bool(imported)),
            deadline=min(
                60, max(0.01, 3600 - (time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9)
            ),
        )
        work["receipts"].append(receipt)
        cli = (
            json.loads(Path(receipt["stdout_path"]).read_bytes())["reports"][0]["flags"]
            if receipt["passed"]
            else []
        )
        work["cli_rows"].append(
            dict(
                unit_id=case["id"],
                expected=case["expected"],
                candidate=e.reference(path),
                imported_findings=imported,
                cli_findings=cli,
                passed=receipt["passed"] and imported == cli,
                receipt=receipt,
            )
        )
    print("[exp8580] cli_controls completed=17 pending=0", flush=True)


def union_valid(work: Json) -> bool:
    """Require actual passing terminal outcomes for every originally collected node."""
    union = work["node_union"]
    authority = work["authority"]
    frozen = authority["regression_nodes"] + authority["affected_nodes"]
    nodes = union["original_node_ids"]
    terminal = []
    passed = True
    for path in union["junit_paths"]:
        terminal.extend(e.xml_nodes(Path(path)))
        passed = passed and all(len(row) == 0 for row in ET.parse(path).iter("testcase"))
    return bool(
        set(frozen) <= set(nodes)
        and exact_union(nodes, union["child_node_ids"])
        and exact_union(nodes, [terminal])
        and passed
    )


def private_checks(work: Json, scratch: Path, command: Any) -> None:
    """Fresh processes must reject altered summaries even after checksums are repaired."""
    from carnot.reporting.current_validator_runner_8580 import CLI, seal

    for name, field, bad in [
        ("cold", None, None),
        ("missing", "primitive_reference", dict(path=str(scratch / "absent"), sha256="absent")),
        ("owned_failure", "required_checks_passed", True),
        ("rehashed", "rows", []),
    ]:
        seal(work)
        path = scratch / (name + ".json")
        value = e.build(work, path)
        if field:
            value[field] = bad
        value["reproducibility_checksum"] = e.checksum(value)
        atomic_json(path, value)
        command(
            name,
            [sys.executable, str(CLI), "--cold-replay", str(path)],
            expected=0 if name == "cold" else 1,
        )
    seal(work)
    path = scratch / "candidate.json"
    atomic_json(path, e.build(work, path))
    command("adversarial", [sys.executable, "scripts/adversarial_verify.py", "--json", str(path)])
    command(
        "row_lint",
        [sys.executable, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)],
    )
    # The explicit consumer rejects an unqualified revision before full validation.
    expression = (
        "from carnot.reporting.current_validator_8580 import accept_revision; import json; from pathlib import Path; v=json.loads(Path("
        + repr(str(path))
        + ").read_bytes()); accept_revision(v['current_validator_reference'])"
    )
    command("consumer", [sys.executable, "-c", expression], expected=1)


def validate(work: Json, scratch: Path) -> None:
    """Run bounded disjoint children, then combine only newly owned statement coverage."""
    raw, py = Path(work["raw"]), sys.executable
    scratch.mkdir(parents=True, exist_ok=True)

    def command(name: str, argv: list[str], *, expected: int = 0, deadline: int = 300) -> Json:
        receipt = child(
            name,
            argv,
            raw / "logs",
            expected=expected,
            deadline=min(
                deadline,
                max(0.01, 4800 - (time.monotonic_ns() - work["started_monotonic_ns"]) / 1e9),
            ),
        )
        work["receipts"].append(receipt)
        return receipt

    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
    )
    frozen_groups = [work["authority"]["regression_nodes"], work["authority"]["affected_nodes"]]
    collection = command(
        "collection",
        [py, "-m", "pytest", *PYTEST, "--collect-only", *e.TESTS, *sum(frozen_groups, [])],
    )
    nodes = [
        s
        for s in Path(collection["stdout_path"]).read_text().splitlines()
        if s.startswith("tests/") and "::" in s
    ]
    owned = [s for s in nodes if s.split("::")[0] in e.TESTS]
    groups = [owned, *frozen_groups]
    work["node_union"] = dict(original_node_ids=nodes, child_node_ids=groups, junit_paths=[])
    for name, group in zip(("owned", "regression", "affected"), groups, strict=True):
        junit = raw / (name + ".xml")
        command(
            name,
            [
                py,
                "-m",
                "coverage",
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *PYTEST,
                "--junitxml=" + str(junit),
                *group,
            ],
            deadline=1200,
        )
        work["node_union"]["junit_paths"].append(str(junit))
    work["union_passed"] = union_valid(work)
    for name, argv in [
        ("ruff_check", [py, "-m", "ruff", "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", [py, "-m", "ruff", "format", "--check", *e.OWNED, *e.TESTS]),
        ("mypy", [py, "-m", "mypy", "--strict", "--follow-imports=silent", *e.OWNED]),
        ("spec", [py, "scripts/check_spec_coverage.py", "--files", *e.TESTS]),
    ]:
        command(name, argv)
    replay_controls(work, scratch)
    private_checks(work, scratch, command)
    command("combine", [py, "-m", "coverage", "combine", "--rcfile=" + str(config), str(raw)])
    command(
        "coverage",
        [py, "-m", "coverage", "json", "--rcfile=" + str(config), "-o", str(raw / "coverage.json")],
    )
    report = json.loads((raw / "coverage.json").read_bytes())
    work["coverage_complete"] = coverage_complete(report, e.OWNED)
