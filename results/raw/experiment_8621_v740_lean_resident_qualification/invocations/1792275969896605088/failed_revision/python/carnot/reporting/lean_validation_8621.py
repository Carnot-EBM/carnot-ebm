"""REQ-VERIFY-8621: collect exact original nodes before bounded disjoint children.

Only the historical provenance suite receives its sealed V738 checkout. Current
tests use unchanged checkout bytes and actual subprocess coverage, not old clocks.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from carnot.reporting import lean_qualification_8621 as e
from carnot.reporting import lean_qualification_8609 as previous
from carnot.reporting.lean_original_8609 import checkout
from carnot.reporting.current_validator_8580 import xml_nodes
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]


def validate(work: Json, raw: Path) -> list[Json]:
    """Keep command, log and node custody without invoking historical controllers."""
    scratch = Path(tempfile.mkdtemp(prefix="carnot8621-validation-", dir="/var/tmp"))
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(scratch / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    historical, _ = previous.upstream(e.ROOT, previous.NAME)
    original = historical["original_node_union"]["collected"]
    suites = list(dict.fromkeys([n.split("::", 1)[0] for n in original] + e.TESTS))
    collection = child(
        "collection",
        [sys.executable, "-m", "pytest", *COMMON, "--collect-only", "-q", *suites],
        raw / "checks",
        deadline=120,
    )
    nodes = [
        line.strip()
        for line in Path(collection["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    authority = checkout(scratch / "original_v738")
    atomic_json(raw / "historical_authority.json", authority)
    shards = [[n for n in nodes if n.startswith(s + "::")] for s in suites]
    shards = [s for s in shards if s]
    atomic_json(
        raw / "collected_nodes.json", dict(nodes=nodes, shards=shards, original122=original)
    )
    receipts, outcomes, retained = [collection], [], []
    for i, selected in enumerate(shards):
        progress("8621_disjoint_tests", i, len(shards) - i)
        xml = raw / f"nodes-{i}.xml"
        prefix = ["env", "COVERAGE_RCFILE=" + str(config)]
        command = [sys.executable, "-m", "coverage"]
        if selected[0].startswith("tests/python/test_resident_provenance_8593.py::"):
            root = authority["root"]
            prefix += ["PYTHONPATH=" + root + "/python:" + root, "CARNOT_REPO_ROOT=" + root]
            command = [
                sys.executable,
                "-c",
                f"import os,sys;os.chdir({root!r});from coverage.cmdline import main;raise SystemExit(main(sys.argv[1:]))",
            ]
        receipt = child(
            f"tests_{i}",
            [
                *prefix,
                *command,
                "run",
                "--rcfile=" + str(config),
                "-m",
                "pytest",
                *COMMON,
                "-q",
                *selected,
                "--junitxml=" + str(xml),
                "--basetemp=" + str(scratch / f"suite-{i}"),
            ],
            raw / "checks",
            deadline=480,
        )
        receipts.append(receipt)
        if receipt["passed"]:
            outcomes.extend(xml_nodes(xml))
            retained.append(previous.reference(xml))
    passed = (
        bool(nodes)
        and len(nodes) == len(set(nodes))
        and sorted(nodes) == sorted(outcomes)
        and set(original) <= set(nodes)
        and len(original) == 122
    )
    union = dict(
        name="exact_disjoint_node_union",
        passed=passed,
        collected=nodes,
        passed_node_ids=outcomes,
        shards=shards,
        junit_shards=retained,
        original_authority=authority,
    )
    work["original_node_union"] = union
    work["original_122_node_union"] = dict(
        collected=original,
        passed=passed,
        passed_node_ids=[n for n in outcomes if n in set(original)],
        original_publication=previous.reference(e.ROOT / "results" / (previous.NAME + ".json")),
    )
    receipts.append(union)
    report = raw / "coverage.json"
    plans = [
        (
            "coverage_combine",
            [sys.executable, "-m", "coverage", "combine", "--rcfile=" + str(config), "--keep"],
        ),
        (
            "coverage_json",
            [
                sys.executable,
                "-m",
                "coverage",
                "json",
                "--rcfile=" + str(config),
                "-o",
                str(report),
            ],
        ),
        ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, *e.TESTS]),
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
        ("spec_coverage", [sys.executable, "scripts/check_spec_coverage.py", "--files", *e.TESTS]),
    ]
    receipts.extend(child(name, argv, raw / "checks", deadline=90) for name, argv in plans)
    coverage = json.loads(report.read_bytes())
    data_refs = []
    for source in sorted(scratch.glob(".coverage*")):
        target = raw / "coverage_data" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        data_refs.append(previous.reference(target))
    measured = {str(Path(p).resolve()): row for p, row in coverage["files"].items()}
    owned = {str((e.ROOT / p).resolve()) for p in e.OWNED}
    covered = owned <= set(measured) and all(
        not measured[p]["missing_lines"] and measured[p]["summary"]["num_statements"] > 0
        for p in owned
    )
    work["child_statement_coverage"] = dict(
        passed=covered,
        report=previous.reference(report),
        files=coverage["files"],
        subprocess_patch="Coverage.py patch=subprocess; actual CLI children",
        owned_files=e.OWNED,
        data_references=data_refs,
    )
    receipts.extend(
        [
            dict(
                name="new_owned_statement_coverage_100",
                passed=covered,
                report=previous.reference(report),
            ),
            dict(name="current_methods", passed=not work.get("current_gate_check_summary")),
            dict(
                name="historical_substitution_rejected",
                passed=bool(work.get("historical_replay_rejection_rows")),
            ),
        ]
    )
    return receipts
