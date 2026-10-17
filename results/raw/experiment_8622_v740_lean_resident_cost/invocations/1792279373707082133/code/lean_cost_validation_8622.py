"""REQ-VERIFY-8622: collect original test identities before bounded disjoint execution."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from carnot.reporting import lean_cost_8622 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.current_validator_8580 import xml_nodes
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]


def validate(raw: Path) -> list[Json]:
    """Use executable Coverage.py statements, including actual subprocess shards."""
    folder = raw / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    config = folder / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude = "
        + ",".join(str(e.ROOT / p) for p in e.OWNED)
        + "\n[report]\nexclude_lines =\n"
    )
    data = folder / ".coverage"
    env = ["env", f"COVERAGE_RCFILE={config}", f"COVERAGE_FILE={data}"]
    receipts = []
    collected = []
    passed = []
    for i, test in enumerate(e.TESTS):
        receipt = child(
            f"collect{i}",
            [sys.executable, "-m", "pytest", *COMMON, "--collect-only", "-q", test],
            folder,
            deadline=90,
        )
        receipts.append(receipt)
        nodes = [
            s
            for s in Path(receipt["stdout_path"]).read_text().splitlines()
            if s.startswith(test + "::")
        ]
        collected.extend(nodes)
    for i, test in enumerate(e.TESTS):
        nodes = [node for node in collected if node.startswith(test + "::")]
        xml = folder / f"tests{i}.xml"
        run = child(
            f"tests{i}",
            [
                *env,
                sys.executable,
                "-m",
                "coverage",
                "run",
                "-m",
                "pytest",
                *COMMON,
                "-q",
                *(nodes or [test]),
                f"--junitxml={xml}",
            ],
            folder,
            deadline=180,
        )
        receipts.append(run)
        if xml.exists():
            passed.extend(xml_nodes(xml))
    receipts.extend(
        [
            dict(name="owned_tests", passed=all(r["passed"] for r in receipts)),
            dict(
                name="exact_node_union",
                passed=bool(collected)
                and sorted(collected) == sorted(passed)
                and len(set(collected)) == len(collected),
                original_node_ids=collected,
                passed_node_ids=passed,
            ),
        ]
    )
    receipts.append(
        child(
            "combine",
            [*env, sys.executable, "-m", "coverage", "combine", str(folder)],
            folder,
            deadline=60,
        )
    )
    report = folder / "coverage.json"
    receipts.append(
        child(
            "coverage_report",
            [
                *env,
                sys.executable,
                "-m",
                "coverage",
                "json",
                "-o",
                str(report),
                "--include=" + ",".join(e.OWNED),
            ],
            folder,
            deadline=60,
        )
    )
    coverage = json.loads(report.read_bytes()) if report.exists() else {}
    files = coverage.get("files", {})
    receipts.append(
        dict(
            name="owned_coverage",
            passed=set(e.OWNED) <= set(files)
            and all(
                not files[p]["missing_lines"] and files[p]["summary"]["num_statements"] > 0
                for p in e.OWNED
                if p in files
            ),
            report=e.reference(report) if report.exists() else None,
            files=files,
        )
    )
    for name, args in [
        ("ruff_check", ["ruff", "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", ["ruff", "format", "--check", *e.OWNED, *e.TESTS]),
        ("strict_mypy", ["mypy", "--strict", "--follow-imports=silent", *e.OWNED]),
        ("spec_coverage", ["python", "scripts/check_spec_coverage.py", "--files", *e.TESTS]),
        (
            "e2e018",
            [
                "pytest",
                *COMMON,
                "-q",
                "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
            ],
        ),
        (
            "e2e021",
            ["pytest", *COMMON, "-q", "tests/python/test_restricted_decision_audit_8210.py"],
        ),
        (
            "primary_publication",
            ["pytest", *COMMON, "-q", "tests/python/test_primary_publication_7928.py"],
        ),
    ]:
        receipts.append(
            child(name, [str(e.ROOT / ".venv/bin" / args[0]), *args[1:]], folder, deadline=180)
        )
    atomic_json(folder / "receipts.json", dict(receipts=receipts))
    return receipts
