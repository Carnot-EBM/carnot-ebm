"""REQ-VERIFY-8633: exact original node union and real child statement coverage."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

from carnot.reporting import installed_lean_8633 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.current_validator_8580 import xml_nodes
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]
GROUPS = [
    ("owned_tests", e.TESTS + ["tests/python/test_lean_resident_8608.py"]),
    ("e2e018", ["tests/python/test_experiment_7891_v685_authority_lifecycle.py"]),
    ("e2e021", ["tests/python/test_restricted_decision_audit_8210.py"]),
    ("primary_publication", ["tests/python/test_primary_publication_7928.py"]),
]


def changed_statements(files: dict[str, Any]) -> Json:
    """Require every new executable line without assigning old uncovered code to this task."""
    rows = {}
    for path in e.OWNED:
        lines = files.get(path, {})
        executable = set(lines.get("executed_lines", [])) | set(lines.get("missing_lines", []))
        if path in {"python/carnot/__init__.py", "python/carnot/inference/resident_protocol.py"}:
            diff = subprocess.run(
                ["git", "diff", "--unified=0", "--", path],
                cwd=e.ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            added = set()
            for start, length in re.findall(r"^@@ .*?\+(\d+)(?:,(\d+))? @@", diff, re.M):
                added.update(range(int(start), int(start) + int(length or 1)))
            executable &= added
        missing = executable & set(lines.get("missing_lines", []))
        rows[path] = dict(
            statements=sorted(executable),
            missing_lines=sorted(missing),
            passed=bool(executable) and not missing,
        )
    return dict(name="owned_coverage", passed=all(r["passed"] for r in rows.values()), files=rows)


def validate(raw: Path) -> list[Json]:
    """Collect once, then execute disjoint bounded children with unchanged authority fixtures."""
    folder = raw / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    config = folder / "coverage.ini"
    config.write_text(
        "[run]\nparallel = true\npatch = subprocess\ninclude = "
        + str(e.ROOT / "python/carnot/*")
        + ","
        + str(e.ROOT / e.CLI)
        + ",/var/tmp/carnot-8633-*/environment/lib/python*/site-packages/carnot/*"
        + "\n[paths]\npackage =\n    "
        + str(e.ROOT / "python/carnot")
        + "\n    /var/tmp/carnot-8633-*/environment/lib/python*/site-packages/carnot"
        + "\n[report]\nexclude_lines =\n"
    )
    data = folder / ".coverage"
    env = ["env", f"COVERAGE_RCFILE={config}", f"COVERAGE_FILE={data}"]
    tests = [p for _, paths in GROUPS for p in paths]
    collected = child(
        "collect",
        [sys.executable, "-m", "pytest", *COMMON, "--collect-only", "-q", *tests],
        folder,
        deadline=90,
    )
    receipts = [collected]
    original = [
        line
        for line in Path(collected["stdout_path"]).read_text().splitlines()
        if any(line.startswith(p + "::") for p in tests)
    ]
    passed = []
    for name, paths in GROUPS:
        nodes = [node for node in original if any(node.startswith(p + "::") for p in paths)]
        xml = folder / (name + ".xml")
        receipts.append(
            child(
                name,
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
                    *(nodes or paths),
                    f"--junitxml={xml}",
                ],
                folder,
                deadline=240,
            )
        )
        if xml.exists():
            passed.extend(xml_nodes(xml))
    receipts.append(
        dict(
            name="exact_node_union",
            passed=bool(original)
            and sorted(original) == sorted(passed)
            and len(original) == len(set(original)),
            original_node_ids=original,
            passed_node_ids=passed,
        )
    )
    receipts.append(
        child(
            "combine",
            [*env, sys.executable, "-m", "coverage", "combine", "--keep", str(folder)],
            folder,
            deadline=60,
        )
    )
    report = folder / "coverage.json"
    receipts.append(
        child(
            "coverage_report",
            [*env, sys.executable, "-m", "coverage", "json", "-o", str(report)],
            folder,
            deadline=60,
        )
    )
    files = json.loads(report.read_bytes())["files"] if report.exists() else {}
    receipts.append(
        dict(changed_statements(files), report=e.reference(report) if report.exists() else None)
    )
    for name, argv in [
        ("ruff_check", ["ruff", "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", ["ruff", "format", "--check", *e.OWNED, *e.TESTS]),
        ("strict_mypy", ["mypy", "--strict", "--follow-imports=silent", *e.OWNED]),
        ("spec_coverage", ["python", "scripts/check_spec_coverage.py", "--files", *e.TESTS]),
    ]:
        receipts.append(
            child(name, [str(e.ROOT / ".venv/bin" / argv[0]), *argv[1:]], folder, deadline=120)
        )
    atomic_json(folder / "receipts.json", dict(receipts=receipts))
    return receipts
