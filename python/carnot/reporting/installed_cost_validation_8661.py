"""REQ-VERIFY-8661: disjoint children keep exact node and raw coverage custody."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import coverage
from typing import Any

from carnot.reporting import installed_cost_8661 as c
from carnot.reporting.current_validator_8580 import xml_nodes
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]
E2E = [
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def validate(work: Json, raw: Path) -> list[Json]:
    """Collect once, instrument actual sources, retain originals, then combine copies."""
    folder = raw / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="carnot-8661-validation-", dir="/var/tmp"))
    suites = c.TESTS + E2E
    collected = child(
        "collection",
        [sys.executable, "-m", "pytest", *COMMON, "--collect-only", "-q", *suites],
        folder,
        deadline=120,
    )
    nodes = [
        line
        for line in Path(collected["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/python/") and "::" in line
    ]
    atomic_json(
        folder / "nodes.json", dict(nodes=nodes, sources=[c.reference(c.ROOT / p) for p in suites])
    )
    receipts = [collected]
    actual = []
    shards = []
    for index, suite in enumerate(suites):
        selected = [n for n in nodes if n.startswith(suite + "::")]
        if not selected:
            receipts.append(dict(name="missing_suite_" + str(index), passed=False))
            continue
        data = scratch / str(index)
        data.mkdir()
        config = data / "coverage.ini"
        sources = (
            [str(c.ROOT / p) for p in c.OWNED]
            if suite in c.TESTS
            else [str(c.ROOT / "python/carnot/*")]
        )
        config.write_text(
            "[run]\nparallel=true\npatch=subprocess\ninclude="
            + ",".join(sources)
            + "\n[report]\nexclude_lines=\n"
        )
        xml = folder / (str(index) + ".xml")
        name = "owned-" + str(index) if suite in c.TESTS else "e2e018" if index == 2 else "e2e021"
        receipt = child(
            name,
            [
                "env",
                f"COVERAGE_RCFILE={config}",
                f"COVERAGE_FILE={data / '.coverage'}",
                sys.executable,
                "-u",
                "-m",
                "coverage",
                "run",
                "-m",
                "pytest",
                *COMMON,
                "-q",
                *selected,
                "--junitxml=" + str(xml),
                "--basetemp=" + str(scratch / ("tests-" + str(index))),
            ],
            folder,
            deadline=180,
        )
        receipts.append(receipt)
        if receipt["passed"] and xml.exists():
            actual.extend(xml_nodes(xml))
        retained = folder / "shards" / str(index)
        retained.mkdir(parents=True)
        captured = []
        for source in sorted(data.glob(".coverage*")):
            destination = retained / source.name
            shutil.copyfile(source, destination)
            captured.append(c.reference(destination))
        measured = set()
        for ref in captured:
            sqlite = coverage.CoverageData(basename=ref["path"])
            sqlite.read()
            measured.update(sqlite.measured_files())
        source_hashes = [c.reference(Path(p)) for p in sorted(measured) if Path(p).is_file()]
        shards.extend(captured if suite in c.TESTS else [])
        atomic_json(
            retained / "manifest.json",
            dict(
                command=receipt,
                sources=sources,
                source_hashes=source_hashes,
                sqlite=captured,
                junit=c.reference(xml) if xml.exists() else None,
            ),
        )
        receipts.append(dict(name="shard_present_" + str(index), passed=bool(captured)))
    receipts.extend(
        [
            dict(
                name="owned_tests",
                passed=all(r["passed"] for r in receipts if r["name"].startswith("owned-")),
            ),
            dict(
                name="exact_node_union",
                passed=collected["passed"]
                and bool(nodes)
                and len(nodes) == len(set(nodes))
                and sorted(nodes) == sorted(actual),
            ),
        ]
    )
    if not shards:
        return receipts + [dict(name="owned_coverage", passed=False)]
    combined = scratch / "combined"
    combined.mkdir()
    for index, ref in enumerate(shards):
        shutil.copyfile(ref["path"], combined / (".coverage." + str(index)))
    datafile = folder / "combined.coverage"
    cov = coverage.Coverage(data_file=str(datafile), config_file=False)
    cov.combine(data_paths=[str(p) for p in sorted(combined.glob(".coverage.*"))], keep=True)
    cov.save()
    report = folder / "coverage.json"
    cov.json_report(
        outfile=str(report), include=[str(c.ROOT / p) for p in c.OWNED], ignore_errors=False
    )
    files = json.loads(report.read_bytes())["files"]
    normalized = {str(Path(p).resolve()): row for p, row in files.items()}
    owned = {p: normalized.get(str(c.ROOT / p), {}) for p in c.OWNED}
    receipts.append(
        dict(
            name="owned_coverage",
            passed=all(
                r.get("summary", {}).get("num_statements", 0) > 0 and not r.get("missing_lines")
                for r in owned.values()
            ),
            files=owned,
        )
    )
    before = [c.reference(Path(ref["path"])) for ref in shards]
    cov.combine(data_paths=[str(p) for p in sorted(combined.glob(".coverage.*"))], keep=True)
    cov.save()
    receipts.append(
        dict(
            name="original_sqlite_unchanged",
            passed=before == [c.reference(Path(ref["path"])) for ref in shards],
        )
    )
    for name, argv in [
        ("ruff_check", ["ruff", "check", *c.OWNED, *c.TESTS]),
        ("ruff_format", ["ruff", "format", "--check", *c.OWNED, *c.TESTS]),
        ("strict_mypy", ["mypy", "--strict", "--follow-imports=silent", *c.OWNED]),
        ("spec_coverage", ["python", "scripts/check_spec_coverage.py", "--files", *c.TESTS]),
    ]:
        receipts.append(
            child(name, [str(c.ROOT / ".venv/bin" / argv[0]), *argv[1:]], folder, deadline=60)
        )
    return receipts
