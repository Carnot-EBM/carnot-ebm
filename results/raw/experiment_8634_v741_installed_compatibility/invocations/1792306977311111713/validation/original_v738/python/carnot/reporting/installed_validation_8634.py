"""REQ-VERIFY-8634: freeze node families, retain child shards and check owned code."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from carnot.reporting import installed_compatibility_8634 as e
from carnot.reporting.current_validator_8580 import xml_nodes
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.lean_validation_8621 import seal_authority
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]
COVERED = e.OWNED + ["python/carnot/__init__.py", "python/carnot/inference/resident_protocol.py"]
PUBLIC = [
    "tests/python/test_installed_lean_8633.py",
    "tests/python/test_cli.py::test_verify_correct_function",
    "tests/python/test_cli.py::test_verify_buggy_function",
    "tests/python/test_cli.py::test_main_no_args",
    "tests/python/test_mcp_server.py::TestHealthCheck",
    "tests/python/test_mcp_server.py::TestVerifyCode",
    "tests/python/test_abstention_http_rest_3801.py",
    "tests/python/test_abstention_http_rest_3810.py",
    "tests/python/test_primary_publication_7928.py",
]


def coverage_result(files: Json) -> Json:
    """Missing files cannot masquerade as measured files with zero missing statements."""
    rows = {
        p: dict(
            statements=files.get(p, {}).get("summary", {}).get("num_statements", 0),
            missing_lines=files.get(p, {}).get("missing_lines", []),
        )
        for p in COVERED
    }
    return dict(
        name="owned_coverage",
        passed=all(r["statements"] > 0 and not r["missing_lines"] for r in rows.values()),
        files=rows,
    )


def validate(work: Json, raw: Path) -> list[Json]:
    """Collect once and execute bounded disjoint children without historical controllers."""
    folder = raw / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    config, data = folder / "coverage.ini", folder / ".coverage"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ninclude="
        + str(e.ROOT / "python/carnot/*")
        + ","
        + str(e.ROOT / e.CLI)
        + ",/var/tmp/carnot-8634-*/environment/lib/python*/site-packages/carnot/*\n[paths]\npackage=\n    "
        + str(e.ROOT / "python/carnot")
        + "\n    /var/tmp/carnot-8634-*/environment/lib/python*/site-packages/carnot\n[report]\nexclude_lines=\n"
    )
    original = work["bundle"].get("original_nodes", [])
    historical_suites = list(dict.fromkeys(n.split("::")[0] for n in original))
    families = list(dict.fromkeys(historical_suites + e.TESTS + PUBLIC))
    atomic_json(
        folder / "declared_families.json",
        dict(
            families=families,
            original_nodes=original,
            test_hashes=[
                e.reference(e.ROOT / p) for p in dict.fromkeys(f.split("::")[0] for f in families)
            ],
        ),
    )
    collected = child(
        "collect",
        [sys.executable, "-m", "pytest", *COMMON, "--collect-only", "-q", *families],
        folder,
        deadline=90,
    )
    nodes = [
        line
        for line in Path(collected["stdout_path"]).read_text().splitlines()
        if line.startswith("tests/python/") and "::" in line
    ]
    suites = list(dict.fromkeys(n.split("::")[0] for n in nodes))
    authorities = [
        seal_authority(folder / "original_v738"),
        seal_authority(folder / "original_checkout", live_source=True),
    ]
    receipts, passed = [collected], []
    for index, suite in enumerate(suites):
        progress("8634_disjoint_tests", index, len(suites) - index)
        selected = [n for n in nodes if n.startswith(suite + "::")]
        xml = folder / f"nodes_{index}.xml"
        prefix = ["env", f"COVERAGE_RCFILE={config}", f"COVERAGE_FILE={data}"]
        command = [sys.executable, "-m", "coverage"]
        if suite in {
            "tests/python/test_resident_provenance_8593.py",
            "tests/python/test_lean_runner_8609.py",
        }:
            root = authorities[int("runner" in suite)]["root"]
            prefix += ["PYTHONPATH=" + root + "/python:" + root, "CARNOT_REPO_ROOT=" + root]
            command = [
                sys.executable,
                "-c",
                f"import os,sys;os.chdir({root!r});from coverage.cmdline import main;raise SystemExit(main(sys.argv[1:]))",
            ]
        receipt = child(
            f"nodes_{index}",
            [
                *prefix,
                *command,
                "run",
                "-m",
                "pytest",
                *COMMON,
                "-q",
                *selected,
                "--junitxml=" + str(xml),
            ],
            folder,
            deadline=360,
        )
        receipts.append(receipt)
        if receipt["passed"] and xml.exists():
            passed.extend(xml_nodes(xml))
    work["original_node_union"] = dict(
        name="exact_node_union",
        passed=bool(nodes)
        and len(nodes) == len(set(nodes))
        and sorted(nodes) == sorted(passed)
        and len(original) == 122
        and set(original) <= set(nodes),
        original_nodes=original,
        collected=nodes,
        passed_node_ids=passed,
        declared_families=families,
        authorities=authorities,
    )
    receipts.append(work["original_node_union"])
    for name, suite in [
        ("e2e018", "test_experiment_7891_v685_authority_lifecycle.py"),
        ("e2e021", "test_restricted_decision_audit_8210.py"),
        ("primary_publication", "test_primary_publication_7928.py"),
    ]:
        selected = [n for n in nodes if suite + "::" in n]
        receipts.append(dict(name=name, passed=bool(selected) and set(selected) <= set(passed)))
    env = ["env", f"COVERAGE_RCFILE={config}", f"COVERAGE_FILE={data}"]
    for name, argv in [
        (
            "coverage_combine",
            [*env, sys.executable, "-m", "coverage", "combine", "--keep", str(folder)],
        ),
        (
            "coverage_json",
            [*env, sys.executable, "-m", "coverage", "json", "-o", str(folder / "coverage.json")],
        ),
        ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, *e.TESTS]),
        (
            "strict_mypy",
            [str(e.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *e.OWNED],
        ),
        ("spec_coverage", [sys.executable, "scripts/check_spec_coverage.py", "--files", *e.TESTS]),
    ]:
        receipts.append(child(name, argv, folder, deadline=120))
    report = folder / "coverage.json"
    files = json.loads(report.read_bytes())["files"] if report.exists() else {}
    receipts.append(coverage_result(files))
    work["child_shard_manifest"] = [
        e.reference(p) for p in sorted(folder.glob(".coverage.*")) if p.is_file()
    ]
    work["coverage_reference"] = e.reference(report) if report.exists() else None
    atomic_json(folder / "receipts.json", dict(receipts=receipts))
    return receipts
