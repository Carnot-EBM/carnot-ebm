"""REQ-VERIFY-8660: exact disjoint tests retain source and cleanup custody."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from carnot.reporting import installed_checkout_8660 as c
from carnot.reporting import installed_compatibility_8646 as old
from carnot.reporting.installed_validation_8646 import PUBLIC
from carnot.reporting.current_validator_8580 import xml_nodes
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
ROOT = c.ROOT
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]


def coverage_result(files: Json) -> Json:
    """Each owned file must have real executable statements and no missing lines."""
    from carnot.reporting.installed_report_8660 import OWNED

    normalized = {str(Path(p).resolve()): row for p, row in files.items()}
    rows = {p: normalized.get(str(ROOT / p), {}) for p in OWNED}
    return dict(
        name="owned_coverage",
        passed=all(
            row.get("summary", {}).get("num_statements", 0) > 0 and not row.get("missing_lines")
            for row in rows.values()
        ),
        files=rows,
    )


def historical_command(manifest: Json, folder: Path, suite: str) -> list[str]:
    """Only historical children see the old protocol and sealed scheduling inputs."""
    argv = [
        "bwrap",
        "--bind",
        "/",
        "/",
        "--dev-bind",
        "/dev",
        "/dev",
        "--proc",
        "/proc",
        "--ro-bind",
        manifest["original_protocol"]["path"],
        str(ROOT / c.PROTOCOL),
    ]
    if suite.endswith(("test_resident_provenance_8593.py", "test_lean_runner_8609.py")):
        protocol = json.loads(
            (ROOT / "openspec/change-proposals/v739-methods-protocol.json").read_bytes()
        )
        for key, destination in [
            ("v738_design_reference", "openspec/change-proposals/research-roadmap-vNEXT.md"),
            ("v738_activation_reference", "research-roadmap-next.yaml"),
        ]:
            ref = protocol[key]
            if c.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("historical_authority_hash")
            argv += ["--ro-bind", ref["path"], str(ROOT / destination)]
        if "provenance" in suite:
            argv += [
                "--ro-bind",
                protocol["v738_activation_reference"]["path"],
                str(ROOT / "research-roadmap.yaml"),
            ]
        argv += ["--ro-bind", manifest["materializer"]["path"], str(ROOT / c.HELPER)]
    return argv + ["--"]


def validate(work: Json, raw: Path) -> list[Json]:
    """Collect each source cohort once and preserve raw child SQLite before combine."""
    from carnot.reporting.installed_report_8660 import OWNED, TESTS

    folder = raw / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="carnot-8660-validation-", dir="/var/tmp"))
    manifest = c.overlay(folder / "overlay")
    original = work.get("original_nodes", [])
    original_suites = list(dict.fromkeys(n.split("::")[0] for n in original))
    current_suites = list(dict.fromkeys(TESTS + old.TESTS + PUBLIC))
    receipts, outcomes, current, shards = [], [], [], []
    for label, suites in [("historical", original_suites), ("current", current_suites)]:
        if not suites:
            continue
        collection = child(
            "collect-" + label,
            [sys.executable, "-m", "pytest", *COMMON, "--collect-only", "-q", *suites],
            folder,
            deadline=120,
        )
        receipts.append(collection)
        nodes = [
            line
            for line in Path(collection["stdout_path"]).read_text().splitlines()
            if line.startswith("tests/python/") and "::" in line
        ]
        if label == "historical" and sorted(nodes) != sorted(original):
            receipts.append(dict(name="original_collection_exact", passed=False))
        if label == "current":
            current = nodes
        atomic_json(
            folder / (label + "-nodes.json"),
            dict(nodes=nodes, sources=[c.reference(ROOT / p.split("::")[0]) for p in suites]),
        )
        for index, suite in enumerate(dict.fromkeys(n.split("::")[0] for n in nodes)):
            progress("8660_" + label + "_tests", index, len(suites) - index)
            selected = [n for n in nodes if n.startswith(suite + "::")]
            data_dir = scratch / (label + "-" + str(index))
            data_dir.mkdir()
            config = data_dir / "coverage.ini"
            config.write_text(
                "[run]\nparallel=true\npatch=subprocess\ninclude="
                + str(ROOT / "python/carnot/*")
                + ","
                + str(ROOT / "scripts/experiments/experiment_8660_v743_installed_compatibility.py")
                + "\n[report]\nexclude_lines=\n"
            )
            xml = folder / (label + "-" + str(index) + ".xml")
            prefix = historical_command(manifest, folder, suite) if label == "historical" else []
            plugin = (
                ["-p", "carnot.reporting.installed_checkout_8660"]
                if label == "historical" and "lean_runner_8609" in suite
                else []
            )
            materialized = folder / "materialized"
            receipt = child(
                label + "-" + str(index),
                [
                    *prefix,
                    "env",
                    f"CARNOT_8660_MATERIALIZATION_LOG={materialized}",
                    f"COVERAGE_RCFILE={config}",
                    f"COVERAGE_FILE={data_dir / '.coverage'}",
                    sys.executable,
                    "-u",
                    "-m",
                    "coverage",
                    "run",
                    "-m",
                    "pytest",
                    *plugin,
                    *COMMON,
                    "-q",
                    *selected,
                    "--junitxml=" + str(xml),
                    "--basetemp=" + str(scratch / (label + "-tests-" + str(index))),
                ],
                folder,
                deadline=420,
            )
            receipts.append(receipt)
            if receipt["passed"] and xml.exists():
                outcomes.extend((label, node) for node in xml_nodes(xml))
            retained = folder / "shards" / label / str(index)
            retained.mkdir(parents=True)
            for shard in sorted(data_dir.glob(".coverage*")):
                target = retained / shard.name
                shutil.copyfile(shard, target)
                shards.append(
                    dict(
                        c.reference(target),
                        historical=label == "historical",
                        nodes=selected,
                        source=c.reference(ROOT / suite),
                        receipt=receipt["name"],
                    )
                )
    work["original_node_union"] = dict(
        old.union(original, [n for label, n in outcomes if label == "historical"]),
        original_nodes=original,
        intended_count=122,
        authority=manifest,
    )
    work["original_node_union"]["passed"] &= len(original) == 122
    work["current_node_union"] = old.union(
        current, [n for label, n in outcomes if label == "current"]
    )
    receipts.append(
        dict(
            name="exact_node_union",
            passed=work["original_node_union"]["passed"] and work["current_node_union"]["passed"],
        )
    )
    for name, suite in [
        ("e2e018", "test_experiment_7891_v685_authority_lifecycle.py"),
        ("e2e021", "test_restricted_decision_audit_8210.py"),
        ("primary_publication", "test_primary_publication_7928.py"),
    ]:
        wanted = [n for n in original + current if suite + "::" in n]
        receipts.append(
            dict(name=name, passed=bool(wanted) and set(wanted) <= {n for _, n in outcomes})
        )
    copies = folder / "combine-copies"
    copies.mkdir()
    for index, ref in enumerate(shards):
        if not ref["historical"]:
            shutil.copyfile(ref["path"], copies / (".coverage.shard-" + str(index)))
    config = folder / "coverage.ini"
    config.write_text(
        "[run]\ninclude=" + ",".join(str(ROOT / p) for p in OWNED) + "\n[report]\nexclude_lines=\n"
    )
    env = ["env", f"COVERAGE_RCFILE={config}", f"COVERAGE_FILE={folder / '.coverage'}"]
    plans = [
        (
            "coverage_combine",
            [*env, sys.executable, "-m", "coverage", "combine", "--keep", str(copies)],
        ),
        (
            "coverage_json",
            [*env, sys.executable, "-m", "coverage", "json", "-o", str(folder / "coverage.json")],
        ),
        ("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, *TESTS]),
        ("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, *TESTS]),
        (
            "strict_mypy",
            [str(ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *OWNED],
        ),
        ("spec_coverage", [sys.executable, "scripts/check_spec_coverage.py", "--files", *TESTS]),
    ]
    receipts.extend(child(name, argv, folder, deadline=120) for name, argv in plans)
    report = folder / "coverage.json"
    receipts.append(
        coverage_result(json.loads(report.read_bytes())["files"] if report.exists() else {})
    )
    work["child_shard_manifest"] = shards
    manifest["copies"] = [
        json.loads(p.read_bytes()) for p in sorted((folder / "materialized").glob("*.json"))
    ]
    work["materialization_manifest"] = manifest
    atomic_json(folder / "receipts.json", dict(receipts=receipts))
    return receipts
