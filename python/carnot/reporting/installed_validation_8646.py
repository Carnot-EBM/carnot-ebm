"""REQ-VERIFY-8646: execute original and current nodes with separate data custody."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

from carnot.reporting import installed_compatibility_8646 as e
from carnot.reporting import installed_validation_8634 as old
from carnot.reporting.current_validator_8580 import xml_nodes
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.lean_validation_8621 import seal_authority
from carnot.reporting.v709_execution import child, progress

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]
PUBLIC = old.PUBLIC + [
    "tests/python/test_installed_compatibility_8634.py",
    "tests/python/test_installed_validation_8634.py",
]


def coverage_result(files: Json) -> Json:
    """Only actual newly owned current statements can qualify current coverage."""
    normalized = {str(Path(p).resolve()): row for p, row in files.items()}
    rows = {p: normalized.get(str(e.ROOT / p), {}) for p in e.OWNED}
    return dict(
        name="owned_coverage",
        passed=all(
            r.get("summary", {}).get("num_statements", 0) > 0 and not r.get("missing_lines")
            for r in rows.values()
        ),
        files=rows,
    )


def validate(work: Json, raw: Path) -> list[Json]:
    """Capture raw shards after each child; combine copies and never reuse old summaries."""
    folder = raw / "validation"
    folder.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="carnot-8646-validation-", dir="/var/tmp"))
    config = folder / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ninclude="
        + str(e.ROOT / "python/carnot/*")
        + ","
        + str(e.ROOT / e.CLI)
        + "\n[report]\nexclude_lines=\n"
    )
    original = work.get("original_nodes", [])
    original_suites = list(dict.fromkeys(n.split("::")[0] for n in original))
    families = list(dict.fromkeys(original_suites + e.TESTS + PUBLIC))
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
    original_set = set(original)
    current = [n for n in nodes if n not in original_set]
    historical_nodes = [n for n in nodes if n in original_set]
    atomic_json(
        folder / "collected_nodes.json",
        dict(
            original=original,
            collected=nodes,
            current=current,
            source_hashes=[e.reference(e.ROOT / f.split("::")[0]) for f in families],
        ),
    )
    authorities = []
    if original:
        authorities = [
            seal_authority(scratch / "original_v738"),
            seal_authority(scratch / "original_checkout", live_source=True),
        ]
        overlay = old.original_overlay(folder / "historical_overlay", authorities)
    else:
        overlay = []
    receipts, outcomes, shards = [collected], [], []
    for index, suite in enumerate(dict.fromkeys(n.split("::")[0] for n in nodes)):
        progress("8646_disjoint_tests", index, len(families) - index)
        selected = [n for n in nodes if n.startswith(suite + "::")]
        historical = suite in original_suites
        data_dir = scratch / f"child-{index}"
        data_dir.mkdir()
        data = data_dir / ".coverage"
        prefix = ["env", f"COVERAGE_RCFILE={config}", f"COVERAGE_FILE={data}"]
        root = e.ROOT
        if suite in {
            "tests/python/test_resident_provenance_8593.py",
            "tests/python/test_lean_runner_8609.py",
        }:
            root = Path(authorities[int("runner" in suite)]["root"])
            prefix += [f"PYTHONPATH={root}/python:{root}", f"CARNOT_REPO_ROOT={root}"]
        xml = folder / f"nodes-{index}.xml"
        identity = folder / f"imports-{index}.json"
        import_code = (
            f"sys.path.insert(0,{str(root / 'python/carnot/inference')!r});import resident_protocol as p;"
            if historical
            else "import carnot.inference.resident_protocol as p;"
        )
        code = (
            "import os,sys,json,hashlib;from pathlib import Path;"
            f"os.chdir({str(root)!r});"
            + import_code
            + "r=dict(path=p.__file__,sha256='sha256:'+hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest());"
            f"assert r['sha256']=={(e.HASH if historical else e.reference(e.ROOT / e.PROTOCOL)['sha256'])!r};"
            f"Path({str(identity)!r}).write_text(json.dumps(r));"
            "from coverage.cmdline import main;raise SystemExit(main(sys.argv[1:]))"
        )
        receipt = child(
            f"nodes-{index}",
            [
                *(overlay if historical else []),
                *prefix,
                sys.executable,
                "-u",
                "-c",
                code,
                "run",
                "-m",
                "pytest",
                *COMMON,
                "-q",
                *selected,
                "--junitxml=" + str(xml),
                "--basetemp=" + str(scratch / f"tests-{index}"),
            ],
            folder,
            deadline=360,
        )
        receipts.append(receipt)
        if receipt["passed"] and xml.exists():
            outcomes.extend(xml_nodes(xml))
        if identity.exists():
            work.setdefault("resolved_import_rows", []).append(
                dict(
                    json.loads(identity.read_bytes()),
                    historical=historical,
                    passed=receipt["passed"],
                )
            )
        retained = folder / "shards" / ("historical" if historical else "current") / str(index)
        retained.mkdir(parents=True)
        for shard in sorted(data_dir.glob(".coverage*")):
            target = retained / shard.name
            shutil.copyfile(shard, target)
            shards.append(
                dict(
                    e.reference(target),
                    historical=historical,
                    nodes=selected,
                    source=e.reference(e.ROOT / suite),
                )
            )
    work["original_node_union"] = dict(
        e.union(original, [n for n in outcomes if n in original_set]),
        name="original122",
        intended_count=122,
        original_roster_exact=len(original) == 122 and sorted(original) == sorted(historical_nodes),
        authorities=authorities,
    )
    work["original_node_union"]["passed"] &= work["original_node_union"]["original_roster_exact"]
    work["current_node_union"] = e.union(current, [n for n in outcomes if n not in original_set])
    exact = (
        work["original_node_union"]["passed"]
        and work["current_node_union"]["passed"]
        and len(nodes) == len(set(nodes))
    )
    receipts.append(dict(name="exact_node_union", passed=exact))
    for name, suite in [
        ("e2e018", "test_experiment_7891_v685_authority_lifecycle.py"),
        ("e2e021", "test_restricted_decision_audit_8210.py"),
        ("primary_publication", "test_primary_publication_7928.py"),
    ]:
        selected = [n for n in nodes if suite + "::" in n]
        receipts.append(dict(name=name, passed=bool(selected) and set(selected) <= set(outcomes)))
    copies = folder / "combine-copies"
    copies.mkdir()
    for index, shard in enumerate(shards):
        if not shard["historical"]:
            shutil.copyfile(shard["path"], copies / f".coverage.shard-{index}")
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
        ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, *e.TESTS]),
        ("ruff_format", [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, *e.TESTS]),
        (
            "strict_mypy",
            [str(e.ROOT / ".venv/bin/mypy"), "--strict", "--follow-imports=silent", *e.OWNED],
        ),
        ("spec_coverage", [sys.executable, "scripts/check_spec_coverage.py", "--files", *e.TESTS]),
    ]
    receipts.extend(child(name, argv, folder, deadline=120) for name, argv in plans)
    report = folder / "coverage.json"
    receipts.append(
        coverage_result(json.loads(report.read_bytes())["files"] if report.exists() else {})
    )
    work["child_shard_manifest"] = shards
    atomic_json(folder / "receipts.json", dict(receipts=receipts))
    return receipts
