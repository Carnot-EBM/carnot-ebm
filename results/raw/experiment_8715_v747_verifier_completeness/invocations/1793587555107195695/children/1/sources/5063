"""REQ-REPORT-8533: exact collected shards, child coverage and bounded tool receipts."""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import nine_input_qualification_8533 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child
from carnot.reporting.historical_consumer_roots_8402 import parse_outcomes
from carnot.reporting import v727_historical_inputs as history
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]
SUITES = [
    "tests/python/test_sparse_energy_7996.py",
    "tests/python/test_v726_criterion_energy_fit_8431.py",
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def validate(work: Json, raw: Path) -> list[Json]:
    """Collect unchanged node IDs before executing their exact disjoint union."""
    with TemporaryDirectory(prefix="exp8533-tests-", dir="/var/tmp") as directory:
        return _validate(work, raw, Path(directory))


def _validate(work: Json, raw: Path, private: Path) -> list[Json]:
    """Private fixtures stay outside repository guards; command logs remain durable."""
    logs = raw / "validation"
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    py = str(e.ROOT / ".venv/bin/python")
    pytest = str(e.ROOT / ".venv/bin/pytest")
    prefix = [
        "env",
        "CARNOT_8533_EXTENSION=" + work["extension"].get("path", ""),
        "LLVM_PROFILE_FILE=" + str(raw / "native_logs/rust-%p-%m.profraw"),
    ]
    receipts, nodes, terminal = [], [], []
    suites = [e.TEST, *SUITES]
    shards = []
    for i, suite in enumerate(suites):
        receipt = child(
            f"collect_{i}", [pytest, *COMMON, "--collect-only", "-q", suite], logs, deadline=120
        )
        receipts.append(receipt)
        found = [
            l.strip()
            for l in Path(receipt["stdout_path"]).read_text().splitlines()
            if l.startswith("tests/") and "::" in l
        ]
        nodes.extend(found)
        shards.append(found)
    atomic_json(
        raw / "collection.json",
        dict(node_ids=nodes, disjoint=len(nodes) == len(set(nodes)), suites=suites),
    )
    for i, shard in enumerate(shards):
        argv = (
            [py, "-m", "coverage", "run", "--rcfile=" + str(config), "-m", "pytest"]
            if i == 0
            else [pytest]
        )
        if i == 2:
            argv = [py, "-u", str(e.ROOT / e.CLI), "--historical-tests"]
        receipt = child(
            f"shard_{i}",
            prefix + argv + [*COMMON, "-vv", *shard, "--basetemp=" + str(private / f"pytest-{i}")],
            logs,
            deadline=360,
        )
        receipts.append(receipt)
        outcomes = parse_outcomes(Path(receipt["stdout_path"]))
        terminal.extend(outcomes["passed"])
    union_ok = (
        bool(nodes)
        and len(nodes) == len(set(nodes))
        and len(terminal) == len(nodes)
        and set(nodes) == set(terminal)
    )
    atomic_json(
        raw / "node_union.json", dict(passed=union_ok, collected=nodes, passed_node_ids=terminal)
    )
    receipts.append(dict(name="exact_disjoint_node_union", passed=union_ok))
    commands = [
        (
            "coverage_combine",
            [py, "-m", "coverage", "combine", "--rcfile=" + str(config), "--keep"],
        ),
        (
            "coverage_report",
            [
                py,
                "-m",
                "coverage",
                "report",
                "--rcfile=" + str(config),
                "--show-missing",
                "--fail-under=100",
            ],
        ),
        (
            "coverage_json",
            [
                py,
                "-m",
                "coverage",
                "json",
                "--rcfile=" + str(config),
                "-o",
                str(raw / "coverage.json"),
            ],
        ),
        ("ruff_check", [str(e.ROOT / ".venv/bin/ruff"), "check", *e.OWNED, e.TEST]),
        ("ruff_format", [str(e.ROOT / ".venv/bin/ruff"), "format", "--check", *e.OWNED, e.TEST]),
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
        ("spec_coverage", [py, "scripts/check_spec_coverage.py", "--files", e.TEST]),
    ]
    for name, argv in commands:
        receipts.append(child(name, argv, logs, deadline=90))
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=[r.get("argv") for r in receipts],
            task_cap_s=4800,
            measurement_cap_s=3600,
            validation_reserve_s=900,
            heartbeat_s=30,
        ),
    )
    return receipts


def rust_coverage(work: Json, raw: Path) -> list[Json]:
    """Combine genuine Rust unit and actual PyO3 execution profiles, scoped to owned sources."""
    logs = raw / "rust_coverage"
    profiles = sorted((raw / "native_logs").glob("*.profraw"))
    profdata = raw / "rust.profdata"
    merge = child(
        "llvm_merge",
        ["llvm-profdata", "merge", "-sparse", *map(str, profiles), "-o", str(profdata)],
        logs,
        deadline=60,
    )
    objects = [
        Path(work["extension"]["path"]),
        *sorted((raw / "native_binary").glob("nine_input_8533-*")),
    ]
    argv = ["llvm-cov", "export", "--instr-profile=" + str(profdata)]
    for path in objects:
        argv += ["--object", str(path)]
    export = child("llvm_export", argv, logs, deadline=60)
    files = []
    if export["passed"]:
        data = json.loads(Path(export["stdout_path"]).read_bytes())
        files = [
            f
            for d in data["data"]
            for f in d["files"]
            if f["filename"].endswith("/src/nine_input_8533.rs")
        ]
    passed = len(files) >= 2 and all(f["summary"]["lines"]["percent"] == 100 for f in files)
    atomic_json(
        raw / "rust_coverage_summary.json",
        dict(
            passed=passed,
            files=files,
            scope="owned Rust source lines including binding, excluding generated PyO3 macro internals",
        ),
    )
    return [merge, export, dict(name="owned_rust_coverage_100", passed=passed)]


def historical(argv: list[str], *, script: bool = False) -> int:
    """Run unchanged V726 assertions under their original authority, including CLI children."""
    import pytest

    with TemporaryDirectory(prefix="exp8533-v726-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        refs = history.custody(e.ROOT, scratch / "authority")
        popen = subprocess.Popen
        original_cli = str(
            e.ROOT / "scripts/experiments/experiment_8431_v726_criterion_energy_fit.py"
        )

        def start(args: Any, *pos: Any, **kw: Any) -> Any:
            """Propagate original authority without running a historical validation orchestrator."""
            if isinstance(args, (list, tuple)) and original_cli in args:
                index = args.index(original_cli)
                args = [*args[:index], str(e.ROOT / e.CLI), "--historical-exec", *args[index:]]
            return popen(args, *pos, **kw)

        with frozen_inputs(refs, scratch), patch.object(subprocess, "Popen", start):
            if script:
                previous = sys.argv
                sys.argv = argv
                try:
                    runpy.run_path(argv[0], run_name="__main__")
                except SystemExit as error:
                    return int(error.code) if isinstance(error.code, int) else 0
                finally:
                    sys.argv = previous
                return 0
            return int(pytest.main(argv))
