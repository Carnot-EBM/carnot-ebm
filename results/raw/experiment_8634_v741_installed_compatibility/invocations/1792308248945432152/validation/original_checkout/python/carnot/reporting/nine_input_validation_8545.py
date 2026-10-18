"""REQ-VERIFY-8545: original suites run in disjoint measured private children.

Test scratch stays outside results because historical write guards correctly
redirect repository writes. Coverage follows actual subprocesses before import.
"""

from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from carnot.reporting import nine_input_qualification_8545 as e
from carnot.reporting import nine_input_validation_8533 as old
from carnot.reporting import v727_historical_inputs as history
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.historical_consumer_roots_8402 import parse_outcomes
from carnot.reporting.v709_execution import child
from carnot.reporting.v721_capstone_evidence import frozen_inputs

Json = dict[str, Any]
COMMON = ["-n", "0", "-o", "addopts=", "--no-cov"]
SUITES = [e.TEST, e.legacy.TEST, *old.SUITES]


def historical(suite: str, argv: list[str], *, script: bool = False) -> int:
    """Bind each suite's own authority and carry it through its original CLI children."""
    import pytest

    with TemporaryDirectory(prefix="exp8545-sealed-", dir="/var/tmp") as directory:
        scratch = Path(directory)
        refs = (
            history.custody(e.ROOT, scratch / "authority")
            if "8431" in suite
            else e.original_authority()
            if "8533" in suite
            else []
        )
        for arg in argv:
            if arg.startswith("--basetemp="):
                Path(arg.split("=", 1)[1]).parent.mkdir(parents=True, exist_ok=True)
        original = subprocess.Popen
        cli = str(
            e.ROOT
            / (
                e.legacy.CLI
                if "8533" in suite
                else "scripts/experiments/experiment_8431_v726_criterion_energy_fit.py"
            )
        )

        def start(args: Any, *pos: Any, **kw: Any) -> Any:
            """Only the original producer CLI gains an explicit sealed-authority wrapper."""
            if isinstance(args, (tuple, list)) and cli in args:
                index = args.index(cli)
                args = [*args[:index], str(e.ROOT / e.CLI), "--sealed-exec", suite, *args[index:]]
            return original(args, *pos, **kw)

        with frozen_inputs(refs, scratch), patch.object(subprocess, "Popen", start):
            if not script:
                return int(pytest.main(argv))
            previous = sys.argv
            sys.argv = argv
            try:
                runpy.run_path(argv[0], run_name="__main__")
            except SystemExit as error:
                return int(error.code) if isinstance(error.code, int) else 0
            finally:
                sys.argv = previous
            return 0


def validate(work: Json, raw: Path) -> list[Json]:
    """Freeze collection before execution; preserve terminal outcomes for the exact union."""
    scratch = Path("/var/tmp") / ("exp8545-validation-" + raw.name)
    scratch.mkdir(parents=True, exist_ok=True)
    logs = raw / "validation"
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\npatch=subprocess\ndata_file="
        + str(scratch / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
        + "[report]\nexclude_lines=\n"
    )
    py = str(e.ROOT / ".venv/bin/python")
    prefix = [
        "env",
        "CARNOT_8533_EXTENSION=" + work["extension"]["path"],
        "LLVM_PROFILE_FILE=" + str(raw / "native_logs/rust-%p-%m.profraw"),
        "COVERAGE_RCFILE=" + str(config),
    ]
    receipts, shards, nodes, passed = [], [], [], []
    work["original_authority_by_suite"] = {}
    for i, suite in enumerate(SUITES):
        receipt = child(
            f"collect_{i}",
            [py, "-m", "pytest", *COMMON, "--collect-only", "-q", suite],
            logs,
            deadline=120,
        )
        receipts.append(receipt)
        found = [
            line.strip()
            for line in Path(receipt["stdout_path"]).read_text().splitlines()
            if line.startswith("tests/") and "::" in line
        ]
        shards.append(found)
        nodes.extend(found)
        work["original_authority_by_suite"][suite] = (
            history.custody(e.ROOT, scratch / "v726")
            if "8431" in suite
            else e.original_authority()
            if "8533" in suite
            else "suite-owned private fixtures; original CLI dates unchanged"
        )
    atomic_json(raw / "collection.json", dict(node_ids=nodes, suites=SUITES))
    for i, shard in enumerate(shards):
        parent = scratch / f"suite-{i}"
        parent.mkdir(parents=True, exist_ok=True)
        argv = [
            py,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            str(e.ROOT / e.CLI),
            "--sealed-tests",
            SUITES[i],
            *COMMON,
            "-vv",
            *shard,
            "--basetemp=" + str(parent / "pytest"),
        ]
        receipt = child(f"shard_{i}", prefix + argv, logs, deadline=360)
        receipts.append(receipt)
        outcomes = parse_outcomes(Path(receipt["stdout_path"]))
        passed.extend(outcomes["passed"])
    union = dict(
        passed=bool(nodes)
        and len(nodes) == len(set(nodes))
        and len(passed) == len(nodes)
        and set(nodes) == set(passed),
        collected=nodes,
        passed_node_ids=passed,
        shards=shards,
    )
    atomic_json(raw / "node_union.json", union)
    work["exact_union_receipt"] = union
    receipts.append(dict(name="exact_disjoint_node_union", passed=union["passed"]))
    work["coverage_config"] = str(config)
    work["coverage_scratch"] = str(scratch)
    return receipts


def finish(work: Json, raw: Path) -> list[Json]:
    """Report real subprocess coverage and unchanged lint/type/spec checks once."""
    config = work["coverage_config"]
    py = str(e.ROOT / ".venv/bin/python")
    scratch = Path(work["coverage_scratch"])
    commands = [
        ("coverage_combine", [py, "-m", "coverage", "combine", "--rcfile=" + config, "--keep"]),
        (
            "coverage_report",
            [
                py,
                "-m",
                "coverage",
                "report",
                "--rcfile=" + config,
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
                "--rcfile=" + config,
                "-o",
                str(scratch / "coverage.json"),
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
    receipts = [child(name, argv, raw / "validation", deadline=90) for name, argv in commands]
    report = json.loads((scratch / "coverage.json").read_bytes())
    work["python_coverage"] = report
    atomic_json(raw / "coverage.json", report)
    receipts.append(
        dict(name="owned_python_coverage_100", passed=report["totals"]["percent_covered"] == 100)
    )
    rust = old.rust_coverage(work, raw)
    work["rust_coverage"] = json.loads((raw / "rust_coverage_summary.json").read_bytes())
    return list(receipts + rust)


def reproduce(work: Json, raw: Path) -> list[Json]:
    """Retain the actual old layout failures separately from the corrected test union."""
    nodes = [
        e.legacy.TEST + "::" + name
        for name in [
            "test_offline_build_and_loader_rejections",
            "test_private_cli_publish_replay_and_absence",
        ]
    ]
    base = e.ROOT / "results/raw" / e.NAME / "original_layout_reproduction"
    receipt = child(
        "original_owned_failures",
        [
            "env",
            "CARNOT_8533_EXTENSION=" + work["extension"]["path"],
            str(e.ROOT / ".venv/bin/pytest"),
            *COMMON,
            "-vv",
            *nodes,
            "--basetemp=" + str(base),
        ],
        raw / "reproduction",
        deadline=120,
        expected=1,
    )
    text = Path(receipt["stdout_path"]).read_text()
    receipt["passed"] = (
        receipt["passed"]
        and all("FAILED " + node in text for node in nodes)
        and "private control output must be outside repository" in text
    )
    work["original_failure_reproduction"] = [receipt]
    return [receipt]
