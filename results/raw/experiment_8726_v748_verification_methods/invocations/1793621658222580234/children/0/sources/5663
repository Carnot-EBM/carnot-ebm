"""REQ-REPORT-7993: freeze owned checks before the custody measurement.

Required checks qualify only this audit. The broad suite records repository
health once and cannot erase either historical failures or failed owned checks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_7993_v693_capture_custody"
TASK = "exp7993-capture-custody"
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/reporting/capture_custody_7993.py",
    "python/carnot/reporting/validation_7993.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_capture_custody_7993.py", f"tests/python/test_{NAME}.py"]
CONSUMERS = [
    "tests/python/test_experiment_7981_v692_qwen_stream_capture.py",
    "tests/python/test_qwen_stream_capture_7981.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_primary_publication_7928.py",
]


def reference(path: Path) -> Json:
    """Bind current code and private logs to actual bytes rather than names."""
    return dict(path=str(path), sha256=sha256_file(path))


def freeze(scratch: Path) -> Json:
    """Save exact commands, outputs and code hashes before reconstruction."""
    py, cov = str(ROOT / ".venv/bin/python"), str(ROOT / ".venv/bin/coverage")
    include = ",".join(str(ROOT / p) for p in OWNED)
    coverage_file = str(scratch / ".coverage")
    covered = [
        cov,
        "run",
        "--rcfile=/dev/null",
        "--parallel-mode",
        "--data-file=" + coverage_file,
        "--include=" + include,
    ]
    pytest = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    commands: list[Json] = []

    def add(name: str, argv: list[str], required: bool = True, deadline: int = 120) -> None:
        commands.append(dict(name=name, argv=argv, required=required, deadline_s=deadline))

    add(
        "unit_and_consumers",
        [
            *covered,
            "-m",
            "pytest",
            *pytest,
            *TESTS,
            *CONSUMERS,
            "--basetemp=" + str(scratch / "pytest"),
        ],
    )
    cli = str(ROOT / OWNED[-1])
    success = scratch / "success" / (NAME + ".json")
    blocked = scratch / "blocked" / (NAME + ".json")
    for name, args in [
        ("private_cli_success", ["--private-run", "--output", str(success)]),
        ("private_cli_cold", ["--cold-replay", str(success)]),
        (
            "private_cli_blocked",
            ["--private-run", "--root", str(scratch / "missing"), "--output", str(blocked)],
        ),
        ("private_cli_blocked_cold", ["--cold-replay", str(blocked)]),
    ]:
        add(name, [*covered, cli, *args])
    add(
        "e2e015",
        [
            str(ROOT / ".venv/bin/pytest"),
            *pytest,
            "tests/python/test_source_boundary_7852.py",
            "--basetemp=" + str(scratch / "e2e015"),
        ],
    )
    add(
        "coverage_combine",
        [
            cov,
            "combine",
            "--rcfile=/dev/null",
            "--keep",
            "--data-file=" + coverage_file,
            str(scratch),
        ],
    )
    add(
        "coverage_json",
        [
            cov,
            "json",
            "--rcfile=/dev/null",
            "--data-file=" + coverage_file,
            "--include=" + include,
            "-o",
            str(scratch / "coverage.json"),
        ],
    )
    add(
        "changed_statement_coverage",
        [
            cov,
            "report",
            "--rcfile=/dev/null",
            "--data-file=" + coverage_file,
            "--include=" + include,
            "--show-missing",
            "--fail-under=100",
        ],
    )
    add("ruff_check", [str(ROOT / ".venv/bin/ruff"), "check", *OWNED, *TESTS])
    add("ruff_format", [str(ROOT / ".venv/bin/ruff"), "format", "--check", *OWNED, *TESTS])
    add(
        "strict_mypy",
        [
            str(ROOT / ".venv/bin/mypy"),
            "--config-file=/dev/null",
            "--strict",
            "--follow-imports=skip",
            *OWNED,
        ],
    )
    add("spec_coverage", [py, "scripts/check_spec_coverage.py", *TESTS, *CONSUMERS])
    add("repository_health", [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"], False, 300)
    manifest = dict(
        schema="carnot.validation.7993.v1",
        commands=commands,
        owned=OWNED,
        private_scratch=str(scratch),
        coverage_includes=include,
        code_config_hashes=[reference(ROOT / p) for p in OWNED + TESTS],
        artifact_guard_enabled=True,
        execution_date="20261001",
    )
    atomic_json(scratch / "validation_command_manifest.json", manifest)
    return manifest


def execute(manifest: Json, scratch: Path) -> tuple[list[Json], Json]:
    """Run the frozen scope and require a nonempty new-statement denominator."""
    for item in manifest["code_config_hashes"]:
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("frozen_code_changed")
    specs = [
        CommandSpec(
            c["name"],
            tuple(c["argv"]),
            "required" if c["required"] else "repository_health",
            c["deadline_s"],
        )
        for c in manifest["commands"]
    ]
    receipts = run_commands(
        ROOT,
        specs,
        log_dir=scratch / "logs",
        heartbeat_s=10,
        extra_env={"COVERAGE_FILE": str(scratch / "health.coverage")},
    )
    for receipt, command in zip(receipts, manifest["commands"], strict=True):
        receipt["required"] = command["required"]
    counts: Json = {}
    coverage = scratch / "coverage.json"
    if coverage.is_file():
        totals = json.loads(coverage.read_text())["totals"]
        counts = dict(
            statements=totals["num_statements"],
            covered=totals["covered_lines"],
            missing=totals["missing_lines"],
        )
    receipts.append(
        dict(
            name="nonempty_coverage",
            required=True,
            passed=bool(counts) and counts["statements"] > 0 and counts["missing"] == 0,
            command_argv=["coverage", "json"],
            exit_code=0 if counts else 1,
            log_path=str(coverage),
            log_sha256=sha256_file(coverage) if coverage.is_file() else None,
        )
    )
    atomic_json(
        scratch / "validation_receipts.json",
        dict(receipts=receipts, coverage_statement_counts=counts),
    )
    return receipts, counts
