"""Adapt the private validation plan while retaining historical failure rules.

Spec ref: REQ-REPORT-7926-V687. A historical fixture keeps its producer date;
the current host run has a separate execution date.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from carnot.reporting import validation_7913 as old

ROOT = old.ROOT
MODULE = "python/carnot/reporting/experiment_7926_v687_hardware_evidence.py"
PLAN = "python/carnot/reporting/validation_7926.py"
RUNNER = "python/carnot/reporting/qualification_7926.py"
SCRIPT = "scripts/experiments/experiment_7926_v687_hardware_evidence.py"
TEST = "tests/python/test_experiment_7926_v687_hardware_evidence.py"
MEASURED = (MODULE, PLAN, RUNNER, SCRIPT)
TESTS = (
    TEST,
    *old.CONSUMERS,
    old.TEST,
    "tests/python/test_experiment_7868_v683_intervention_protocol.py",
)
LIBRARIES = (*old.LIBRARIES, *old.MEASURED)
INCLUDE = "--include=" + ",".join(str(ROOT / name) for name in MEASURED)


def manifest(private: Path, run_date: str) -> list[dict[str, Any]]:
    """Freeze all current commands by adapting the already qualified private plan."""
    commands = deepcopy(old.manifest(private, run_date))
    replacements = {
        old.MODULE: MODULE,
        old.PLAN: PLAN,
        old.SCRIPT: SCRIPT,
        old.TEST: TEST,
        "carnot.reporting.experiment_7913_v686_hardware_evidence": "carnot.reporting.experiment_7926_v687_hardware_evidence",
    }
    for spec in commands:
        argv = []
        for arg in spec["argv"]:
            if arg.startswith("--include="):
                arg = INCLUDE
            else:
                for before, after in replacements.items():
                    arg = arg.replace(before, after)
            argv.append(arg)
        spec["argv"] = argv
        if spec["name"] in {"affected_pytest", "scoped_spec"}:
            argv.extend(TESTS[-2:])
        if spec["name"] in {"ruff_check", "ruff_format", "mypy"}:
            argv.append(RUNNER)
        if spec["name"] in {"e2e_016_fixture", "e2e_016_replay"}:
            argv[argv.index("--date") + 1] = "20260929"
    terminal_shard = private / "terminal.coverage"
    covered_recheck = old.command(
        "terminal_recheck",
        [
            str(ROOT / ".venv/bin/coverage"),
            "run",
            f"--data-file={terminal_shard}",
            INCLUDE,
            str(ROOT / SCRIPT),
            "--terminal-recheck",
            str(private / "success.json"),
        ],
    )
    index = next(i for i, spec in enumerate(commands) if spec["name"] == "coverage_shards")
    commands.insert(index, covered_recheck)
    for spec in commands:
        if spec["name"] in {"coverage_shards", "coverage_combine"}:
            spec["argv"].append(str(terminal_shard))
    index = next(i for i, spec in enumerate(commands) if spec["name"] == "changed_coverage") + 1
    commands.insert(
        index,
        old.command(
            "coverage_json",
            [
                str(ROOT / ".venv/bin/coverage"),
                "json",
                f"--data-file={private / 'combined.coverage'}",
                INCLUDE,
                "-o",
                str(private / "coverage.json"),
            ],
        ),
    )
    commands.insert(
        -1,
        old.command(
            "e2e_016_wrong_date",
            [
                str(ROOT / ".venv/bin/python"),
                "-u",
                "scripts/experiments/experiment_7868_v683_intervention_protocol.py",
                "--date",
                run_date,
                "--fixture-e2e",
                str(private / "wrong-date.json"),
            ],
            expected=1,
            reason="run_date_mismatch",
        ),
    )
    return commands


def terminal_manifest(private: Path) -> list[dict[str, Any]]:
    """Cold reduction and both existing validators inspect the terminal candidate."""
    return [
        *old.terminal_manifest(private),
        old.command(
            "terminal_cold_replay",
            [
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / SCRIPT),
                "--terminal-recheck",
                str(private / "terminal-candidate.json"),
            ],
            classification="terminal_validator",
        ),
    ]
