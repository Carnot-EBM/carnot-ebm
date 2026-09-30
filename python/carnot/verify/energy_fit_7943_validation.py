"""Freeze checks and reduce diagnostics for SCENARIO-REPORT-7943-TERMINAL."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec
from carnot.verify import energy_fit_7943 as core

PREFIT = {
    "cli_expected_failure",
    "cli_success",
    "cli_blocked",
    "cli_blocked_replay",
    "cli_fixture_replay",
    "cli_terminal_recheck",
}


def freeze(run: ModuleType, private: Path, raw: Path) -> tuple[Path, list[CommandSpec]]:
    """Fix executable paths and expectations before private or natural heads are fitted."""
    path, original = run._original_freeze(private, raw)
    os.environ["CARNOT7943_COVERAGE_FILE"] = str(private / ".coverage")
    commands = []
    prefix: tuple[str, ...] = ()
    for command in original:
        if command.name == "cli_expected_failure":
            prefix = command.argv[: command.argv.index(core.CLI)]
            command = replace(
                command,
                argv=(
                    *prefix,
                    core.CLI,
                    "--date",
                    "20260930",
                    "--fixture",
                    "--publication",
                    str(private / "missing.json"),
                    "--output",
                    str(raw / "routes/negative/experiment_7943_fixture.json"),
                    "--assert-ready",
                ),
            )
        elif command.name == "cli_success":
            command = replace(
                command,
                argv=(
                    *prefix,
                    core.CLI,
                    "--date",
                    "20260930",
                    "--fixture",
                    "--output",
                    str(raw / "routes/success/experiment_7943_fixture.json"),
                    "--assert-ready",
                ),
            )
        elif command.name == "cli_blocked_replay":
            command = replace(
                command,
                argv=(
                    *prefix,
                    core.CLI,
                    "--date",
                    "20260930",
                    "--cold-replay",
                    str(raw / "routes/blocked/experiment_7943_fixture.json"),
                ),
            )
        elif command.name == "mypy":
            command = replace(
                command,
                argv=(
                    str(core.ROOT / ".venv/bin/mypy"),
                    "--strict",
                    "--follow-imports=silent",
                    *core.OWNED,
                ),
            )
        elif command.name.startswith("e2e_016"):
            command = replace(command, argv=(*command.argv[:-1], str(raw / "e2e016/fixture.json")))
        commands.append(command)
    insert = next(i for i, c in enumerate(commands) if c.name == "cli_blocked_replay")
    commands.insert(
        insert,
        CommandSpec(
            "cli_blocked",
            (
                *prefix,
                core.CLI,
                "--date",
                "20260930",
                "--fixture",
                "--publication",
                str(private / "missing.json"),
                "--output",
                str(raw / "routes/blocked/experiment_7943_fixture.json"),
            ),
            "required",
            90,
        ),
    )
    commands.extend(
        CommandSpec(
            name,
            (
                *prefix,
                core.CLI,
                "--date",
                "20260930",
                flag,
                str(raw / "routes/success/experiment_7943_fixture.json"),
            ),
            "required",
            90,
        )
        for name, flag in (
            ("cli_fixture_replay", "--cold-replay"),
            ("cli_terminal_recheck", "--terminal-recheck"),
        )
    )
    commands = [c for c in commands if c.scope != "late"] + [
        c for c in commands if c.scope == "late"
    ]
    value = json.loads(path.read_text())
    value.update(
        affected_files=list(core.OWNED),
        transitive_consumers=list(run.CONSUMERS),
        coverage_includes=core.INCLUDES,
        current_producer={"experiment_id": 7943, "task_id": core.TASK},
        checkpoint_policy="current task and complete dependency hash only",
    )
    value["commands"] = [
        {
            **asdict(c),
            "expected_exit": 2 if c.scope == "expected_failure" else 0,
            "expected_reason": "source evidence blocked"
            if c.scope == "expected_failure"
            else "command must pass",
            "deadline_s": c.timeout_s,
        }
        for c in commands
    ]
    value["dependencies"].update(
        {
            name: sha256_file(core.ROOT / name)
            for name in (
                *run.TESTS,
                *run.CONSUMERS,
                "python/carnot/verify/training_publication_7941.py",
                "python/carnot/verify/training_publication_7941_run.py",
            )
        }
    )
    value["coverage_file"] = str(private / ".coverage")
    atomic_json(path, value)
    return path, commands


def diagnostics(value: dict[str, Any]) -> None:
    """Class balance and calibration bins separate limited information from fitting failure."""
    rows = value["rows"]
    unique = {row["family_id"]: row for row in rows}
    value["class_counts"] = {
        role: dict(Counter(row["label"] for row in unique.values() if row["role"] == role))
        for role in core.old.prior.ROLE_BUDGET
    }
    bins = []
    for index in range(10):
        members = [row for row in rows if min(9, int(row["probability"] * 10)) == index]
        if members:
            bins.append(
                {
                    "bin": index,
                    "count": len(members),
                    "mean_probability": sum(r["probability"] for r in members) / len(members),
                    "observed_frequency": sum(r["label"] for r in members) / len(members),
                }
            )
    value["calibration_bins"] = bins
    value["sample_size_budget"].update(
        unit="family_arm_seed_prediction",
        family_unit="source_family",
        intended_families=640,
        eligible_families=len(unique),
        excluded_families=len(value["sample_size_budget"]["exclusion_rows"]),
    )
    path = Path(value["prediction_rows_path"]).parent / "calibration.svg"
    points = " ".join(
        f"{40 + 320 * b['mean_probability']:.2f},{360 - 320 * b['observed_frequency']:.2f}"
        for b in bins
    )
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="420" height="420"><rect width="420" height="420" fill="white"/><path d="M40 40V360H360M40 360L360 40" fill="none" stroke="gray"/><polyline points="{points}" fill="none" stroke="blue"/><text x="45" y="400">Mean risk vs observed frequency; exposed development</text></svg>'
    )
    value["calibration_plot_path"] = str(path)
    value["calibration_plot_sha256"] = sha256_file(path)
    value["label_mask_checks"] = {
        "known_positive": sum(x == 1 for row in unique.values() for x in row["known_mask"]),
        "known_negative": sum(x == 0 for row in unique.values() for x in row["known_mask"]),
        "unknown": sum(x == -1 for row in unique.values() for x in row["known_mask"]),
        "unknown_loss": "zero through qualified numerical library",
    }
