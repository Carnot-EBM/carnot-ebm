"""Reduce authenticated ARC supervisor receipts for REQ-REPORT-7887."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import sha256_file

MODULE = "python/carnot/reporting/arc_supervisor_v684_delta.py"
CLI = "scripts/experiments/experiment_7887_v684_arc_supervisor_delta.py"
TESTS = (
    "tests/python/test_arc_supervisor_delta_7874.py",
    "tests/python/test_arc_supervisor_delta_7887.py",
)


def build_manifest(root: Path, private: Path, cutoff_ns: int) -> list[dict[str, Any]]:
    """Freeze actual files, child arguments and deadlines before reading outcomes."""

    paths = (*TESTS, MODULE, CLI)
    missing = [path for path in paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    py = str(root / ".venv/bin/python")
    pytest = str(root / ".venv/bin/pytest")
    coverage = str(root / ".venv/bin/coverage")
    include = "*/carnot/reporting/arc_supervisor_v684_delta.py,*/scripts/experiments/experiment_7887_v684_arc_supervisor_delta.py"
    common = ["-n", "0", "-o", "addopts=", "--no-cov", "-q"]
    result: list[dict[str, Any]] = []

    def add(
        name: str, argv: list[str], deadline: int = 120, classification: str = "required"
    ) -> None:
        result.append(
            {"name": name, "argv": argv, "deadline_s": deadline, "classification": classification}
        )

    add("affected_pytest", [pytest, *TESTS, *common, f"--basetemp={private / 'test-temp'}"])
    add(
        "unit_coverage",
        [
            coverage,
            "run",
            f"--data-file={private / 'coverage.unit'}",
            f"--include={include}",
            "-m",
            "pytest",
            *TESTS,
            *common,
            f"--basetemp={private / 'cov-temp'}",
        ],
    )
    add(
        "cli_success_coverage",
        [
            coverage,
            "run",
            f"--data-file={private / 'coverage.cli-success'}",
            f"--include={include}",
            CLI,
            "--reduce-ledger",
            str(private / "empty"),
            "--cutoff-ns",
            str(cutoff_ns),
            "--output",
            str(private / "cli-empty.json"),
        ],
    )
    add(
        "cli_failure_coverage",
        [
            coverage,
            "run",
            f"--data-file={private / 'coverage.cli-failure'}",
            f"--include={include}",
            CLI,
            "--reduce-ledger",
            str(private / "empty"),
            "--cutoff-ns",
            str(cutoff_ns),
        ],
        30,
        "diagnostic",
    )
    add(
        "coverage_combine",
        [
            coverage,
            "combine",
            f"--data-file={private / 'coverage.combined'}",
            str(private / "coverage.unit"),
            str(private / "coverage.cli-success"),
        ],
        30,
    )
    add(
        "coverage_report",
        [
            coverage,
            "report",
            f"--data-file={private / 'coverage.combined'}",
            f"--include={include}",
            "--show-missing",
            "--fail-under=100",
        ],
        30,
    )
    add("ruff_check", [str(root / ".venv/bin/ruff"), "check", MODULE, CLI, *TESTS])
    add("ruff_format", [str(root / ".venv/bin/ruff"), "format", "--check", MODULE, CLI, *TESTS])
    add("mypy", [str(root / ".venv/bin/mypy"), "--strict", MODULE, CLI], 180)
    add("scoped_spec", [py, "scripts/check_spec_coverage.py", *TESTS], 60)
    return result


def aggregate(delta: dict[str, Any]) -> dict[str, Any]:
    """Recount eligible primitive rows and leave small samples without arm claims."""

    games: dict[str, dict[str, Any]] = {}
    for row in delta["outcome_rows"]:
        if row.get("status") not in {"completed", "censored"}:
            continue
        game = str(row.get("game"))
        arm = str(row.get("arm"))
        game_cell = games.setdefault(game, {"eligible": 0, "arms": {}})
        game_cell["eligible"] += 1
        arm_cell = game_cell["arms"].setdefault(
            arm,
            {
                "firings": 0,
                "helped": 0,
                "regressions": 0,
                "resolved_by_levelup": 0,
                "actions_to_levelup": [],
                "stagnations_unredirected": [],
            },
        )
        arm_cell["firings"] += int(row.get("fired") is True)
        arm_cell["helped"] += int(row.get("helped") is True)
        arm_cell["regressions"] += int(row.get("helped") is False)
        arm_cell["resolved_by_levelup"] += int(row.get("resolved_by_levelup") is True)
        if isinstance(row.get("actions_to_levelup"), int):
            arm_cell["actions_to_levelup"].append(row["actions_to_levelup"])
        if isinstance(row.get("stagnations_unredirected"), int):
            arm_cell["stagnations_unredirected"].append(row["stagnations_unredirected"])
    delta["per_game_results"] = games
    delta["no_new_outcomes"] = delta["new_live_outcome_count"] == 0
    delta["recommendation_rows"] = []
    return delta


def cold_replay(candidate: Path) -> list[str]:
    """Rehash sealed logs and compare claimed firings with eligible primitive rows."""

    doc = json.loads(candidate.read_text(encoding="utf-8"))
    errors = [
        str(receipt["name"])
        for receipt in doc.get("validation_receipts", [])
        if not Path(receipt["log_path"]).is_file()
        or sha256_file(Path(receipt["log_path"])) != receipt["log_sha256"]
    ]
    firings = sum(
        row.get("fired") is True
        for row in doc.get("outcome_rows", [])
        if row.get("status") in {"completed", "censored"}
    )
    if firings != doc.get("firings"):
        errors.append("primitive_firing_count")
    return errors
