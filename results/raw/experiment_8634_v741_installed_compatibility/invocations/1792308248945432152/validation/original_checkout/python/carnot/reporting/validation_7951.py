"""REQ-REPORT-7951: freeze current checks while retaining historical failures.

Each CLI route owns its publication directory. Identical coverage includes
allow unit and real CLI statements to contribute to one measured total.
"""

from pathlib import Path
from typing import Any

from carnot.reporting import validation_7938 as old
from carnot.reporting.validation_7913 import command

ROOT = old.ROOT
MODULE = "python/carnot/reporting/experiment_7951_v689_hardware_evidence.py"
PLAN = "python/carnot/reporting/validation_7951.py"
PUBLICATION = "python/carnot/reporting/publication_7951.py"
SCRIPT = "scripts/experiments/experiment_7951_v689_hardware_evidence.py"
TEST = "tests/python/test_experiment_7951_v689_hardware_evidence.py"
MEASURED = (MODULE, PLAN, PUBLICATION, SCRIPT)
TESTS = (TEST, *old.TESTS)
LIBRARIES = (*old.LIBRARIES, *old.MEASURED)
INCLUDE = "--include=" + ",".join(str(ROOT / name) for name in MEASURED)


def manifest(private: Path, run_date: str) -> list[dict[str, Any]]:
    """Historical fixtures keep their date; current private paths stay separate."""
    commands = old.manifest(private, run_date)
    success = private / "success/experiment_7951_evidence.json"
    success.parent.mkdir(parents=True, exist_ok=True)
    # The reused runner reads this alias only to construct its negative replay.
    (private / "success.json").symlink_to(success)
    replacements = {
        old.MODULE: MODULE,
        old.PLAN: PLAN,
        old.SCRIPT: SCRIPT,
        old.TEST: TEST,
        old.RUNNER: PUBLICATION,
        "carnot.reporting.experiment_7938_v688_hardware_evidence": "carnot.reporting.experiment_7951_v689_hardware_evidence",
    }
    for spec in commands:
        for index, arg in enumerate(spec["argv"]):
            if arg.startswith("--include="):
                arg = INCLUDE
            else:
                for before, after in replacements.items():
                    arg = arg.replace(before, after)
                arg = arg.replace(str(private / "success.json"), str(success))
                arg = arg.replace(
                    str(private / "missing.json"),
                    str(private / "blocked/experiment_7951_evidence.json"),
                )
            spec["argv"][index] = arg
        if spec["name"] in {"affected_pytest", "scoped_spec"}:
            spec["argv"].append(old.TEST)
        if spec["name"] == "full_pytest":
            spec["required_reason"] = None
    blocked = private / "blocked-replay.coverage"
    commands.insert(
        next(i for i, s in enumerate(commands) if s["name"] == "coverage_shards"),
        command(
            "blocked_cold_replay",
            [
                str(ROOT / ".venv/bin/coverage"),
                "run",
                f"--data-file={blocked}",
                INCLUDE,
                str(ROOT / SCRIPT),
                "--date",
                run_date,
                "--root",
                str(private / "absent"),
                "--cold-replay",
                str(private / "blocked/experiment_7951_evidence.json"),
            ],
        ),
    )
    for spec in commands:
        if spec["name"] in {"coverage_shards", "coverage_combine"}:
            spec["argv"].append(str(blocked))
    return commands


def terminal_manifest(private: Path) -> list[dict[str, Any]]:
    """The fresh process reducer and both validators inspect identical bytes."""
    commands = old.terminal_manifest(private)
    for spec in commands:
        spec["argv"] = [arg.replace(old.SCRIPT, SCRIPT) for arg in spec["argv"]]
    return commands
