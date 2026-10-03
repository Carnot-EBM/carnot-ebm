"""REQ-REPORT-7977: freeze private checks with identical owned coverage scope."""

from pathlib import Path
from typing import Any

from carnot.reporting import validation_7964 as old

ROOT = old.ROOT
MODULE = "python/carnot/reporting/experiment_7977_v691_hardware_evidence.py"
PLAN = "python/carnot/reporting/validation_7977.py"
PUBLICATION = "python/carnot/reporting/publication_7977.py"
SCRIPT = "scripts/experiments/experiment_7977_v691_hardware_evidence.py"
TEST = "tests/python/test_experiment_7977_v691_hardware_evidence.py"
MEASURED = (MODULE, PLAN, PUBLICATION, SCRIPT)
TESTS = (TEST, *old.TESTS, "tests/python/test_service_cost_7976.py")
LIBRARIES = (
    *old.LIBRARIES,
    *old.MEASURED,
    "python/carnot/reporting/service_cost_7976.py",
    "openspec/capabilities/hardware/spec.md",
)
INCLUDE = "--include=" + ",".join(str(ROOT / name) for name in MEASURED)


def manifest(private: Path, run_date: str) -> list[dict[str, Any]]:
    """Historical fixture dates and failure reasons survive task substitution."""
    commands = old.manifest(private, run_date)
    replacements = {
        old.MODULE: MODULE,
        old.PLAN: PLAN,
        old.PUBLICATION: PUBLICATION,
        old.SCRIPT: SCRIPT,
        old.TEST: TEST,
        "carnot.reporting.experiment_7964_v690_hardware_evidence": "carnot.reporting.experiment_7977_v691_hardware_evidence",
        "experiment_7964_evidence": "experiment_7977_evidence",
    }
    for spec in commands:
        for index, arg in enumerate(spec["argv"]):
            if arg.startswith("--include="):
                arg = INCLUDE
            else:
                for before, after in replacements.items():
                    arg = arg.replace(before, after)
            spec["argv"][index] = arg
        if spec["name"] in {"affected_pytest", "scoped_spec"}:
            spec["argv"].extend((old.TEST, TESTS[-1]))
        if spec["name"] == "full_pytest":
            spec["classification"] = "repository_health"
    alias = private / "success.json"
    alias.unlink()
    alias.symlink_to(private / "success/experiment_7977_evidence.json")
    return commands


def terminal_manifest(private: Path) -> list[dict[str, Any]]:
    """Both validators and cold replay inspect identical terminal candidate bytes."""
    commands = old.terminal_manifest(private)
    for spec in commands:
        spec["argv"] = [arg.replace(old.SCRIPT, SCRIPT) for arg in spec["argv"]]
    return commands
