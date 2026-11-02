"""Freeze the qualified private plan for the workload mapper.

REQ-REPORT-7938-V688. Identical includes let separate CLI routes contribute
to the same owned coverage claim while old failures stay explicit.
"""

from pathlib import Path
from typing import Any

from carnot.reporting import validation_7926 as old

ROOT = old.ROOT
MODULE = "python/carnot/reporting/experiment_7938_v688_hardware_evidence.py"
PLAN = "python/carnot/reporting/validation_7938.py"
RUNNER = old.RUNNER
SCRIPT = "scripts/experiments/experiment_7938_v688_hardware_evidence.py"
TEST = "tests/python/test_experiment_7938_v688_hardware_evidence.py"
MEASURED = (MODULE, PLAN, RUNNER, SCRIPT)
CONSUMERS = (
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_primary_publication_7928.py",
    "tests/python/test_conductor_gates.py",
    "tests/python/test_in_process_doc_reconcile.py",
)
TESTS = (TEST, *old.TESTS, *CONSUMERS)
LIBRARIES = (
    *old.LIBRARIES,
    *old.MEASURED,
    "python/carnot/reporting/primary_publication.py",
    "scripts/conductor_gates.py",
    "scripts/in_process_doc_reconcile.py",
)
INCLUDE = "--include=" + ",".join(str(ROOT / name) for name in MEASURED)


def manifest(private: Path, run_date: str) -> list[dict[str, Any]]:
    """Reuse historical failures and dates, replacing only task-owned paths."""
    commands = old.manifest(private, run_date)
    replacements = {
        old.MODULE: MODULE,
        old.PLAN: PLAN,
        old.SCRIPT: SCRIPT,
        old.TEST: TEST,
        "carnot.reporting.experiment_7926_v687_hardware_evidence": "carnot.reporting.experiment_7938_v688_hardware_evidence",
    }
    for spec in commands:
        for index, arg in enumerate(spec["argv"]):
            if arg.startswith("--include="):
                arg = INCLUDE
            else:
                for before, after in replacements.items():
                    arg = arg.replace(before, after)
            spec["argv"][index] = arg
        if spec["name"] == "unit_coverage":
            spec["argv"].append(old.TEST)
        if spec["name"] in {"affected_pytest", "scoped_spec"}:
            spec["argv"].append(old.TEST)
            spec["argv"].extend(CONSUMERS)
        if spec["name"] == "full_pytest":
            spec["required_reason"] = None
    return commands


def terminal_manifest(private: Path) -> list[dict[str, Any]]:
    """Keep real validators and use the current cold reducer in a fresh process."""
    commands = old.terminal_manifest(private)
    for spec in commands:
        spec["argv"] = [arg.replace(old.SCRIPT, SCRIPT) for arg in spec["argv"]]
    return commands
