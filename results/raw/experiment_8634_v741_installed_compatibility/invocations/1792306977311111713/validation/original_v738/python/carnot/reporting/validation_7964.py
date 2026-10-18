"""REQ-REPORT-7964: freeze private commands and identical coverage includes.

Historical fixture dates and required negative reasons stay unchanged. Each
current route has its own publication directory to preserve primary uniqueness.
"""

from pathlib import Path
from typing import Any

from carnot.reporting import validation_7951 as old

ROOT = old.ROOT
MODULE = "python/carnot/reporting/experiment_7964_v690_hardware_evidence.py"
PLAN = "python/carnot/reporting/validation_7964.py"
PUBLICATION = "python/carnot/reporting/publication_7964.py"
SCRIPT = "scripts/experiments/experiment_7964_v690_hardware_evidence.py"
TEST = "tests/python/test_experiment_7964_v690_hardware_evidence.py"
MEASURED = (MODULE, PLAN, PUBLICATION, SCRIPT)
TESTS = (TEST, *old.TESTS)
LIBRARIES = (*old.LIBRARIES, *old.MEASURED)
INCLUDE = "--include=" + ",".join(str(ROOT / name) for name in MEASURED)


def manifest(private: Path, run_date: str) -> list[dict[str, Any]]:
    """Keep the qualified historical plan while binding all current commands."""
    commands = old.manifest(private, run_date)
    replacements = {
        old.MODULE: MODULE,
        old.PLAN: PLAN,
        old.PUBLICATION: PUBLICATION,
        old.SCRIPT: SCRIPT,
        old.TEST: TEST,
        "carnot.reporting.experiment_7951_v689_hardware_evidence": "carnot.reporting.experiment_7964_v690_hardware_evidence",
    }
    for spec in commands:
        for index, arg in enumerate(spec["argv"]):
            if arg.startswith("--include="):
                arg = INCLUDE
            else:
                for before, after in replacements.items():
                    arg = arg.replace(before, after)
                arg = arg.replace("experiment_7951_evidence", "experiment_7964_evidence")
            spec["argv"][index] = arg
        if spec["name"] in {"affected_pytest", "scoped_spec"}:
            spec["argv"].append(old.TEST)
    alias = private / "success.json"
    alias.unlink()
    alias.symlink_to(private / "success/experiment_7964_evidence.json")
    return commands


def terminal_manifest(private: Path) -> list[dict[str, Any]]:
    """Both validators and cold reduction inspect the same terminal candidate."""
    commands = old.terminal_manifest(private)
    for spec in commands:
        spec["argv"] = [arg.replace(old.SCRIPT, SCRIPT) for arg in spec["argv"]]
    return commands
