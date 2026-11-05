"""REQ-REPORT-8852: reuse bounded validation and atomic primary publication."""

from pathlib import Path

from carnot.reporting import v756_capstone_runner as previous
from carnot.reporting import v757_capstone as c
from carnot.reporting.v748_collection_contract import namespace

runner = namespace(previous, "capstone8852_runner")
runner.c = c
runner.checks.c = c


def evidence_path(path: str | Path) -> Path:
    """The reused runner must read this experiment's health receipt, never V756's."""
    return (
        c.ROOT / "results/raw" / c.NAME / "development/repository_health.json"
        if str(path) == "/tmp/exp8839-global/receipt.json"
        else Path(path)
    )


runner.Path = evidence_path
main = runner.main
