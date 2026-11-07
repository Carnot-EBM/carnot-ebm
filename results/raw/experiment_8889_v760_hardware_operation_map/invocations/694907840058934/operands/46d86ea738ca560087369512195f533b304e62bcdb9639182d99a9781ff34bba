"""REQ-REPORT-8865: use bounded checks and the existing atomic primary publisher."""

from pathlib import Path
from typing import Any

from carnot.reporting import v756_capstone_runner as previous
from carnot.reporting import v758_capstone as c
from carnot.reporting.v748_collection_contract import namespace

runner = namespace(previous, "capstone8865_runner")
runner.c = c
runner.checks.c = c
base_check = runner.checks.run_check


def run_check(
    name: str, argv: list[str], raw: Path, deadline: float = 120, scope: str = "owned"
) -> dict[str, Any]:
    """Freeze enough time for large custody readers; progress never renews their budget."""
    if name == "owned_tests":
        argv = [*argv, "-s"]
    return dict(
        base_check(
            name,
            argv,
            raw,
            600 if name in {"private_e2e018_021", "cold_replay"} else deadline,
            scope,
        )
    )


runner.checks.run_check = run_check


def evidence_path(path: str | Path) -> Path:
    """Use this invocation's health receipt so an older milestone cannot supply validation."""
    return (
        c.ROOT / "results/raw" / c.NAME / "development/repository_health.json"
        if str(path) == "/tmp/exp8839-global/receipt.json"
        else Path(path)
    )


runner.Path = evidence_path
main = runner.main
