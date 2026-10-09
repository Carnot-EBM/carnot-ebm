"""REQ-VERIFY-8356: retain bounded children, typed findings and atomic publication."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import kv260_workload_cost_8356 as e
from carnot.reporting import kv260_workload_runner_8329 as qualified
from carnot.reporting import v717_contract_runner as checks

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Freeze only owned coverage and exercised private consumers before measurement."""
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(checks, "m", e):
        plan = checks.manifest(private)
    plan[0]["deadline"] = 600
    for name in [
        "tests/python/test_kv260_workload_cost_8329.py",
        "tests/python/test_hard_exit_learning_qualification_8206.py",
    ]:
        plan[1]["argv"].insert(-1, name)
    plan[1]["name"] = "private_E2E018_E2E020_cost_consumers"
    plan[1]["deadline"] = 600
    for spec in plan:
        if spec["name"] in {"ruff_check", "ruff_format", "spec_coverage"}:
            spec["argv"].append("tests/python/test_kv260_workload_cost_8329.py")
    return list(plan)


def main(argv: list[str] | None = None) -> int:
    """Adapt task identity while reusing checked failure publication and CLI guards."""
    with (
        patch.object(qualified, "e", e),
        patch.object(qualified, "primitives", e.p),
        patch.object(qualified, "manifest", manifest),
    ):
        return int(qualified.main(argv))
