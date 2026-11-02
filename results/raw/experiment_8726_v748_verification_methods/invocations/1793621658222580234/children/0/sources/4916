"""REQ-VERIFY-8343: small adapters reuse the bounded executor and publisher."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import kv260_workload_cost_8343 as e
from carnot.reporting import kv260_arithmetic_8343 as p
from carnot.reporting import kv260_workload_runner_8329 as qualified
from carnot.reporting import v717_contract_runner as checks

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Only exercised authority and arithmetic consumers belong to this check plan."""
    private.mkdir(parents=True, exist_ok=True)
    with patch.object(checks, "m", e):
        plan = list(checks.manifest(private))
    plan[0]["deadline"] = 600
    plan[1]["argv"].insert(-1, "tests/python/test_kv260_workload_cost_8329.py")
    return plan


def main(argv: list[str] | None = None) -> int:
    """Keep the tested child deadlines and atomic rejection/recovery enforcement."""
    with (
        patch.object(qualified, "e", e),
        patch.object(qualified, "primitives", p),
        patch.object(qualified, "manifest", manifest),
    ):
        return int(qualified.main(argv))
