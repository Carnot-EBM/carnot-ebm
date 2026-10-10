"""REQ-VERIFY-8385: small adapters retain the unchanged child and publication guards."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import board_operation_evidence_8385 as e
from carnot.reporting import hardware_operation_runner_8371 as old
from carnot.reporting import v717_contract_runner as qualified

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Freeze private coverage and the original board consumers before aggregation."""
    with patch.object(qualified, "m", e):
        plan = list(qualified.manifest(private))
    plan[0]["deadline"] = 600
    plan[1]["argv"].insert(-1, "tests/python/test_restricted_decision_audit_8210.py")
    plan[1]["argv"].insert(
        -1, "tests/python/test_kv260_workload_cost_8356.py::test_table_operation_semantics"
    )
    plan[1]["argv"].insert(
        -1, "tests/python/test_hardware_operation_boundary_8371.py::test_typed_map"
    )
    plan[1]["argv"].insert(
        -1,
        "tests/python/test_hardware_operation_boundary_8371.py::test_five_distinct_historical_repeats",
    )
    plan[1].update(name="private_E2E018_E2E021_original_board_consumers", deadline=600)
    plan.append(
        dict(
            name="repository_health_once",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "tests/python",
                "-q",
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "--basetemp=" + str(private / "global-suite"),
            ],
            expected=0,
            deadline=900,
            scope="global",
        )
    )
    return plan


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Reuse actual child receipts, cold controls, typed findings and atomic publication."""
    with patch.object(old, "e", e), patch.object(old, "manifest", manifest):
        return int(old.run(root, output, private, control=control))
