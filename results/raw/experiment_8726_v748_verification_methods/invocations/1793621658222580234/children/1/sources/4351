"""REQ-VERIFY-8399: retain actual children, immutable receipts and atomic publication."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import board_operation_boundary_8399 as e
from carnot.reporting import hardware_operation_runner_8371 as old
from carnot.reporting import v717_contract_runner as qualified

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Freeze owned coverage and private consumers without repeating the global suite."""
    with patch.object(qualified, "m", e):
        plan = list(qualified.manifest(private))
    plan[0]["deadline"] = 600
    plan[1]["argv"].insert(
        -1, "tests/python/test_hardware_operation_boundary_8371.py::test_typed_map"
    )
    plan[1]["argv"].insert(
        -1, "tests/python/test_v721_contract_methods_8360.py::test_private_e2e018"
    )
    plan[1]["argv"].insert(
        -1, "tests/python/test_kv260_workload_cost_8356.py::test_table_operation_semantics"
    )
    plan[1].update(name="private_E2E018_board_consumers", deadline=600)
    historical = (
        "import json,sys,pytest; from carnot.reporting import board_operation_boundary_8399 as e; "
        "p=e.ROOT/e.TERMINAL; assert e.base.sha256_file(p)==e.TERMINAL_PIN; "
        "refs=json.loads(p.read_bytes())['source_artifact_hashes']; "
        "\nwith e.prior.source_aliases(refs):\n raise SystemExit(pytest.main(sys.argv[1:]))\n"
    )
    plan[1]["argv"] = [
        str(e.ROOT / ".venv/bin/python"),
        "-u",
        "-c",
        historical,
        *plan[1]["argv"][1:],
    ]
    plan.append(
        dict(
            name="private_disk_scratch",
            argv=["/usr/bin/df", "-T", str(private)],
            deadline=10,
            expected=0,
            scope="owned",
        )
    )
    return plan


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Shipped child and publication guards decide readiness from this invocation's receipts."""
    with patch.object(old, "e", e), patch.object(old, "manifest", manifest):
        return int(old.run(root, output, private, control=control))
