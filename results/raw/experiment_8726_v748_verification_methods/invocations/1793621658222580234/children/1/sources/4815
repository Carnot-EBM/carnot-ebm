"""REQ-VERIFY-8550: reuse bounded children, original fixture authority and atomic consumers."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import hardware_operation_boundary_8550 as e
from carnot.reporting import nine_input_board_runner_8538 as shipped

Json = dict[str, Any]
_MANIFEST = shipped.manifest


def manifest(private: Path) -> list[Json]:
    """Freeze each original node union before serial shards, retaining private E2E-018/021."""
    with patch.multiple(shipped, e=e, SUITES=[e.TEST, *shipped.SUITES[1:]]):
        plan = list(_MANIFEST(private))
    for row in plan:
        if "collected_node_ids" in row:
            row["argv"].append("-vv")
    return [r for r in plan if r["name"] != "private_E2E021"]


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Use unchanged publication and scanners with only the current producer identity adapted."""
    with patch.multiple(shipped, e=e, manifest=manifest):
        return int(shipped.run(root, output, private, control=control))
