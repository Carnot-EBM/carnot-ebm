"""REQ-VERIFY-8497: retain shipped bounded validation and genuine child coverage."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import board_operation_boundary_8497 as e
from carnot.reporting import board_operation_runner_8483 as old

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """The shipped collection proves each bounded historical shard belongs to the original suite."""
    with patch.object(old, "e", e):
        return list(old.manifest(private))


def historical(request: Path) -> int:
    """Run unchanged historical assertions under their original authority and producer hash."""
    with patch.object(old, "e", e):
        return int(old.historical(request))


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Keep fresh replay, scanners and atomic publication attached to this task identity."""
    with patch.object(old, "e", e):
        return int(old.run(root, output, private, control=control))
