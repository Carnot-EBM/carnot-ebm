"""REQ-VERIFY-8510: reuse bounded validation with genuine child statement coverage."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import board_operation_boundary_8510 as e
from carnot.reporting import board_operation_runner_8483 as old

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Original node collection proves each bounded shard retains every historical assertion."""
    with patch.object(old, "e", e):
        return list(old.manifest(private))


def historical(request: Path) -> int:
    """Historical tests retain original authority and dates in an isolated scratch root."""
    with patch.object(old, "e", e):
        return int(old.historical(request))


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Cold replay and unchanged scanners bind actual terminal bytes before publication."""
    with patch.object(old, "e", e):
        return int(old.run(root, output, private, control=control))
