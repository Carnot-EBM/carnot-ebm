"""REQ-VERIFY-8524: reuse original private test shards and byte-bound publication."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import board_cost_boundary_8524 as e
from carnot.reporting import board_operation_runner_8483 as shipped

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Keep exact historical nodes, unchanged private E2E checks and genuine child coverage."""
    with patch.object(shipped, "e", e):
        return list(shipped.manifest(private))


def historical(request: Path) -> int:
    """Original assertions retain their original authority in isolated disk scratch."""
    with patch.object(shipped, "e", e):
        return int(shipped.historical(request))


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Preserve unchanged scanners, atomic publication and the actual downstream consumer."""
    with patch.object(shipped, "e", e):
        return int(shipped.run(root, output, private, control=control))
