"""REQ-VERIFY-8563: bounded original-suite shards and unchanged publication consumers."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import hardware_operation_boundary_8563 as e
from carnot.reporting import hardware_operation_runner_8550 as shipped

Json = dict[str, Any]
_MANIFEST = shipped.manifest


def manifest(private: Path) -> list[Json]:
    """Collect the original union before disjoint serial children, retaining private E2E018/021."""
    with patch.object(shipped, "e", e):
        return list(_MANIFEST(private))


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Small adapters preserve real-child receipts, terminal scanners and atomic replacement."""
    with patch.multiple(shipped, e=e, manifest=manifest):
        return int(shipped.run(root, output, private, control=control))
