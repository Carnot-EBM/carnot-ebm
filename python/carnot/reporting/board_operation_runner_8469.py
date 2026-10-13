"""REQ-VERIFY-8469: reuse bounded private validation and atomic publication.

The shipped workflow records real command receipts and genuine child coverage.
Only task bindings change, so validators and historical assertions stay intact.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import board_operation_boundary_8469 as e
from carnot.reporting import board_operation_runner_8454 as old

Json = dict[str, Any]
_MANIFEST = old.manifest


def manifest(private: Path) -> list[Json]:
    """The shipped plan already runs private authority and frozen replay checks."""
    with patch.object(old, "e", e):
        return list(_MANIFEST(private))


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """Restore orchestration bindings after every invocation to protect other producers."""
    with patch.multiple(old, e=e, manifest=manifest):
        return int(old.run(root, output, private, control=control))
