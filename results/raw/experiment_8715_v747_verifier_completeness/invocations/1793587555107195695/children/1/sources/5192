"""REQ-VERIFY-8577: reuse serial original-suite shards and checked atomic publication."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import board_operation_runner_8454 as publication
from carnot.reporting import hardware_operation_runner_8550 as validation
from carnot.reporting import resident_hardware_boundary_8577 as e

Json = dict[str, Any]


def manifest(private: Path) -> list[Json]:
    """Keep the collected original union and private E2E-018/021 in disjoint serial children."""
    with patch.object(validation, "e", e):
        return list(validation.manifest(private))


def run(root: Path, output: Path, private: Path, *, control: bool = False) -> int:
    """The shipped controller supervises children and checks exact bytes with unchanged gates."""
    with patch.multiple(publication, e=e, manifest=manifest):
        return int(publication.run(root, output, private, control=control))
