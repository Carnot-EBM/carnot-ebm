"""REQ-VERIFY-8579: reuse bounded serial validation with explicit current bindings.

The shipped runner already supervises real children and validates exact bytes.
These bindings change only the current producer, not its original consumers.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v737_contract_methods as e
from carnot.reporting import v737_contract_report as report
from carnot.reporting import v736_contract_runner as shipped

collection = shipped.collection
_MANIFEST = shipped.manifest


@contextmanager
def binding() -> Iterator[None]:
    """Keep old imports intact after each invocation so historical readers keep their authority."""
    with patch.multiple(shipped, e=e, report=report, manifest=manifest):
        yield


def manifest(private: Path) -> list[dict[str, Any]]:
    """Reuse exact original node collection and private serial E2E shards."""
    with binding():
        plan = list(_MANIFEST(private))
        next(r for r in plan if r["name"] == "owned_tests")["deadline"] = 300
        return plan


def main(argv: list[str] | None = None) -> int:
    """The original clock, rejection controls and atomic publisher remain the execution path."""
    with binding():
        return int(shipped.main(argv))
