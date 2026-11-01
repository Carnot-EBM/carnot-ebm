"""REQ-VERIFY-8615: reuse bounded execution without rebinding historical readers.

Temporary bindings give this invocation its own sources, tests and cold reader.
Exiting restores every original module so historical authority stays pinned.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v738_contract_runner as previous
from carnot.reporting import v740_contract_methods as e
from carnot.reporting import v740_contract_report as report

Json = dict[str, Any]
_MANIFEST = previous.manifest
collection = previous.collection
shipped = previous.shipped
exact_union = previous.exact_union
date_value = previous.date_value


@contextmanager
def binding() -> Iterator[None]:
    """Restore old producer identities after one current invocation or private test."""
    with patch.multiple(previous, e=e, report=report):
        yield


def manifest(private: Path) -> list[Json]:
    """Keep original disjoint E2E nodes and limit coverage to newly owned statements."""
    with binding():
        return list(_MANIFEST(private))


def main(argv: list[str] | None = None) -> int:
    """Use the shipped4800-second clock, cold controls and exact atomic publication."""
    with binding(), patch.object(previous, "manifest", manifest):
        return int(previous.main(argv))
