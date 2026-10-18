"""REQ-VERIFY-8485: retain real historical tests and bounded private execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v730_contract_methods as e
from carnot.reporting import v729_contract_runner as shipped

Json = dict[str, Any]
_MANIFEST = shipped.manifest


def manifest(private: Path) -> list[Json]:
    """Use unchanged collected shards so every original assertion and CLI date survives."""
    with patch.object(shipped, "e", e):
        return list(_MANIFEST(private))


def main(argv: list[str] | None = None) -> int:
    """Reuse bounded execution and atomic publication; only invocation identity changes."""
    with patch.object(shipped, "e", e), patch.object(shipped, "manifest", manifest):
        return int(shipped.main(argv))
