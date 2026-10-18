"""REQ-VERIFY-8581: bind the existing scoped controller to new owned paths.

Only validation configuration changes. Historical readers, original test
identities, numerical tolerances and shared publication checks stay fixed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import causal_accounting_8581 as e
from carnot.reporting import causal_accounting_validation_8568 as previous

Json = dict[str, Any]


def configuration(raw: Path) -> Path:
    """Keep subprocess coverage explicit so no live parent tracer is inspected."""
    with patch.object(previous, "e", e):
        return previous.configuration(raw)


def ready(work: Json) -> bool:
    """Require the exact test union, genuine passing children and measured statements."""
    with patch.object(previous, "e", e):
        return previous.ready(work)


def validate(work: Json) -> None:
    """Reuse bounded serial validation with this producer's explicit file manifest."""
    invoke = previous.child

    def bounded(
        name: str, argv: list[str], logs: Path, *, deadline: float = 180, **options: Any
    ) -> Json:
        # Current revision replay adds real work. This changes only supervision.
        return invoke(name, argv, logs, deadline=900 if name == "owned" else deadline, **options)

    with patch.object(previous, "e", e), patch.object(previous, "child", bounded):
        previous.validate(work)
