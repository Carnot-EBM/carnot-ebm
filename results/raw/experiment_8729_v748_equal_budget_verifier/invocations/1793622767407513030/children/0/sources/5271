"""Resolve one archived task slug without weakening current numeric IDs.

The archived source artifact used its task name where a number was expected.
These exact bytes are the only evidence allowed to use that spelling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

LEGACY_PATH = (
    Path(__file__).resolve().parents[3]
    / "results/experiment_7810_v679_source_view_qualification.json"
)
LEGACY_HASH = "sha256:59eac9b918b0d4f8a74e78cc3f114208ad9bb68bac28651e1c5d9435a110ba08"
LEGACY_SLUG = "exp7810-source-view-qualification"
LEGACY_MILESTONE = "2026.09.679"


def resolve(path: Path, content_hash: str, artifact: dict[str, Any]) -> int:
    """Accept the sole archived alias after all custody operands agree."""
    operands = (
        ("path", path.resolve(), LEGACY_PATH.resolve()),
        ("hash", content_hash, LEGACY_HASH),
        ("experiment_id", artifact.get("experiment_id"), LEGACY_SLUG),
        ("milestone", artifact.get("milestone"), LEGACY_MILESTONE),
        ("run_date", artifact.get("run_date"), "20260928"),
        ("task_id", artifact.get("task_id", LEGACY_SLUG), LEGACY_SLUG),
    )
    for field, observed, expected in operands:
        if observed != expected:
            raise ValueError(f"legacy_identity:{field}:{observed!s}")
    return 7810
