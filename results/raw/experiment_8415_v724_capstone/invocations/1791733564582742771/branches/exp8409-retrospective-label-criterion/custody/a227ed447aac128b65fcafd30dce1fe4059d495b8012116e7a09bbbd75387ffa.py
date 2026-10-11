"""REQ-VERIFY-8409: small adapters reuse bounded custody and publication paths."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import retrospective_label_criterion_8409 as e
from carnot.reporting import released_question_identity_runner_8403 as shipped

Json = dict[str, Any]
SCRATCH = Path.home() / ".cache/carnot-exp8409-private"
INPUTS = [
    *shipped.INPUTS,
    "python/carnot/reporting/human_label_custody_8389.py",
    "openspec/change-proposals/v723-service-and-label-protocol.json",
    "docs/research-notes/v724-label-criterion.md",
]


def plan(private: Path) -> list[Json]:
    """The shipped plan freezes genuine child coverage and unchanged private E2E checks."""
    with patch.object(shipped, "e", e):
        return list(shipped.plan(private))


def measure(root: Path, raw: Path, private: Path) -> Json:
    """Shipped preconditions authenticate complete current authority before CSV reduction."""
    with patch.object(shipped, "e", e), patch.object(shipped, "INPUTS", INPUTS):
        return dict(shipped.measure(root, raw, private))


def controls(candidate: Path, raw: Path) -> list[Json]:
    """Shipped controls execute valid, missing, deliberate-error and rehashed failure children."""
    with patch.object(shipped, "e", e):
        return list(shipped.controls(candidate, raw))


def validate(candidate: Path, raw: Path) -> Json:
    """Unchanged typed consumers check the exact candidate before atomic publication."""
    with patch.object(shipped, "e", e):
        return dict(shipped.validate(candidate, raw))


def main(argv: list[str] | None = None) -> int:
    """Only this task identity changes; the proven capped lifecycle stays shared."""
    with (
        patch.object(shipped, "e", e),
        patch.object(shipped, "INPUTS", INPUTS),
        patch.object(shipped, "SCRATCH", SCRATCH),
    ):
        return int(shipped.main(argv))
