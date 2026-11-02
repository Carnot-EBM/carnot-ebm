"""REQ-VERIFY-8571: reuse the qualified serial controller with explicit owned scope."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import complete_native_validation_8559 as legacy
from carnot.reporting import resident_inference_8571 as e

SUITES = [
    e.TEST,
    "tests/python/test_experiment_7891_v685_authority_lifecycle.py",
    "tests/python/test_restricted_decision_audit_8210.py",
]


def validate(work: dict[str, Any], raw: Path) -> list[dict[str, Any]]:
    """Only suite and ownership operands change; serial deadlines and receipts stay shipped."""
    with patch.object(legacy, "e", e), patch.object(legacy, "SUITES", SUITES):
        return list(legacy.validate(work, raw))


def finish(work: dict[str, Any], raw: Path) -> list[dict[str, Any]]:
    """The shipped coverage merger keeps genuine child shards and measures only owned paths."""
    with patch.object(legacy, "e", e):
        return list(legacy.finish(work, raw))
