"""REQ-VERIFY-8457: bounded inherited validation preserves original CLI dates."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from carnot.reporting import v725_contract_runner as shipped
from carnot.reporting import v727_contract_runner as legacy
from carnot.reporting import v728_contract_methods as e

Json = dict[str, Any]
_MANIFEST = shipped.manifest


def manifest(private: Path) -> list[Json]:
    """Real serial owned coverage and both historical E2E checks stay private."""
    with patch.object(shipped, "e", e):
        plan = list(_MANIFEST(private))
    plan.insert(
        2,
        dict(
            name="private_E2E021",
            argv=[
                str(e.ROOT / ".venv/bin/pytest"),
                "-n",
                "0",
                "-o",
                "addopts=",
                "--no-cov",
                "-q",
                "tests/python/test_restricted_decision_audit_8210.py",
                "--basetemp=" + str(private / "frozen-replay"),
            ],
            expected=0,
            deadline=240,
            scope="owned",
        ),
    )
    next(p for p in plan if p["name"] == "owned_tests")["deadline"] = 1200
    return plan


def main(argv: list[str] | None = None) -> int:
    """The existing dated CLI enforces private output, child deadlines and atomic publication."""
    with patch.object(legacy, "e", e), patch.object(legacy, "manifest", manifest):
        return int(legacy.main(argv))
