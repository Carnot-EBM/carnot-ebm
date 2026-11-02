"""REQ-VERIFY-8453: reuse bounded publication while adding frozen replay E2E."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

from carnot.reporting import arc_seal_and_support_8453 as e
from carnot.reporting import arc_supervisor_support_runner_8439 as shipped

_MANIFEST, _RUN, _MAIN = shipped.manifest, shipped.run, shipped.main


@contextmanager
def binding() -> Iterator[None]:
    """Owned paths and functions change; every imported validator remains unchanged."""
    with patch.multiple(shipped, e=e, manifest=manifest, run=run):
        yield


def manifest(private: Path) -> list[e.Json]:
    """Real serial unit and child shards cover only the newly owned statements."""
    with binding():
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
                "--basetemp=" + str(private / "e2e021"),
            ],
            expected=0,
            deadline=240,
            scope="owned",
        ),
    )
    return plan


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, supplied: Path | None = None
) -> int:
    """Shipped negative controls and exact scanners precede one atomic terminal write."""
    with binding():
        return int(_RUN(root, output, private, control=control, supplied=supplied))


def main(argv: list[str] | None = None) -> int:
    """The dated direct CLI retains its private-output guard and4800-second deadline."""
    with binding():
        return int(_MAIN(argv))
