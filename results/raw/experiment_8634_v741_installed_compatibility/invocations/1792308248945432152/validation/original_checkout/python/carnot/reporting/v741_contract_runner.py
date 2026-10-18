"""REQ-VERIFY-8627: reuse bounded execution without rebinding historical readers.

Temporary bindings give this invocation its own sources, tests and cold reader.
Exiting restores every original module so historical authority stays pinned.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import v738_contract_runner as previous
from carnot.reporting import v741_contract_methods as e
from carnot.reporting import v741_contract_report as report

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
        plan = list(_MANIFEST(private))
    for spec in plan:
        if spec["name"].startswith("collect_"):
            path = private / "collection" / (spec["name"] + ".receipt.json")
            spec["original_collection_receipt_reference"] = e.reference(path)
            spec["argv"] = [
                sys.executable,
                "-c",
                "from pathlib import Path; import sys; from carnot.reporting.v741_contract_runner import verify_collection; raise SystemExit(not verify_collection(Path(sys.argv[1]), sys.argv[2]))",
                str(path),
                e.sha256_file(path),
            ]
    return plan


def main(argv: list[str] | None = None) -> int:
    """Use the shipped4800-second clock, cold controls and exact atomic publication."""
    with binding(), patch.object(previous, "manifest", manifest):
        return int(previous.main(argv))


def verify_collection(path: Path, digest: str) -> bool:
    """Authenticate the single original collection and its logs without rerunning pytest."""
    try:
        receipt = e.read(dict(path=str(path), sha256=digest))
        return bool(
            receipt["passed"]
            and receipt["normal_exit"]
            and all(
                e.checked(dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"]))
                for stream in ["stdout", "stderr"]
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
