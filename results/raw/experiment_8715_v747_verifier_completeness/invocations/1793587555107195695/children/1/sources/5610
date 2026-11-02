"""REQ-VERIFY-8653: reuse bounded execution without rebinding historical readers.

Temporary bindings give this invocation its own sources, tests and cold reader.
Exiting restores every original module so historical authority stays pinned.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
import sys
import shutil
from coverage import CoverageData
from coverage.exceptions import CoverageException
from carnot.reporting.v709_execution import child
from typing import Any
from unittest.mock import patch

from carnot.reporting import v738_contract_runner as previous
from carnot.reporting import v743_contract_methods as e
from carnot.reporting import v743_contract_report as report

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
                "from pathlib import Path; import sys; from carnot.reporting.v743_contract_runner import verify_collection; raise SystemExit(not verify_collection(Path(sys.argv[1]), sys.argv[2]))",
                str(path),
                e.sha256_file(path),
            ]
    next(r for r in plan if r["name"] == "coverage_combine")["argv"] += [
        str(private / "coverage_copies")
    ]
    return plan


def main(argv: list[str] | None = None) -> int:
    """Use the shipped4800-second clock, cold controls and exact atomic publication."""
    with binding(), patch.multiple(previous, manifest=manifest, execute=execute):
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


def retain_coverage(private: Path, raw: Path) -> Json:
    """Read completed child data immediately and combine copies, preserving original bytes."""
    copies, originals = private / "coverage_copies", raw / "originals"
    copies.mkdir(parents=True, exist_ok=True)
    originals.mkdir(parents=True, exist_ok=True)
    refs, inventories, errors = [], [], []
    for index, source in enumerate(sorted(private.glob(".coverage.*"))):
        try:
            with source.open("rb") as stream:
                if stream.read(16) != b"SQLite format 3\x00":
                    raise ValueError("not_sqlite_child_coverage")
            data = CoverageData(basename=str(source))
            data.read()
            files = sorted(data.measured_files())
            target = originals / str(index)
            shutil.copyfile(source, target)
            shutil.copyfile(target, copies / source.name)
            refs.append(e.reference(target))
            inventories.append(dict(source=str(source), files=files, sha256=e.sha256_file(source)))
        except (OSError, ValueError, CoverageException) as error:
            errors.append(str(error))
    receipt = dict(
        name="original_coverage_retention",
        passed=bool(refs) and not errors,
        original_references=refs,
        source_inventories=inventories,
        errors=errors,
    )
    e.atomic_json(raw / "retention.json", receipt)
    return receipt


def execute(plan: list[Json], logs: Path) -> list[Json]:
    """Capture child evidence at completion, before any combine command can change it."""
    receipts, custody = [], []
    for index, spec in enumerate(plan):
        e.progress("validation", index, len(plan) - index)
        receipts.append(
            child(
                spec["name"],
                spec["argv"],
                logs,
                deadline=spec["deadline"],
                expected=spec["expected"],
                scope=spec["scope"],
            )
        )
        if spec["name"] == "owned_tests":
            combine = next(s for s in plan if s["name"] == "coverage_combine")
            custody.append(
                retain_coverage(Path(combine["argv"][-1]).parent, logs.parent / "coverage_shards")
            )
    return receipts + custody
