"""REQ-VERIFY-8511: shipped bounded checks qualify a single terminal V731 primary."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any
from unittest.mock import patch

from carnot.reporting import v730_capstone as old
from carnot.reporting import v731_capstone_evidence as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v731_capstone_report import build

Json = dict[str, Any]
SUITES = old.SUITES


def manifest(private: Path) -> list[Json]:
    """Preserve collected disjoint historical unions while coverage measures only newly owned code."""
    with patch.object(old, "e", e):
        return list(old.manifest(private))


def controls(value: Json, raw: Path) -> list[Json]:
    """Fresh interpreters reject missing operands, real failures and rehashed task-count drift."""
    rows = []
    for name, mutation, expected in [("valid", None, 0), ("rehashed", "task_count", 1)]:
        changed = deepcopy(value)
        if mutation:
            changed[mutation] = 12
            changed["reproducibility_checksum"] = e.checksum(changed)
        path = raw / (name + ".json")
        atomic_json(path, changed)
        rows.append(
            old._CHILD(
                "cold_" + name,
                [sys.executable, "-u", str(e.ROOT / e.CLI), "--cold-replay", str(path)],
                raw / "logs",
                expected=expected,
                deadline=600,
                heartbeat=30,
            )
        )
    for name, args, expected in [
        ("missing", ["--cold-replay", str(raw / "absent.json")], 1),
        ("deliberate_error", ["--deliberate-error"], 7),
    ]:
        rows.append(
            old._CHILD(
                "cold_" + name,
                [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
                raw / "logs",
                expected=expected,
                deadline=60,
                heartbeat=30,
            )
        )
    return rows


def run(
    root: Path, output: Path, private: Path, *, control: bool = False, failed: bool = False
) -> int:
    """The unchanged publisher and consumers keep failed owned checks distinct from upstream blocks."""
    with patch.multiple(
        old.old,
        e=e,
        build=build,
        manifest=old._MANIFEST if control else manifest,
        controls=controls,
        child=old.complete_child,
        terminal_child=old.complete_child,
    ):
        return int(old._RUN(root, output, private, control=control, failed=failed))


def main(argv: list[str] | None = None) -> int:
    """The shipped dated CLI owns disk scratch and a hard cap without loading a model."""
    with patch.multiple(old, e=e, run=run):
        return int(old.main(argv))
