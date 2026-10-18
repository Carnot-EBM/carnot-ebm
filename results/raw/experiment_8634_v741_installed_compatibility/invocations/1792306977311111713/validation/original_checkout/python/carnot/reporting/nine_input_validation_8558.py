"""REQ-VERIFY-8558: small adapters retain the shipped original-suite validators."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any
from unittest.mock import patch

from carnot.reporting import nine_input_qualification_8558 as e
from carnot.reporting import nine_input_validation_8545 as previous
from carnot.reporting.v709_execution import child

Json = dict[str, Any]
__all__ = ["previous"]


def historical(suite: str, argv: list[str], *, script: bool = False) -> int:
    """Route original children through the owned wrapper so their coverage remains visible."""
    with patch.object(previous, "e", e):
        return int(previous.historical(suite, argv, script=script))


def validate(work: Json, raw: Path) -> list[Json]:
    """Add owned tests to the unchanged original union before any shard executes."""
    suites = [e.TEST, e.previous.TEST, e.legacy.TEST, *previous.old.SUITES]
    with patch.object(previous, "e", e), patch.object(previous, "SUITES", suites):
        return list(previous.validate(work, raw))


def finish(work: Json, raw: Path) -> list[Json]:
    """Use actual original Rust test bytes so coverage includes real rejection arms."""
    original = Path(work["extension"]["path"]).parent
    saved = raw / "native_binary"
    saved.mkdir(parents=True, exist_ok=True)
    receipts = []
    for binary in sorted(original.glob("nine_input_8533-*")):
        shutil.copyfile(binary, saved / binary.name)
        (saved / binary.name).chmod(0o755)
        receipts.append(
            child(
                "original_rust_tests_" + binary.name,
                [
                    "env",
                    "LLVM_PROFILE_FILE=" + str(raw / "native_logs/rust-%p-%m.profraw"),
                    str(saved / binary.name),
                ],
                raw / "validation",
                deadline=120,
            )
        )
    with patch.object(previous, "e", e):
        receipts.extend(previous.finish(work, raw))
    return receipts
