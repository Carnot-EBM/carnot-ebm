"""REQ-VERIFY-8368: reuse bounded real children and the unchanged publication path."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import sys
from typing import Any
from unittest.mock import patch

from carnot.verify import runtime_reader_execution_8353 as base
from carnot.verify import runtime_closure_8368 as q

Json = dict[str, Any]
PY = str(q.ROOT / ".venv/bin/python")
SCRATCH = Path.home() / ".cache/carnot-exp8368-private"
_manifest = base.manifest


def manifest(private: Path) -> list[Json]:
    """The inherited full manifest is scoped to new statements and retained consumers."""
    with patch.object(base, "q", q):
        plan = _manifest(private)
    owned = next(row for row in plan if row["name"] == "owned_coverage")
    owned["argv"].extend(
        ["tests/python/test_typed_runtime_8368.py", "tests/python/test_runtime_reader_8353.py"]
    )
    owned["deadline"] = 600
    for row in plan:
        if row["name"] in {"ruff_check", "ruff_format", "spec_coverage"}:
            row["argv"].extend(
                [
                    "tests/python/test_typed_runtime_8368.py",
                    "tests/python/test_runtime_reader_8353.py",
                ]
            )
        if row["name"] == "strict_mypy":
            row["argv"].append("python/carnot/verify/runtime_reader_8353.py")
    plan.append(
        dict(
            name="private_disk_scratch",
            argv=["/usr/bin/df", "-T", str(private)],
            deadline=10,
            expected=0,
            scope="external_preconditions",
        )
    )
    return plan


def main(argv: list[str] | None = None) -> int:
    """A wall-clock cap owns this invocation; scratch stays on private disk."""
    args = list(sys.argv[1:] if argv is None else argv)
    q.progress("start_no_model_load")
    if "--date" in args:
        index = args.index("--date") + 1
        if index == len(args) or args[index] != "20261010":
            raise SystemExit("date must be 20261010")
        args[index] = "20261009"
    previous = signal.setitimer(signal.ITIMER_REAL, 4800)
    try:
        SCRATCH.mkdir(parents=True, exist_ok=True, mode=0o700)
        SCRATCH.chmod(0o700)
        with (
            patch("tempfile.tempdir", str(SCRATCH)),
            patch.dict(os.environ, {"TMPDIR": str(SCRATCH), "PYTHONUNBUFFERED": "1"}),
            patch.object(base, "q", q),
            patch.object(base, "manifest", manifest),
        ):
            return int(base.main(args))
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous)
