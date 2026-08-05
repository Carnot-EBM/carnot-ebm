"""Runtime guard for tests that try to write tracked ``results/**`` files."""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RESULTS_ROOT = _REPO_ROOT / "results"
_installed = False
_violations: list[dict[str, str]] = []


class TrackedResultWriteError(RuntimeError):
    """Raised when a test attempts to write tracked evidence under ``results/``."""


def _tracked_result_paths() -> frozenset[str]:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "results"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return frozenset()
    return frozenset(str((_REPO_ROOT / rel).resolve()) for rel in proc.stdout.splitlines() if rel)


_TRACKED_RESULTS = _tracked_result_paths()


def is_installed() -> bool:
    return _installed


def recorded_violations() -> list[dict[str, str]]:
    return list(_violations)


def clear_violations() -> None:
    _violations.clear()


def _is_write_intent(mode: object, flags: object) -> bool:
    if isinstance(mode, str):
        return any(char in mode for char in "wxa+")
    if isinstance(flags, int):
        write_bits = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
        return bool(flags & write_bits)
    return True


def _looks_like_results_path(path: str) -> bool:
    if os.path.isabs(path):
        return "results" in Path(path).parts
    return path == "results" or path.startswith(f"results{os.sep}") or path.startswith("results/")


def _violation_for(path: object) -> str | None:
    if isinstance(path, bytes):
        path = path.decode("utf-8", "surrogateescape")
    if not isinstance(path, str) or not path or not _looks_like_results_path(path):
        return None
    try:
        resolved = Path(path).resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None
    resolved_str = str(resolved)
    if resolved_str in _TRACKED_RESULTS:
        return resolved_str
    return None


def _record_and_raise(resolved: str, event: str) -> None:
    relative = os.path.relpath(resolved, str(_REPO_ROOT))
    _violations.append(
        {
            "path": relative,
            "event": event,
            "stack": "".join(traceback.format_stack(limit=12)),
        }
    )
    raise TrackedResultWriteError(
        f"Test attempted to {event} tracked result evidence {relative!r}. "
        "Experiment tests must redirect artifact output to a temporary root."
    )


def _audit_hook(event: str, args: tuple) -> None:
    if event == "open":
        if len(args) < 3 or not _is_write_intent(args[1], args[2]):
            return
        hit = _violation_for(args[0])
        if hit is not None:
            _record_and_raise(hit, "write")
        return

    if event in ("os.rename", "os.replace", "shutil.move", "shutil.copyfile", "shutil.copy2"):
        if len(args) < 2:
            return
        hit = _violation_for(args[1])
        if hit is not None:
            _record_and_raise(hit, event.rpartition(".")[2] + " onto")
        return

    if event in ("os.remove", "os.unlink", "os.truncate"):
        if not args:
            return
        hit = _violation_for(args[0])
        if hit is not None:
            _record_and_raise(hit, event.rpartition(".")[2])


def install() -> bool:
    global _installed
    if _installed:
        return False
    sys.addaudithook(_audit_hook)
    _installed = True
    return True
