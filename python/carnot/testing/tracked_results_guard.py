"""Runtime guard for tests that try to write tracked ``results/**`` files."""

from __future__ import annotations

import os
import builtins
import io
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from carnot.experiment_artifacts import (
    ARTIFACT_ROOT_ENV,
    resolve_legacy_results_write_path,
    is_legacy_results_path,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RESULTS_ROOT = _REPO_ROOT / "results"
_installed = False
_legacy_compat_installed = False
_violations: list[dict[str, str]] = []
_legacy_compat_redirects: list[dict[str, str]] = []
_ORIGINAL_BUILTIN_OPEN = builtins.open
_ORIGINAL_IO_OPEN = io.open
_ORIGINAL_OS_OPEN = os.open
_ORIGINAL_OS_RENAME = os.rename
_ORIGINAL_OS_REPLACE = os.replace


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


def is_legacy_compat_installed() -> bool:
    return _legacy_compat_installed


def recorded_legacy_compat_redirects() -> list[dict[str, str]]:
    return list(_legacy_compat_redirects)


def clear_legacy_compat_redirects() -> None:
    _legacy_compat_redirects.clear()


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


def _path_text(path: object) -> str:
    if isinstance(path, bytes):
        return path.decode("utf-8", "surrogateescape")
    try:
        return os.fspath(path)
    except TypeError:
        return str(path)


def _redirect_legacy_write_path(path: object, event: str, *, ensure_parent: bool) -> object:
    if os.environ.get(ARTIFACT_ROOT_ENV) is None or not is_legacy_results_path(path):
        return path
    redirected = resolve_legacy_results_write_path(
        path, ensure_parent=ensure_parent, allow_override=True
    )
    _legacy_compat_redirects.append(
        {
            "event": event,
            "requested": _path_text(path),
            "redirected": os.fspath(redirected),
        }
    )
    return os.fspath(redirected)


def _compat_builtin_open(
    file: object,
    mode: str = "r",
    buffering: int = -1,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
    closefd: bool = True,
    opener: Any | None = None,
) -> Any:
    target = (
        _redirect_legacy_write_path(file, "open", ensure_parent=True)
        if _is_write_intent(mode, None)
        else file
    )
    return _ORIGINAL_BUILTIN_OPEN(
        target,
        mode,
        buffering,
        encoding,
        errors,
        newline,
        closefd,
        opener,
    )


def _compat_io_open(
    file: object,
    mode: str = "r",
    buffering: int = -1,
    encoding: str | None = None,
    errors: str | None = None,
    newline: str | None = None,
    closefd: bool = True,
    opener: Any | None = None,
) -> Any:
    target = (
        _redirect_legacy_write_path(file, "io.open", ensure_parent=True)
        if _is_write_intent(mode, None)
        else file
    )
    return _ORIGINAL_IO_OPEN(
        target,
        mode,
        buffering,
        encoding,
        errors,
        newline,
        closefd,
        opener,
    )


def _compat_os_open(
    path: object, flags: int, mode: int = 0o777, *, dir_fd: int | None = None
) -> int:
    target = (
        _redirect_legacy_write_path(path, "os.open", ensure_parent=True)
        if _is_write_intent(None, flags)
        else path
    )
    if dir_fd is None:
        return _ORIGINAL_OS_OPEN(target, flags, mode)
    return _ORIGINAL_OS_OPEN(target, flags, mode, dir_fd=dir_fd)


def _compat_os_rename(src: object, dst: object, *args: Any, **kwargs: Any) -> None:
    redirected_src = _redirect_legacy_write_path(src, "rename from", ensure_parent=False)
    redirected_dst = _redirect_legacy_write_path(dst, "rename onto", ensure_parent=True)
    return _ORIGINAL_OS_RENAME(redirected_src, redirected_dst, *args, **kwargs)


def _compat_os_replace(src: object, dst: object, *args: Any, **kwargs: Any) -> None:
    redirected_src = _redirect_legacy_write_path(src, "replace from", ensure_parent=False)
    redirected_dst = _redirect_legacy_write_path(dst, "replace onto", ensure_parent=True)
    return _ORIGINAL_OS_REPLACE(redirected_src, redirected_dst, *args, **kwargs)


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


def install_legacy_results_write_compat() -> bool:
    """Redirect relative legacy ``results/...`` writes while pytest uses a temp root.

    The tracked-results audit hook still owns absolute production writes. This
    wrapper only covers old relative literals, which are the collection-time
    compatibility hazard the resolver cannot see from inside the writer.
    """

    global _legacy_compat_installed
    if _legacy_compat_installed:
        return False
    builtins.open = _compat_builtin_open
    io.open = _compat_io_open
    os.open = _compat_os_open
    os.rename = _compat_os_rename
    os.replace = _compat_os_replace
    _legacy_compat_installed = True
    return True
