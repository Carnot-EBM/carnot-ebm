"""Give every pytest invocation its own tmp base, so runs cannot destroy each other.

WHY THIS EXISTS (2026-08-28). pyproject pins `tmp_path_retention_count = 1`. With
pytest's DEFAULT tmp root, every invocation shares one numbered directory
(`/tmp/pytest-of-<user>/pytest-N`), and each new invocation prunes old numbered
dirs down to the retention count. The conductor runs pytest every few minutes,
so any long measurement run had its live tmp base deleted out from under it.
Two full-suite runs (~8 hours each) were destroyed that way, and the damage was
worse than lost time: mid-run deletion turned real failures into setup errors,
which SUPPRESSED the failure count the runs existed to measure.

THE FIX. When no `--basetemp` was given, assign one: a fresh directory unique to
this invocation (timestamp + pid + random suffix) under a per-user parent. With
an explicit basetemp pytest never rotates numbered dirs and never prunes
siblings, so the deletion vector is gone entirely -- no run can reach into
another run's base.

WHAT THIS DOES NOT CHANGE. An explicit `--basetemp` on the command line is
respected untouched. Under pytest-xdist the controller assigns each worker a
subdirectory of its own base before the worker configures, so workers see a
basetemp already set and this module does nothing there.

CLEANUP. Unique bases accumulate where shared rotation used to prune them, so
stale sibling bases are removed best-effort at configure time -- only bases
older than MAX_AGE_S (a day; the full suite runs ~4h), and never the base just
created. Cleanup failures are swallowed: a tmp janitor must never be the reason
a test session dies.
"""

from __future__ import annotations

import getpass
import os
import shutil
import time
import uuid
from pathlib import Path
from tempfile import gettempdir

#: Stale sibling bases older than this are pruned. One day dwarfs the ~4h full suite,
#: so an age-based prune cannot repeat the destroy-a-live-run failure this fixes.
MAX_AGE_S = 24 * 60 * 60


def basetemp_parent() -> Path:
    """The per-user directory all isolated bases live under."""
    try:
        user = getpass.getuser()
    except Exception:  # noqa: BLE001 - no passwd entry in some sandboxes
        user = f"uid{os.getuid()}" if hasattr(os, "getuid") else "unknown"
    return Path(gettempdir()) / f"pytest-carnot-{user}"


def new_isolated_basetemp(*, pid: int | None = None, now: float | None = None) -> Path:
    """A fresh, collision-free base for one pytest invocation. Not yet created."""
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(time.time() if now is None else now))
    return (
        basetemp_parent() / f"{stamp}-{os.getpid() if pid is None else pid}-{uuid.uuid4().hex[:8]}"
    )


def prune_stale_bases(
    parent: Path | None = None,
    *,
    keep: Path | None = None,
    now: float | None = None,
    max_age_s: float = MAX_AGE_S,
) -> list[Path]:
    """Remove sibling bases older than `max_age_s`. Best-effort; returns what was removed.

    Age is judged by directory mtime. A live run touches files continuously, so a
    still-running session's base stays young; only abandoned bases age out.
    """
    root = basetemp_parent() if parent is None else Path(parent)
    cutoff = (time.time() if now is None else now) - max_age_s
    removed: list[Path] = []
    try:
        entries = list(root.iterdir())
    except OSError:
        return removed
    for entry in entries:
        try:
            if keep is not None and entry.resolve() == Path(keep).resolve():
                continue
            if not entry.is_dir() or entry.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry)
        except OSError:
            continue
    return removed


def install_isolated_basetemp(config) -> Path | None:
    """Assign this invocation a private tmp base unless one was given explicitly.

    Returns the assigned path, or None when an explicit --basetemp was respected
    (which includes xdist workers, whose controller assigned them one already).
    """
    try:
        given = config.option.basetemp
    except AttributeError:
        return None
    if given:
        return None
    base = new_isolated_basetemp()
    # pytest creates the base itself but NOT its parent; missing parent is INTERNALERROR.
    base.parent.mkdir(parents=True, exist_ok=True)
    config.option.basetemp = str(base)
    prune_stale_bases(keep=base)
    return base
