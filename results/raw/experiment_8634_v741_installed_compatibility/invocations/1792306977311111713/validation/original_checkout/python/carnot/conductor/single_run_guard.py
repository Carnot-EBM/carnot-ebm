"""flock-based single-run guard for experiment scripts.

**Why this module exists:**
    The 2026-04-26 swap-saturation incident and the 2026-04-27 runaway-
    Sonnet incidents shared one root cause: the autoresearch subagent
    (Sonnet) decided to retry an experiment script while a previous
    instance was still running. Without a guard, both instances run to
    completion (or until wall-clock kills the conductor's claude-p
    subprocess). Memory + GPU pressure stacks; one of the duplicate runs
    is wasted compute; on conductor restart the orphans accumulate.

    The flock guard lets the *first* invocation of a given experiment
    proceed and refuses subsequent invocations while the lock is held.
    flock is the right primitive because:

      - It is per-process (not per-thread). Subprocess re-launches from
        a confused agent are exactly the case we want to block.
      - It is automatically released on process exit (including SIGKILL).
        Stale lockfiles are not a recovery problem.
      - It is non-blocking via LOCK_NB. The second caller fails
        immediately rather than queueing.

**Usage pattern (in experiment_template.py setup()):**

    from carnot.conductor import acquire, SingleRunHeld
    try:
        with acquire(f"experiment_{exp_num}"):
            # the rest of the experiment
            ...
    except SingleRunHeld:
        # another instance is running; exit cleanly without writing a
        # blocked artifact (the OTHER instance will produce the artifact)
        sys.exit(0)

**Design choices:**

    Lockfile location: `/tmp/carnot-locks/<name>.lock`. /tmp is
    appropriate because we want the lock to NOT survive reboot — if the
    host reboots, the question "is anything running?" resets to no.

    `name` parameter: caller-supplied. Convention is `experiment_<N>`
    where N is the experiment number, so each experiment script has
    its own lock. Two different experiments running concurrently is
    fine; two of the *same* experiment is what we block.

    Non-blocking: `LOCK_NB` raises BlockingIOError immediately. We
    convert that to `SingleRunHeld` for clearer error semantics.

    No timeout: the guard is binary — held or not. Callers that want a
    "wait up to N seconds" pattern should call this in a retry loop
    themselves.

Spec: REQ-INFRA-072 (single-run-guard from
openspec/change-proposals/conductor-process-isolation.md Exp B).
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


class SingleRunHeld(Exception):
    """Raised by `acquire()` when the named lock is already held by another
    process. The caller should treat this as a soft skip — the OTHER
    holder will produce the experiment artifact, this attempt should
    exit without writing a blocked-artifact (which would confuse the
    conductor's deliverable-existence check).
    """


# /tmp is intentional — the lock state must NOT survive reboot. If the
# host reboots, "is anything running?" resets to the empty answer.
_LOCK_DIR = Path("/tmp/carnot-locks")


@contextmanager
def acquire(name: str) -> Iterator[None]:
    """Acquire a single-run flock for `name`. Raises `SingleRunHeld` if
    another process holds it.

    The lock is released when the contextmanager block exits — including
    on exception or on SIGKILL of the calling process (kernel releases
    flock automatically on process death).

    Parameters
    ----------
    name : str
        The lock name. Convention: `experiment_<N>` for experiment
        scripts, `pytest_<suite>` for pytest invocations the conductor
        wants to deduplicate.

    Raises
    ------
    SingleRunHeld
        If another process holds the lock. Caller decides whether to
        soft-skip, retry-with-delay, or surface the conflict.
    """
    _LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lockfile = _LOCK_DIR / f"{name}.lock"
    fd = os.open(lockfile, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise SingleRunHeld(
                f"Single-run lock '{name}' is already held by another process. Lockfile: {lockfile}"
            ) from exc
        try:
            # Write our PID into the lockfile so debugging shows who's
            # holding it. This is best-effort — flock semantics do not
            # depend on the file's contents.
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
