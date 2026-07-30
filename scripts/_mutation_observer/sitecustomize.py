"""Record every path this interpreter opens for WRITING, so a test run can prove what it wrote.

WHY THIS FILE EXISTS
--------------------
``test_suite_mutation_check.py`` answers "did the run modify tracked files?" by diffing the dirty
set before and after. That question is answerable from git alone. The NEXT question -- "did *the
run* write this file, or did a concurrent agent author it?" -- is not, and getting it wrong has now
destroyed in-flight work three times (survey doc sections 3.6 and 7b.3, plus the 2026-07-30
``--check`` incident that flagged six authored files as test damage).

The survey concluded attribution was impossible in principle: "the detector cannot tell a test's
write from a human's concurrent edit, because at the file level they are identical events." That is
true at the FILE level and false at the PROCESS level. The documented damage mechanism is
``runpy.run_path`` executing a real experiment script INSIDE the pytest process -- an event the
interpreter can observe directly via ``sys.addaudithook``. So the run can record what it wrote, and
attribution stops being a guess about authorship and becomes a lookup in a log.

WHY A SITECUSTOMIZE AND NOT JUST A CONFTEST HOOK
------------------------------------------------
An audit hook installed in conftest covers the pytest process, which is where ``runpy.run_path``
runs -- the mechanism behind every recorded incident. It does NOT cover an experiment script that
shells out to another Python (the survey notes some do; see its "no subprocess in its script"
annotation). ``sitecustomize`` is imported automatically by every interpreter start, so putting the
observer here and this directory on ``PYTHONPATH`` extends the coverage to Python children.

DELIBERATELY SELF-CONTAINED (the duplication is the point)
----------------------------------------------------------
This runs at interpreter startup, before anything else, in EVERY Python the run spawns. Importing
the (much larger) checker module here would put its import graph -- and any failure in it -- in
front of every subprocess in the suite. So this file duplicates ~20 lines of hook logic instead,
imports only the standard library, and is wrapped so that ANY failure leaves the interpreter
exactly as it found it. A diagnostic must never break the thing it observes.

The log format is deliberately the dumbest thing that works: one absolute path per line, append
mode, no locking. Concurrent writers append whole short lines to the same file, and the reader
takes a set union, so interleaving costs nothing and a truncated final line is dropped by the
reader rather than corrupting a record.
"""

from __future__ import annotations

# The checker sets this to the run's write log. Absent -> this file does nothing at all, which is
# the normal case for every interpreter on this machine that is not inside an observed test run.
_LOG_ENV = "CARNOT_MUTATION_WRITE_LOG"


def _install() -> None:
    import atexit
    import os
    import sys

    log_path = os.environ.get(_LOG_ENV)
    if not log_path:
        return

    # Buffer in memory and flush once at exit. Appending per-open would put a syscall in the path
    # of every file the suite touches, and the suite touches a great many.
    seen: set[str] = set()

    def _hook(event: str, args: tuple) -> None:
        # Only two event families can create or replace a file's contents.
        if event == "open":
            path, mode, _flags = args
            # `mode` is None for fd-based opens, which cannot name a new path anyway.
            if isinstance(path, str) and mode and any(c in mode for c in "wxa+"):
                # Skip the log itself: the flush below opens it, and the observer recording its
                # own bookkeeping is noise in a record whose only value is being trustworthy.
                if path != log_path:
                    seen.add(path)
        elif event in ("os.rename", "os.replace"):
            # The DESTINATION is what ends up modified; args[0] is the source.
            dest = args[1]
            if isinstance(dest, (str, bytes)):
                seen.add(dest.decode() if isinstance(dest, bytes) else dest)

    sys.addaudithook(_hook)

    @atexit.register
    def _flush() -> None:
        if not seen:
            return
        try:
            # The log's directory may not exist yet -- a child can start before anything has
            # written a snapshot. Without this the append fails, the OSError below swallows it,
            # and the child's writes vanish: they come back as UNATTRIBUTED with no indication
            # that an observation was lost. Silent loss is the exact failure this module exists
            # to prevent, so create the directory rather than dropping the record.
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as fh:
                # One write() call, so this process's block lands atomically enough that a
                # concurrent appender cannot interleave INSIDE a path.
                fh.write("".join(f"{p}\n" for p in sorted(seen)))
        except OSError:
            # An unwritable log must not fail the run. The checker treats "no observation" as
            # UNATTRIBUTED, which is the conservative direction: it withholds the destructive
            # advice rather than issuing it wrongly.
            pass


try:
    _install()
except Exception:  # noqa: BLE001 - startup code; never break the interpreter it is loaded into
    pass
