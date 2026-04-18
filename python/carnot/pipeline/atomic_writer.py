"""AtomicResultWriter — POSIX-safe JSON result file writer.

**Why atomic write? (RETRO-030 root cause)**
    Exp 446 completed without error (exit code 0) but produced no result file.
    The root cause: an exception was raised inside the ``with open(path) as f:``
    block after ``open()`` succeeded but before ``json.dump()`` finished.  The
    half-written file was left on disk (or no file at all, depending on timing),
    and the experiment watchdog reported success because it only checked the
    process exit code — not whether the deliverable file existed.

    This module resolves RETRO-030 by using the atomic write pattern:
    1. Serialise the data with ``json.dumps()`` (can raise; no file touched yet).
    2. Write the serialised bytes to ``<path>.tmp`` (isolated temporary file).
    3. Call ``os.rename(<path>.tmp, <path>)`` — atomic on POSIX; the kernel
       guarantees the destination either points to the old file or the new file.
       There is no window where the destination is absent or partially written.

**Why os.rename() is atomic on POSIX:**
    The POSIX ``rename(2)`` syscall is specified to be atomic: the destination
    path is replaced in a single operation.  Readers observing the path see
    either the old content or the new content — never a partial write, never
    an absent file (when a prior file existed).  This is reliable on Linux
    ext4/xfs/tmpfs and most local filesystems.  It is NOT reliable on NFS mounts
    without proper lock support, but Carnot experiments run on local SSD storage.

Spec: REQ-INFRA-031, REQ-INFRA-032,
      SCENARIO-INFRA-039, SCENARIO-INFRA-040
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class AtomicResultWriter:
    """Write a JSON result file atomically, preventing partial-write silent failures.

    Usage::

        writer = AtomicResultWriter("results/experiment_452.json")
        writer.write(artifact)           # write-to-tmp then rename
        assert writer.verify_exists()   # raises RuntimeError if False

    Parameters
    ----------
    path : str
        Absolute or repo-relative path for the final JSON result file.
        The parent directory is created automatically if absent.

    Spec: REQ-INFRA-031, SCENARIO-INFRA-039, SCENARIO-INFRA-040
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._final = Path(path)
        self._tmp = Path(path + ".tmp")

    # ------------------------------------------------------------------
    # write()
    # ------------------------------------------------------------------

    def write(self, data: dict) -> None:
        """Serialise *data* to JSON and write atomically to ``self.path``.

        Algorithm
        ---------
        1. ``json.dumps(data)`` — can raise; no file is touched if it does.
        2. Create parent directories (``mkdir -p``).
        3. Write serialised bytes to ``<path>.tmp``.
        4. ``os.rename(<path>.tmp, <path>)`` — atomic on POSIX.

        If any step raises, the final path is either absent (no prior file)
        or still contains the previous complete JSON document.  The ``.tmp``
        file may be left behind on a rename failure; it is safe to delete.

        Parameters
        ----------
        data : dict
            JSON-serialisable dict to write.

        Raises
        ------
        Any exception from ``json.dumps()``, file I/O, or ``os.rename()``.
        The caller must handle these and treat an absent result file as an
        experiment failure (REQ-INFRA-032).
        """
        # Step 1: serialise first — if this raises, nothing on disk changes.
        serialised = json.dumps(data, indent=2)

        # Step 2: ensure parent directory exists.
        self._final.parent.mkdir(parents=True, exist_ok=True)

        # Step 3: write to .tmp (not the final path).
        self._tmp.write_text(serialised, encoding="utf-8")

        # Step 4: atomic rename — replaces final path in one kernel operation.
        os.rename(str(self._tmp), str(self._final))

    # ------------------------------------------------------------------
    # verify_exists()
    # ------------------------------------------------------------------

    def verify_exists(self) -> bool:
        """Return ``True`` iff the result file exists at ``self.path``.

        Call this immediately after ``write()`` to catch any silent failure
        (e.g. filesystem issue that prevents the rename from persisting).
        Raise ``RuntimeError`` from the caller when this returns ``False``
        so the conductor can detect and re-queue the experiment.

        Spec: REQ-INFRA-032, SCENARIO-INFRA-039
        """
        return self._final.exists()
