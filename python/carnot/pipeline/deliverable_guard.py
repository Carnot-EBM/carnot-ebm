"""DeliverableGuard and DocOnlyClassifier — prevent missing deliverables and wasted CI time.

**Why DeliverableGuard exists (RETRO-032, RETRO-033, RETRO-036):**
    Three consecutive milestones (2026.04.33, 2026.04.34) ended with missing result JSON files.
    The headline research questions could NOT be answered because the experiment deliverables
    were absent at retrospective time.  Root cause: no assertion at experiment exit that the
    deliverable file actually exists on disk.  build_result() builds a dict in memory but does
    NOT write it — the caller writes it with AtomicResultWriter or json.dump.  If the write
    was skipped (early return, exception, wrong path), the file was silently absent.

    DeliverableGuard fixes this by requiring the experiment to call assert_written() as the
    FINAL line of main(), which raises FileNotFoundError immediately if the file is absent.
    This turns a silent omission into a loud crash that the conductor can observe and flag.

**Why DocOnlyClassifier exists:**
    The full 3900+ test suite (cargo test + pytest) triggers after every commit, including
    changelog and ops doc updates.  At 60-90s per run × 80+ doc-only commits per milestone,
    this wastes 80-120 minutes per milestone.  DocOnlyClassifier identifies commits that only
    touch markdown/docs/ops/_bmad files so CI can skip the full suite and run only ruff+mypy
    (5-10s total), recovering 80-120 min per milestone.

Spec: REQ-INFRA-033, REQ-INFRA-035,
      SCENARIO-INFRA-041, SCENARIO-INFRA-043
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# DeliverableGuard
# ---------------------------------------------------------------------------

# File extensions and path patterns that count as doc-only changes.
# Evaluated against the full relative path string from repo root.
_DOC_ONLY_PREFIXES = (
    "docs/",
    "ops/",
    "_bmad/",
    "openspec/",
)
_DOC_ONLY_EXTENSIONS = (".md",)


class DeliverableGuard:
    """Assert that a JSON deliverable file was actually written to disk.

    This guard closes the RETRO-032/033/036 hole: build_result() returns a
    dict in memory, but if the caller forgets to write it (or writes to the
    wrong path), the file is silently absent.  Call assert_written() as the
    FINAL line of main() to turn that silent omission into an immediate crash.

    Parameters
    ----------
    path : str
        Path to the expected deliverable JSON file (relative to cwd or absolute).
        The guard checks this path verbatim — no inference, no fallbacks.

    Spec: REQ-INFRA-033, SCENARIO-INFRA-041, SCENARIO-INFRA-042
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def assert_written(self) -> None:
        """Raise FileNotFoundError if the deliverable file is absent.

        Why this must be the FINAL call in main(): by the time we reach the
        end of main(), every write path (success, blocked, partial) should
        have produced the deliverable.  If the file is missing at that point,
        something silently went wrong upstream — an exception was swallowed,
        a write was skipped, or the path was wrong.  A loud crash here is
        safer than a silent absence that looks like a successful run to the
        conductor.

        Raises
        ------
        FileNotFoundError
            With a message naming the missing path and citing the RETRO context.
        """
        if not Path(self._path).exists():
            raise FileNotFoundError(
                f"DeliverableGuard: deliverable file '{self._path}' was NOT written. "
                "This is the RETRO-032/033/036 failure mode: build_result() was called "
                "but the result was never flushed to disk.  Check that AtomicResultWriter.write() "
                "or json.dump() was called with this exact path before assert_written()."
            )

    def assert_written_or_partial(self, partial_path: str) -> None:
        """Pass if either the deliverable or a partial result file exists.

        Used by experiments that emit a partial artifact on timeout so the
        conductor can distinguish a graceful partial run from a silent failure.

        Parameters
        ----------
        partial_path : str
            Path to an alternative partial result file (e.g. a checkpoint JSON).

        Raises
        ------
        FileNotFoundError
            If neither the deliverable nor the partial file exists.
        """
        if Path(self._path).exists() or Path(partial_path).exists():
            return
        raise FileNotFoundError(
            f"DeliverableGuard: neither deliverable '{self._path}' "
            f"nor partial result '{partial_path}' was written. "
            "Ensure the experiment writes at least a partial artifact on early exit."
        )


# ---------------------------------------------------------------------------
# DocOnlyClassifier
# ---------------------------------------------------------------------------


class DocOnlyClassifier:
    """Classify a git diff's changed file list as doc-only or code-mixed.

    A diff is doc-only when every changed file is a markdown file, or lives
    under docs/, ops/, _bmad/, or openspec/ (where all spec files are .md).

    Why this saves 80-120 min per milestone:
        The full test suite takes 60-90s.  A milestone has 80+ changelog/ops
        updates that don't touch Python or Rust code.  Skipping the full suite
        for those commits and running only ruff+mypy (5-10s) recovers the time.

    Spec: REQ-INFRA-035, SCENARIO-INFRA-043
    """

    def is_doc_only_diff(self, changed_files: list[str]) -> bool:
        """Return True if every file in *changed_files* is a doc/ops/spec file.

        A file is doc-only if its path ends with a doc extension (.md) OR
        starts with a known docs prefix (docs/, ops/, _bmad/, openspec/).
        An empty list is conservatively classified as NOT doc-only to avoid
        accidentally skipping tests when no file list is provided.

        Parameters
        ----------
        changed_files : list[str]
            List of relative file paths from the diff (e.g. ``['ops/status.md']``).

        Returns
        -------
        bool
            ``True`` iff every file is a doc-only file.
            ``False`` if any file is a code file, or if the list is empty.
        """
        if not changed_files:
            # Empty diff — conservatively treat as not doc-only (don't skip tests).
            return False

        for fpath in changed_files:
            normalized = fpath.replace(os.sep, "/")
            is_doc = normalized.endswith(_DOC_ONLY_EXTENSIONS) or any(
                normalized.startswith(prefix) for prefix in _DOC_ONLY_PREFIXES
            )
            if not is_doc:
                return False
        return True
