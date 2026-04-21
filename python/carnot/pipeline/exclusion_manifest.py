"""Conductor exclusion manifest — prevents completed/stuck experiments from re-running.

**Why this exists (RETRO-056):**
    The same five experiments (308, 260, 309, 425, 410) appeared in the slowest-5 list
    for seven consecutive milestones (.37 through .43), consuming approximately 385 minutes
    per milestone (cumulative waste: 2,485 minutes = 41.4 hours).  These experiments are
    in checkpoint-failure state or have been superseded by better infrastructure, so re-running
    them wastes GPU time and inflates per-milestone wall-clock with no research benefit.

    The exclusion manifest is a simple JSON file the conductor consults at session start.
    Any experiment_id listed there is skipped without spawning an agent or consuming GPU time.

**How it works:**
    1. The conductor calls ``ExclusionManifest.load()`` at session start.
    2. Before picking a task, it calls ``is_excluded(experiment_id)`` — O(1) via set lookup.
    3. Excluded experiments are logged and skipped.  No agent is spawned, no GPU is allocated.
    4. New experiments can be added to the manifest via ``ExclusionManifest.add(entry)``.

Spec: REQ-INFRA-070, REQ-INFRA-071,
      SCENARIO-INFRA-075, SCENARIO-INFRA-076, SCENARIO-INFRA-077
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from carnot.pipeline.atomic_writer import AtomicResultWriter

# Default path, relative to the repo root, where the manifest JSON lives.
DEFAULT_MANIFEST_PATH = "scripts/conductor_exclusion_manifest.json"


@dataclass
class ExclusionEntry:
    """One experiment excluded from conductor re-entry.

    Fields:
        experiment_id: The integer ID of the experiment (e.g. 308).
        completed_milestone: The milestone string when the exclusion was decided
            (e.g. "2026.04.37").  This is NOT necessarily when the experiment
            ran successfully — it may be the milestone when we decided the
            experiment is stuck/superseded and should never run again.
        reason: Human-readable explanation of WHY this experiment is excluded.
            Should be specific enough to explain to a future researcher why
            we are deliberately skipping this experiment.
    """

    experiment_id: int
    completed_milestone: str
    reason: str


class ExclusionManifest:
    """Load, query, and update the conductor exclusion manifest.

    The manifest is stored as a JSON file with the schema::

        {
          "excluded": [
            {"experiment_id": 308, "completed_milestone": "...", "reason": "..."},
            ...
          ]
        }

    The file is optional — if it does not exist, ``load()`` returns an empty list
    and the conductor treats no experiments as excluded (safe default).

    Parameters
    ----------
    manifest_path : str
        Path to the JSON manifest file.  May be absolute or relative to cwd.
        Use ``DEFAULT_MANIFEST_PATH`` for the standard conductor location.
    """

    def __init__(self, manifest_path: str) -> None:
        self._path = Path(manifest_path)
        # Cache the excluded set after load so is_excluded() is O(1).
        self._excluded_ids: Optional[set[int]] = None

    def load(self) -> list[ExclusionEntry]:
        """Read the manifest JSON and return a list of ExclusionEntry objects.

        Returns an empty list if the file does not exist or is empty — the caller
        must not treat a missing manifest as an error.  It simply means no
        experiments have been excluded yet.

        Raises
        ------
        json.JSONDecodeError
            If the file exists but contains invalid JSON.
        KeyError
            If the file exists and is valid JSON but lacks the "excluded" key.
        """
        if not self._path.exists():
            self._excluded_ids = set()
            return []
        raw = json.loads(self._path.read_text())
        entries = [
            ExclusionEntry(
                experiment_id=item["experiment_id"],
                completed_milestone=item["completed_milestone"],
                reason=item["reason"],
            )
            for item in raw["excluded"]
        ]
        self._excluded_ids = {e.experiment_id for e in entries}
        return entries

    def is_excluded(self, experiment_id: int) -> bool:
        """Return True if experiment_id appears in the loaded manifest.

        Call ``load()`` before calling this method.  If ``load()`` has not been
        called yet, this method calls it automatically (lazy init) so the conductor
        does not need to call load() explicitly before checking each experiment.

        Parameters
        ----------
        experiment_id : int
            The experiment ID to check (e.g. 308).
        """
        if self._excluded_ids is None:
            self.load()
        assert self._excluded_ids is not None  # mypy narrowing
        return experiment_id in self._excluded_ids

    def save(self, entries: list[ExclusionEntry]) -> None:
        """Write the list of entries to the manifest file atomically.

        Uses AtomicResultWriter (write-to-tmp then rename) to prevent partial-write
        corruption — the same pattern used for all experiment result files since
        RETRO-030.

        Parameters
        ----------
        entries : list[ExclusionEntry]
            The complete list of entries to persist.  The existing file is replaced.
        """
        payload = {"excluded": [asdict(e) for e in entries]}
        writer = AtomicResultWriter(str(self._path))
        writer.write(payload)
        # Invalidate the in-memory cache so the next is_excluded() call reloads.
        self._excluded_ids = None

    def add(self, entry: ExclusionEntry) -> None:
        """Append one entry to the manifest, loading the current list first.

        This is a read-modify-write: loads existing entries, appends the new entry,
        and saves the combined list atomically.  Duplicate experiment_ids are allowed
        at the data level but the conductor should avoid adding the same ID twice.

        Parameters
        ----------
        entry : ExclusionEntry
            The new exclusion entry to append.
        """
        existing = self.load()
        existing.append(entry)
        self.save(existing)


def load_manifest(path: str) -> "ExclusionManifest | None":
    """Load the exclusion manifest from path; return None if file missing (non-blocking).

    This module-level wrapper is the preferred entry-point for experiment scripts
    that need to check exclusions without managing an ExclusionManifest instance directly.
    A missing manifest is normal (no experiments excluded yet) — returning None instead
    of raising lets callers use is_excluded() safely without try/except.

    Spec: REQ-INFRA-093

    Parameters
    ----------
    path : str
        Path to the conductor_exclusion_manifest.json file.

    Returns
    -------
    ExclusionManifest | None
        Loaded manifest instance, or None if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        em = ExclusionManifest(path)
        em.load()
        return em
    except Exception:  # noqa: BLE001
        return None


def is_excluded(manifest: "ExclusionManifest | None", exp_id: int) -> bool:
    """Return True if exp_id is excluded in manifest; False if manifest is None.

    Designed for use alongside load_manifest() so callers never need to
    null-check before every lookup.  When the manifest is None (file missing),
    we err on the side of allowing the experiment to run.

    Spec: REQ-INFRA-093

    Parameters
    ----------
    manifest : ExclusionManifest | None
        Loaded manifest, or None if load_manifest() returned None.
    exp_id : int
        Experiment ID to check.
    """
    if manifest is None:
        return False
    return manifest.is_excluded(exp_id)


def build_manifest_check_result(
    manifest: "ExclusionManifest | None",
    checked_ids: list[int],
) -> dict:
    """Return a structured summary of which experiment IDs are excluded.

    Used by Exp 666 to emit a verifiable artifact that confirms the manifest
    was consulted and which IDs were found excluded.  The ``all_clear`` field
    is True when ALL checked_ids are excluded (i.e., no chronic experiments
    can re-enter), making it easy for the conductor to assert the wire-in is working.

    Spec: REQ-INFRA-094

    Parameters
    ----------
    manifest : ExclusionManifest | None
        Loaded manifest; if None, all checked IDs report as not excluded.
    checked_ids : list[int]
        The experiment IDs the caller wants to verify.

    Returns
    -------
    dict with keys:
        manifest_loaded (bool): True if manifest is not None.
        excluded_ids (list[int]): subset of checked_ids that are excluded.
        checked_ids (list[int]): echoed back for traceability.
        all_clear (bool): True when every checked_id is excluded.
    """
    excluded = [eid for eid in checked_ids if is_excluded(manifest, eid)]
    return {
        "manifest_loaded": manifest is not None,
        "excluded_ids": excluded,
        "checked_ids": checked_ids,
        "all_clear": len(excluded) == len(checked_ids),
    }


def build_default_manifest() -> list[ExclusionEntry]:
    """Return the five experiments excluded as of RETRO-056.

    These five experiments consumed ~385 minutes per milestone for seven consecutive
    milestones (.37 through .43).  They are in checkpoint-failure state or have been
    superseded by newer infrastructure (e.g. BatchedInferenceRunner replaces
    sequential inference loops).  Excluding them frees ~385 minutes/milestone.

    Returns
    -------
    list[ExclusionEntry]
        The five default exclusions.  This list is also written to
        ``scripts/conductor_exclusion_manifest.json`` by build_default_manifest's
        callers and by Exp 575.
    """
    return [
        ExclusionEntry(
            experiment_id=308,
            completed_milestone="2026.04.37",
            reason="slowest-5 seven consecutive milestones, legacy checkpoint-failure state",
        ),
        ExclusionEntry(
            experiment_id=260,
            completed_milestone="2026.04.37",
            reason="slowest-5 seven consecutive milestones, sequential inference loop",
        ),
        ExclusionEntry(
            experiment_id=309,
            completed_milestone="2026.04.37",
            reason="slowest-5 seven consecutive milestones, checkpoint-failure state",
        ),
        ExclusionEntry(
            experiment_id=425,
            completed_milestone="2026.04.37",
            reason="slowest-5 seven consecutive milestones, ExperimentTimeoutWatchdog already implemented",
        ),
        ExclusionEntry(
            experiment_id=410,
            completed_milestone="2026.04.37",
            reason="slowest-5 seven consecutive milestones, 1000 sequential inference calls, BatchedInferenceRunner migration target",
        ),
    ]
