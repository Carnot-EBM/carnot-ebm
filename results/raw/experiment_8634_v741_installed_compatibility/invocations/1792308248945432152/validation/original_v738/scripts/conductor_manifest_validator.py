"""conductor_manifest_validator.py — dispatch-site exclusion check for the conductor.

WHY THIS EXISTS (REQ-INFRA-046b, Exp 731):
    The existing exclusion manifest is consulted in pick_next_task() via
    _task_is_excluded(), which uses a regex to extract an integer experiment ID
    from the task's "id" field (pattern: "exp<N>-...").  This regex silently
    passes string IDs like "jepa_v15_cascade" and "jepa_v16_cascade" with
    reason="no id parsed", allowing them to be dispatched even when they appear
    in the manifest.  This was the root cause of the 787-minute wall-time gap in
    milestone 2026.04.55 — legacy JEPA cascade tasks re-entered the queue each
    cycle because their string IDs were never matched.

    This module provides a SECOND, complementary validation at the dispatch site
    (the point where the conductor is about to spawn an agent subprocess).  It
    accepts any task_id string, not just "exp<N>-..." patterns, and checks the
    manifest by comparing the raw string against ALL excluded experiment_id values
    (both integer and string entries) after string normalisation.

    The function always returns True (allowed) on any exception so the conductor
    is never blocked by a manifest loading failure — exclusion is a performance
    optimisation, not a safety gate.

HOW TO WIRE THIS IN (see results/manifest_fix_patch.txt for the full diff):
    In scripts/research_conductor.py, inside research_step(), insert the call
    immediately after the three "RESEARCH STEP" logger.info lines (line ~1856)
    and before the dry_run check.

Spec: REQ-INFRA-046b, SCENARIO-INFRA-055b, SCENARIO-INFRA-056b
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

_log = logging.getLogger(__name__)

# Default manifest path — relative to this file's location (scripts/).
_DEFAULT_MANIFEST = Path(__file__).parent / "conductor_exclusion_manifest.json"

# Pattern to extract a bare integer from strings like "exp308-legacy" or "exp308".
_EXP_INT_RE = re.compile(r"(?:exp|experiment)[_-]?(\d+)", re.IGNORECASE)


def _load_excluded_ids(manifest_path: Path) -> set[str]:
    """Read manifest JSON and return a set of normalised excluded ID strings.

    Normalisation rule: every entry's experiment_id is converted to a lowercase
    string.  Integer IDs (e.g. 308) become "308"; string IDs (e.g. "jepa_v15_cascade")
    remain as-is but lowercased.  This lets callers compare task_id strings against
    the set without worrying about type mismatches.

    Returns an empty set when the manifest is missing, empty, or unparseable — the
    conductor must never be blocked by a missing manifest.
    """
    if not manifest_path.exists():
        _log.debug("Manifest not found at %s — treating as empty", manifest_path)
        return set()
    try:
        raw = json.loads(manifest_path.read_text())
        excluded: set[str] = set()
        for entry in raw.get("excluded", []):
            eid = entry.get("experiment_id")
            if eid is not None:
                excluded.add(str(eid).lower())
        return excluded
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to load exclusion manifest (%s) — treating as empty", exc)
        return set()


def _normalise_task_id(task_id: str) -> list[str]:
    """Return the normalised forms of task_id to check against the excluded set.

    Two forms are checked:
    1. The task_id as-is, lowercased (covers string IDs like "jepa_v15_cascade").
    2. The bare integer extracted by _EXP_INT_RE (covers "exp308-legacy" → "308").

    This two-form approach means we catch both integer-ID manifest entries (308)
    and string-ID entries ("jepa_v15_cascade") with a single lookup mechanism.
    """
    forms = [task_id.lower()]
    m = _EXP_INT_RE.search(task_id)
    if m:
        forms.append(m.group(1))
    return forms


def validate_manifest_at_dequeue(
    task_id: str,
    manifest_path: Path | None = None,
) -> bool:
    """Return False (blocked) if task_id is in the exclusion manifest; True otherwise.

    This is the dispatch-site gate (REQ-INFRA-046b).  Call it immediately before
    spawning an agent subprocess for a task.  If it returns False, skip the task
    without spawning an agent — log the skip and move to the next task.

    Parameters
    ----------
    task_id : str
        The task's "id" field from the roadmap YAML, e.g. "exp308-legacy-cleanup"
        or "jepa_v15_cascade".  Both integer-based and string IDs are handled.
    manifest_path : Path | None
        Override the manifest file path.  Defaults to
        scripts/conductor_exclusion_manifest.json.  Pass a custom path in tests.

    Returns
    -------
    bool
        True  → task is allowed to run (not in manifest or manifest unavailable).
        False → task is excluded (found in manifest); do NOT dispatch an agent.

    Side effects
    ------------
    Logs one INFO line: "task_id=<id> allowed=<True/False> reason=<reason>"
    This line is the primary audit trail for the conductor log.
    """
    path = manifest_path or _DEFAULT_MANIFEST
    try:
        excluded_ids = _load_excluded_ids(path)
        forms = _normalise_task_id(task_id)

        for form in forms:
            if form in excluded_ids:
                _log.info(
                    "task_id=%s allowed=False reason=matched_excluded_id(%s)",
                    task_id,
                    form,
                )
                return False

        _log.info("task_id=%s allowed=True reason=not_in_manifest", task_id)
        return True

    except Exception as exc:  # noqa: BLE001
        # Never block the conductor on a validation error — log and allow.
        _log.warning(
            "task_id=%s allowed=True reason=validation_error(%s)",
            task_id,
            exc,
        )
        return True
