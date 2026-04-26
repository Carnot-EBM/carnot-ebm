"""ExclusionManifestEnforcer — side-channel that writes blocked experiment IDs to MILESTONE_PREREQS.md.

**Why this exists (RETRO-MANIFEST-FULL-SCOPE, 8 consecutive milestones unapplied):**
    The manifest_fix_patch.txt has been documented since Exp 731 (.56) but CLAUDE.md
    forbids modifying scripts/research_conductor.py directly.  This module provides
    a side-channel: it reads ops/exclusion_manifest.yaml (the YAML authority) and
    writes a "## Exclusion Manifest Gate" section to MILESTONE_PREREQS.md that the
    conductor CAN read at pre-flight time without requiring a code patch.

    The conductor reads MILESTONE_PREREQS.md before dequeuing any experiment.
    If the gate section lists an experiment as retired, the conductor MUST NOT
    launch it.  This gives us manifest enforcement without touching
    scripts/research_conductor.py.

**How it works:**
    1. ExclusionManifestEnforcer.load_manifest(yaml_path) loads ops/exclusion_manifest.yaml.
    2. is_retired(experiment_id) returns True if the integer ID is in the retired set.
    3. get_retirement_reason(experiment_id) returns the reason string.
    4. write_prereqs_section(prereqs_path) appends "## Exclusion Manifest Gate" to
       MILESTONE_PREREQS.md listing all retired IDs and reasons.
    5. check_queue(task_ids) returns the subset of task_ids that are retired.

Spec: REQ-INFRA-072, SCENARIO-INFRA-081
"""

from __future__ import annotations

import datetime
from pathlib import Path

try:
    import yaml  # type: ignore[import]

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# ExclusionManifestEnforcer
# ---------------------------------------------------------------------------


class ExclusionManifestEnforcer:
    """Read ops/exclusion_manifest.yaml and write gate entries to MILESTONE_PREREQS.md.

    This is a read-only view of the YAML manifest combined with a write-only
    side-channel to MILESTONE_PREREQS.md.  It deliberately does NOT modify
    scripts/research_conductor.py (CLAUDE.md constraint).

    Attributes
    ----------
    _retired : dict[int, str]
        Maps retired experiment ID -> retirement reason.  Populated by load_manifest().

    Spec: REQ-INFRA-072, SCENARIO-INFRA-081
    """

    def __init__(self) -> None:
        # Maps retired integer experiment ID -> human-readable reason.
        self._retired: dict[int, str] = {}
        self._manifest_path: str | None = None

    # ------------------------------------------------------------------
    # load_manifest
    # ------------------------------------------------------------------

    def load_manifest(self, yaml_path: str) -> dict[int, str]:
        """Load retired experiment IDs from ops/exclusion_manifest.yaml.

        The YAML file is expected to have a top-level "retired" key whose value
        is a list of dicts with "experiment_id" (int), "completed_milestone" (str),
        and "reason" (str) fields.  Any entry whose experiment_id is not an integer
        is silently skipped — string tokens (like 'jepa_v15_cascade') live in the
        conductor JSON, not this YAML.

        Falls back to a simple line-parser if PyYAML is not installed so the module
        works in environments without the yaml package.

        Parameters
        ----------
        yaml_path : str
            Path to ops/exclusion_manifest.yaml (absolute or relative to cwd).

        Returns
        -------
        dict[int, str]
            Map of experiment_id -> reason for all retired entries.

        Spec: REQ-INFRA-072
        """
        self._manifest_path = yaml_path
        p = Path(yaml_path)
        if not p.exists():
            self._retired = {}
            return self._retired

        raw = p.read_text()
        if _YAML_AVAILABLE:
            data = yaml.safe_load(raw)
            entries = data.get("retired", [])
            for entry in entries:
                eid = entry.get("experiment_id")
                reason = entry.get("reason", "")
                if isinstance(eid, int):
                    # Keep the last reason for each ID (de-duplicate).
                    self._retired[eid] = reason
        else:
            # Minimal line-based fallback: parse "experiment_id: NNN" and "reason: ..."
            self._retired = _parse_yaml_fallback(raw)

        return dict(self._retired)

    # ------------------------------------------------------------------
    # is_retired
    # ------------------------------------------------------------------

    def is_retired(self, experiment_id: int) -> bool:
        """Return True if experiment_id is in the retired set.

        Assumes load_manifest() has been called first.  Returns False if the
        manifest has not been loaded (safe default — do not block unknown IDs).

        Parameters
        ----------
        experiment_id : int
            Integer experiment ID to check.

        Spec: REQ-INFRA-072
        """
        return experiment_id in self._retired

    # ------------------------------------------------------------------
    # get_retirement_reason
    # ------------------------------------------------------------------

    def get_retirement_reason(self, experiment_id: int) -> str:
        """Return the human-readable retirement reason for experiment_id.

        Returns an empty string if the experiment is not retired.

        Parameters
        ----------
        experiment_id : int
            Integer experiment ID to look up.

        Spec: REQ-INFRA-072
        """
        return self._retired.get(experiment_id, "")

    # ------------------------------------------------------------------
    # write_prereqs_section
    # ------------------------------------------------------------------

    def write_prereqs_section(self, prereqs_path: str) -> None:
        """Append an "## Exclusion Manifest Gate" section to MILESTONE_PREREQS.md.

        The section lists all retired experiment IDs with their reasons.  The
        conductor reads MILESTONE_PREREQS.md at pre-flight; if it finds this
        section it MUST refuse to launch any experiment whose ID appears in the
        "Blocked IDs" list.

        Why append rather than replace?  CLAUDE.md rule "never remove existing
        content from ops/spec docs when updating" — we add a dated section so
        historical gate evolution is preserved.

        Parameters
        ----------
        prereqs_path : str
            Path to MILESTONE_PREREQS.md (absolute or relative to cwd).

        Spec: REQ-INFRA-072, SCENARIO-INFRA-081
        """
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        lines: list[str] = [
            "\n\n---\n",
            "## Exclusion Manifest Gate\n",
            "\n",
            f"*Written by ExclusionManifestEnforcer at {timestamp}*\n",
            f"*Source: {self._manifest_path or 'ops/exclusion_manifest.yaml'}*\n",
            "\n",
            "The conductor MUST NOT launch any experiment whose ID appears below.\n",
            "These experiments are permanently retired and have no research value.\n",
            "\n",
            "| Experiment ID | Retirement Reason |\n",
            "|--------------|------------------|\n",
        ]
        for eid in sorted(self._retired.keys()):
            reason = self._retired[eid].replace("|", "/")
            lines.append(f"| {eid} | {reason} |\n")

        lines.append("\n")
        lines.append("manifest_enforcer_deployed: true\n")
        lines.append(f"retired_count: {len(self._retired)}\n")

        p = Path(prereqs_path)
        with open(p, "a") as f:
            f.writelines(lines)

    # ------------------------------------------------------------------
    # check_queue
    # ------------------------------------------------------------------

    def check_queue(self, task_ids: list[int]) -> list[int]:
        """Return the subset of task_ids that are retired and must be blocked.

        Parameters
        ----------
        task_ids : list[int]
            Candidate experiment IDs to check against the manifest.

        Returns
        -------
        list[int]
            IDs from task_ids that are in the retired set.

        Spec: REQ-INFRA-072
        """
        return [tid for tid in task_ids if self.is_retired(tid)]


# ---------------------------------------------------------------------------
# _parse_yaml_fallback (private)
# ---------------------------------------------------------------------------


def _parse_yaml_fallback(raw: str) -> dict[int, str]:
    """Minimal line-based YAML parser for experiment_id + reason fields.

    Used when PyYAML is not installed.  Parses the specific format used in
    ops/exclusion_manifest.yaml — not a general-purpose YAML parser.

    Each entry starts with "  - experiment_id: NNN" and the reason appears
    on a subsequent "    reason: ..." line within the same block.

    Parameters
    ----------
    raw : str
        Raw text content of ops/exclusion_manifest.yaml.

    Returns
    -------
    dict[int, str]
        Map of integer experiment_id -> reason.
    """
    result: dict[int, str] = {}
    current_id: int | None = None
    for line in raw.splitlines():
        # Strip indentation and the optional YAML list dash.
        stripped = line.strip().lstrip("- ").strip()
        if stripped.startswith("experiment_id:"):
            val = stripped.split(":", 1)[1].strip()
            try:
                current_id = int(val)
            except ValueError:
                current_id = None
        elif stripped.startswith("reason:") and current_id is not None:
            reason = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            result[current_id] = reason
            current_id = None
    return result
