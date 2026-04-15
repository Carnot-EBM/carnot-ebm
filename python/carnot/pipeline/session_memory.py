"""Multi-session persistence layer for learned pipeline state.

**Researcher summary:**
    CaseMemory (Exp 135) and ConstraintTemplateLibrary (Exp 343) accumulate
    knowledge about model error patterns while a pipeline is running, but
    reset to empty when the process restarts.  SessionMemory provides a
    disk-backed persistence layer so these learned patterns survive across
    sessions: constraint addition thresholds, false-positive calibration,
    and accumulated case clusters all carry forward into the next run.

**Why this matters for the self-learning loop:**
    The autoresearch goal is for Carnot to improve its own verification
    accuracy over time — not just within a single run, but cumulatively.
    Without persistence, every new process is starting from scratch: no
    memory of which constraint types are noisy for which models, no
    accumulated evidence that carry errors are common for small arithmetic
    models, no calibration of FP thresholds.  SessionMemory closes that gap.

**Storage format:**
    Each model gets its own subdirectory so saves for different models never
    interfere.  The state is stored as a single JSON file
    ``(storage_dir)/(safe_model_id)/session_state.json`` with schema version
    "carnot.session_memory.v1".  This is intentionally plain JSON (not
    safetensors) because the data is metadata counts and dictionaries, not
    numeric tensors — plain JSON is human-inspectable and diff-friendly.

**CI-safety contract:**
    ``load()`` NEVER raises; it returns ``None`` on any failure (missing file,
    corrupt JSON, missing keys).  This means experiments and tests that don't
    have pre-existing sessions degrade gracefully rather than crashing.

Spec: REQ-LEARN-020, REQ-LEARN-021,
      SCENARIO-LEARN-035, SCENARIO-LEARN-036, SCENARIO-LEARN-037
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from carnot.pipeline.adaptive_thresholds import PerModelFPTracker
    from carnot.pipeline.case_memory import CaseMemory
    from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary

_SCHEMA_VERSION = "carnot.session_memory.v1"
_STATE_FILENAME = "session_state.json"


def _escape_model_id(model_id: str) -> str:
    """Replace forward slashes with '__' so model_id is safe as a directory name.

    **Why this is needed:**
        HuggingFace model IDs often look like "google/gemma-3b".  Using that
        directly as a directory name would create a nested directory structure
        ("google/") that may confuse ``list_sessions()`` and can cause
        cross-platform path issues.  Escaping to "__" is reversible enough for
        display purposes and avoids any filesystem complications.

    Examples:
        "gemma-3b"       → "gemma-3b"
        "google/gemma-3b" → "google__gemma-3b"
        "a/b/c"          → "a__b__c"
    """
    return model_id.replace("/", "__")


class SessionMemory:
    """Persist and restore pipeline learning state across process restarts.

    **Why three components are bundled together:**
        CaseMemory, ConstraintTemplateLibrary, and PerModelFPTracker are the
        three stateful learning components in the pipeline.  They are
        intentionally persisted as a unit because they are coupled: the
        template library's observation counts are driven by CaseMemory
        violation records (via CaseMemoryTemplateWiring), and the FP tracker's
        calibration decisions affect which constraints are applied.  Restoring
        them together ensures the pipeline resumes in a consistent state.

    **Filesystem layout:**
        ``storage_dir/
            <safe_model_id>/
                session_state.json   ← single JSON blob for this model``

    Args:
        storage_dir: Directory under which per-model subdirectories are
                     created.  Created automatically on first ``save()``.
        model_id:    Identifier for the model whose state is being tracked
                     (e.g. "google/gemma-3b").  Forward slashes are escaped.

    Spec: REQ-LEARN-020-1, REQ-LEARN-021-1
    """

    def __init__(self, storage_dir: str, model_id: str) -> None:
        self.storage_dir = storage_dir
        self.model_id = model_id

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _state_path(self) -> pathlib.Path:
        """Return the absolute path to the session state file for this model.

        The model_id is sanitised via ``_escape_model_id`` before use as a
        directory component so that HuggingFace-style "org/model" ids work
        correctly on all platforms.

        Spec: REQ-LEARN-021-1
        """
        safe_id = _escape_model_id(self.model_id)
        return pathlib.Path(self.storage_dir) / safe_id / _STATE_FILENAME

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        case_memory: "CaseMemory",
        template_library: "ConstraintTemplateLibrary",
        fp_tracker: "PerModelFPTracker",
    ) -> None:
        """Serialise all three components to disk.

        **Idempotency guarantee:**
            Calling ``save()`` multiple times always produces a single
            ``session_state.json`` file containing the LATEST state — it
            overwrites any previous save rather than appending.  This means
            callers can ``save()`` after every pipeline ``close()`` without
            worrying about stale data accumulating.

        **Atomic write pattern:**
            We write to the final path directly.  For the learning use case
            the data is small enough that a crash mid-write will produce a
            corrupt file that ``load()`` will safely reject (returning None),
            so the added complexity of a temp-file-then-rename pattern is not
            justified here.

        Args:
            case_memory:      CaseMemory instance to serialise.
            template_library: ConstraintTemplateLibrary instance to serialise.
            fp_tracker:       PerModelFPTracker instance to serialise.

        Spec: REQ-LEARN-020-2, REQ-LEARN-020-3
        """
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema": _SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "case_memory": case_memory.to_dict(),
            "template_library": template_library.to_dict(),
            "fp_tracker": fp_tracker.to_dict(),
        }
        path.write_text(json.dumps(payload, indent=2))

    def load(
        self,
    ) -> "tuple[CaseMemory, ConstraintTemplateLibrary, PerModelFPTracker] | None":
        """Deserialise saved state from disk.

        **CI-safety contract:**
            This method NEVER raises.  It returns None for any failure
            condition: missing file, empty file, malformed JSON, missing
            required keys, or unexpected deserialization errors.  This makes
            it safe to call unconditionally at pipeline init without wrapping
            in try/except at the call site.

        Returns:
            A ``(CaseMemory, ConstraintTemplateLibrary, PerModelFPTracker)``
            tuple if state was successfully loaded, or ``None`` otherwise.

        Spec: REQ-LEARN-020-4, REQ-LEARN-020-5
        """
        # Import inside method to avoid circular imports at module load time.
        from carnot.pipeline.adaptive_thresholds import PerModelFPTracker
        from carnot.pipeline.case_memory import CaseMemory
        from carnot.pipeline.constraint_template_library import ConstraintTemplateLibrary

        path = self._state_path()
        try:
            raw = path.read_text()
        except (FileNotFoundError, OSError):
            return None
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
        try:
            case_memory = CaseMemory.from_dict(payload["case_memory"])
            template_library = ConstraintTemplateLibrary.from_dict(payload["template_library"])
            fp_tracker = PerModelFPTracker.from_dict(payload["fp_tracker"])
        except (KeyError, TypeError, Exception):  # noqa: BLE001 — CI-safety: never raise
            return None
        return case_memory, template_library, fp_tracker

    def exists(self) -> bool:
        """Return True if a state file exists on disk for this model.

        Spec: REQ-LEARN-020-6
        """
        return self._state_path().exists()

    def clear(self) -> None:
        """Delete the state file for this model if it exists.

        This is a no-op when the file does not exist — callers do not need
        to check ``exists()`` before calling ``clear()``.

        Spec: REQ-LEARN-020-7
        """
        path = self._state_path()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # Filesystem errors (permissions, etc.) are silently ignored —
            # the goal is best-effort cleanup, not a strict contract.
            pass

    @classmethod
    def list_sessions(cls, storage_dir: str) -> list[str]:
        """Return a sorted list of model identifiers with saved state.

        **What "saved state" means here:**
            Only subdirectories that contain a ``session_state.json`` file
            are included.  Empty or unrelated subdirectories are excluded.

        **Note on model_id escaping:**
            The strings returned are the safe directory-name forms (with
            slashes replaced by ``__``), NOT the original model IDs.  This
            is intentional: the original model_id is not stored in the
            filename, so round-tripping it without ambiguity is not possible
            in the general case.

        Args:
            storage_dir: Root directory to inspect.

        Returns:
            Sorted list of directory names (safe model IDs) that have a
            ``session_state.json``.  Returns ``[]`` if ``storage_dir`` does
            not exist.

        Spec: REQ-LEARN-020-8, SCENARIO-LEARN-037
        """
        root = pathlib.Path(storage_dir)
        if not root.exists():
            return []
        found: list[str] = []
        for candidate in root.iterdir():
            if candidate.is_dir() and (candidate / _STATE_FILENAME).exists():
                found.append(candidate.name)
        return sorted(found)
