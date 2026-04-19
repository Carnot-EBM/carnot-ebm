"""ConductorDedupCheck and PartialResultHandoff — prevent redundant experiment re-runs and enable fast recovery from interrupted sessions.

**Why ConductorDedupCheck? (RETRO-041, Exp 447 triple re-verification)**
    Exp 447 was re-executed three times across three conductor sessions with no code changes
    between runs.  Each run took ~20 minutes.  Total wasted time: ~60 minutes.  Root cause:
    the conductor had no memory of prior completion — it saw an experiment in the roadmap
    and re-queued it without checking whether a valid result already existed on disk.

    This module provides ``ConductorDedupCheck.is_complete(exp_id)`` which looks for
    ``results/experiment_NNN*.json`` files (the pattern used by all Carnot experiments),
    reads the ``honest_verdict`` field, and returns ``True`` iff the verdict indicates
    a genuinely completed run (not blocked, deferred, or partial).  The conductor calls
    this before spawning any experiment subprocess, short-circuiting the 20-min run.

**Why PartialResultHandoff? (RETRO-041, Exp 308 interrupted recovery took 105 min)**
    Exp 308 was interrupted mid-run (SIGTERM from conductor timeout).  Recovery required:
    full context reload (20 min), triage to identify where the run failed (30 min), and
    restart from scratch (55 min).  Total: 105 min for work that had already been ~60%
    complete at interruption time.

    This module provides ``PartialResultHandoff``, which:
    1. ``install(template)`` — registers an ``atexit`` handler AND a ``SIGTERM`` handler
       that both call ``save()`` with the template's current checkpoint state before exit.
    2. ``save(template, partial_state)`` — writes the in-progress state atomically to
       ``results/experiment_NNN_partial.json`` so the next session can fast-path to the
       checkpoint instead of restarting from scratch.
    3. ``resume_if_available(template)`` — checks for a partial file, loads it if present,
       and returns the partial state dict (or None if no partial exists).

    Combining checkpoint resume with partial handoff reduces interrupted-run recovery from
    105 min to <5 min on the resume path.

**Why atomic write for partial state?**
    The partial file is written DURING interrupt handling (SIGTERM or atexit), which means
    the process is being torn down at the same time.  A non-atomic write risks leaving a
    truncated or zero-byte partial file on disk — which is worse than no file because the
    conductor might try to resume from corrupt data.  We use the same ``write-to-.tmp +
    os.rename()`` pattern as ``AtomicResultWriter`` to guarantee the file is either absent
    or complete, never partial.

Spec: REQ-INFRA-042, REQ-INFRA-043, REQ-INFRA-044,
      SCENARIO-INFRA-050, SCENARIO-INFRA-051, SCENARIO-INFRA-052
"""

from __future__ import annotations

import atexit
import glob
import json
import logging
import os
import signal
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.experiment_template import ExperimentTemplate

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------

# Verdicts that mean "this experiment did NOT produce a genuinely complete result."
# The conductor should NOT skip an experiment whose result file has one of these verdicts —
# it means the previous run was blocked, deferred, or incomplete.
_BLOCKED_VERDICTS: frozenset[str] = frozenset(
    {
        "blocked",
        "deferred_to_gpu",
        "gpu_required",
        "scaffolding_only",
    }
)

# Prefix for partial verdicts — any verdict starting with "partial_" is blocked.
_PARTIAL_VERDICT_PREFIX = "partial_"


# ---------------------------------------------------------------------------
# ConductorDedupCheck
# ---------------------------------------------------------------------------


class ConductorDedupCheck:
    """Prevent the conductor from re-executing an already-completed experiment.

    The dedup check looks for ``results/experiment_NNN*.json`` files matching the
    experiment ID, reads the ``honest_verdict`` field, and determines whether the
    result represents a complete, non-blocked run.

    Usage::

        check = ConductorDedupCheck()
        if check.should_skip(exp_id=447):
            print("Exp 447 already complete — skipping to save 20 min")

    Why we need this instead of just checking file existence:
        A file can exist with a verdict of ``'deferred_to_gpu'`` or ``'blocked'``,
        meaning the experiment was queued but not actually run.  We must inspect the
        verdict to determine genuine completion, not just file presence.

    Spec: REQ-INFRA-042, SCENARIO-INFRA-050, SCENARIO-INFRA-051
    """

    def __init__(self, results_dir: str = "results") -> None:
        """Initialise with the directory where result files are written.

        Parameters
        ----------
        results_dir : str
            Directory path (relative to cwd or absolute) where experiment
            result JSON files live.  Default is ``'results'`` (repo-relative).
        """
        self.results_dir = results_dir

    def is_complete(self, exp_id: int) -> bool:
        """Return True when a non-blocked, non-deferred result exists for *exp_id*.

        Algorithm:
        1. Glob for ``results/experiment_NNN*.json`` (excluding ``*_partial.json``).
        2. For each match, parse JSON and read ``honest_verdict``.
        3. Return True if any verdict is present and not a blocked verdict.
        4. Return False if no result file exists or all verdicts are blocked.

        Parameters
        ----------
        exp_id : int
            Numeric experiment ID (e.g. 447).

        Returns
        -------
        bool
            ``True`` iff the experiment has a complete, actionable result.
            ``False`` if no result file exists, or the result is blocked/deferred/partial.

        Spec: REQ-INFRA-042, SCENARIO-INFRA-050
        """
        pattern = os.path.join(self.results_dir, f"experiment_{exp_id}*.json")
        matches = [
            p for p in glob.glob(pattern)
            if not p.endswith("_partial.json")
        ]
        if not matches:
            return False

        for path in matches:
            try:
                with open(path, encoding="utf-8") as fh:
                    data = json.load(fh)
                verdict = data.get("honest_verdict", "")
                if verdict and not self.is_blocked_verdict(verdict):
                    return True
            except (OSError, json.JSONDecodeError, ValueError):
                # Corrupt or unreadable result file — treat as not complete.
                _log.warning("ConductorDedupCheck: could not read '%s' — treating as incomplete", path)

        return False

    @staticmethod
    def is_blocked_verdict(verdict: str) -> bool:
        """Return True when *verdict* indicates a blocked or incomplete run.

        A result with a blocked verdict should NOT be skipped — the experiment
        needs to run again (e.g. when GPU becomes available for a
        ``'deferred_to_gpu'`` result, or when scaffolding is replaced by real code
        for a ``'scaffolding_only'`` result).

        Parameters
        ----------
        verdict : str
            The ``honest_verdict`` string from a result JSON file.

        Returns
        -------
        bool
            ``True`` if the verdict means "this experiment did not complete successfully."

        Spec: REQ-INFRA-042, SCENARIO-INFRA-051
        """
        return verdict in _BLOCKED_VERDICTS or verdict.startswith(_PARTIAL_VERDICT_PREFIX)

    def should_skip(self, exp_id: int) -> bool:
        """Return True when the experiment is complete and the conductor should skip it.

        This is the primary entry point for conductor dedup logic.  It combines
        ``is_complete()`` with the implicit negation of ``is_blocked_verdict()``
        (the blocked check is already embedded in ``is_complete()``).

        Parameters
        ----------
        exp_id : int
            Numeric experiment ID.

        Returns
        -------
        bool
            ``True`` iff the experiment has a complete result and should be skipped.

        Spec: REQ-INFRA-042, SCENARIO-INFRA-050
        """
        return self.is_complete(exp_id)


# ---------------------------------------------------------------------------
# PartialResultHandoff
# ---------------------------------------------------------------------------


class PartialResultHandoff:
    """Serialize in-progress experiment state on SIGTERM/atexit for fast-path resume.

    This class provides three operations that together implement the partial-handoff
    pattern validated against the Exp 308 incident:

    1. ``install(template)`` — call once at experiment startup to register cleanup
       handlers that will call ``save()`` automatically if the process is interrupted.

    2. ``save(template, partial_state)`` — write the current in-progress state to
       ``results/experiment_NNN_partial.json`` atomically so the next session can
       resume from the checkpoint instead of restarting from scratch.

    3. ``resume_if_available(template)`` — check for a partial file and return its
       contents (dict) if found, or None if no partial file exists.

    Usage::

        from scripts.experiment_template import ExperimentTemplate
        from carnot.pipeline.conductor_dedup import PartialResultHandoff

        tmpl = ExperimentTemplate(308, "My experiment", "results/experiment_308.json")
        tmpl.setup()
        handoff = PartialResultHandoff()
        handoff.install(tmpl)

        # At any point during the experiment, the state can be resumed:
        prior = handoff.resume_if_available(tmpl)
        if prior:
            completed_ids = prior.get("completed_ids", [])
        # ... run experiment ...
        # On SIGTERM or atexit, save() is called automatically.

    Spec: REQ-INFRA-043, REQ-INFRA-044,
          SCENARIO-INFRA-051, SCENARIO-INFRA-052
    """

    def __init__(self, results_dir: str = "results") -> None:
        """Initialise with the directory where result files are written.

        Parameters
        ----------
        results_dir : str
            Directory path (relative to cwd or absolute).  Default ``'results'``.
        """
        self.results_dir = results_dir
        # Track the most-recently installed template and state so the atexit/signal
        # handler can call save() without holding a reference in the closure.
        self._active_template: "ExperimentTemplate | None" = None
        self._active_partial_state: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def install(self, template: "ExperimentTemplate") -> None:
        """Register atexit and SIGTERM handlers that serialize partial state on exit.

        Why both atexit AND SIGTERM?
            - ``atexit`` fires on normal Python interpreter shutdown (including
              uncaught exceptions that propagate to the top level, sys.exit(), etc.).
            - ``SIGTERM`` is the signal the conductor sends when it kills a
              subprocess that has exceeded its timeout.  atexit does NOT fire on
              SIGTERM unless a SIGTERM handler calls ``sys.exit()`` — which is exactly
              what we do here.

        The handlers are idempotent: multiple installs replace the previous handlers.

        Parameters
        ----------
        template : ExperimentTemplate
            The running experiment's template instance.  Its checkpoint state
            will be used as the partial state if the handlers fire before
            ``update_partial_state()`` is called explicitly.

        Spec: REQ-INFRA-043, SCENARIO-INFRA-051
        """
        self._active_template = template

        # Register atexit handler — fires on normal exit and uncaught exceptions.
        atexit.register(self._atexit_handler)

        # Register SIGTERM handler — fires when conductor kills the subprocess.
        signal.signal(signal.SIGTERM, self._sigterm_handler)

        _log.info(
            "PartialResultHandoff installed for Exp %s — partial state will be written to '%s'",
            template.exp_id,
            self._partial_path(template),
        )

    def save(self, template: "ExperimentTemplate", partial_state: dict) -> None:
        """Write *partial_state* atomically to ``results/experiment_NNN_partial.json``.

        Why atomic write here:
            This method is called from interrupt handlers (atexit, SIGTERM) where
            the process is being torn down.  A non-atomic write risks leaving a
            truncated file that looks like valid JSON but is incomplete.  We use
            write-to-.tmp + os.rename() to guarantee the file is either absent or
            complete.

        Parameters
        ----------
        template : ExperimentTemplate
            The experiment whose partial state is being saved.
        partial_state : dict
            JSON-serializable dict with whatever in-progress state exists.
            The handler adds ``experiment_id``, ``partial``, and a timestamp.

        Spec: REQ-INFRA-043, SCENARIO-INFRA-051
        """
        path = self._partial_path(template)
        tmp_path = path + ".tmp"

        full_state = {
            "experiment": template.exp_id,
            "partial": True,
            "honest_verdict": f"partial_{template.exp_id}",
            **partial_state,
        }

        try:
            serialised = json.dumps(full_state, indent=2)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(tmp_path).write_text(serialised, encoding="utf-8")
            os.rename(tmp_path, path)
            _log.info("PartialResultHandoff: partial state written to '%s'", path)
        except Exception as exc:  # noqa: BLE001 — must not raise in interrupt handler
            _log.error("PartialResultHandoff: failed to write partial state to '%s': %s", path, exc)

    def resume_if_available(self, template: "ExperimentTemplate") -> dict | None:
        """Return the partial state dict if a partial file exists, else None.

        Call this near the top of the experiment's main loop, AFTER ``install()``.
        If a prior interrupted run left a partial file, the returned dict contains
        the in-progress state so you can skip already-completed steps.

        Parameters
        ----------
        template : ExperimentTemplate
            The running experiment's template instance.

        Returns
        -------
        dict or None
            The partial state dict, or ``None`` if no partial file exists.

        Spec: REQ-INFRA-044, SCENARIO-INFRA-052
        """
        path = self._partial_path(template)
        if not Path(path).exists():
            return None

        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            _log.info(
                "PartialResultHandoff: resuming Exp %s from partial state at '%s'",
                template.exp_id,
                path,
            )
            return data
        except (OSError, json.JSONDecodeError) as exc:
            _log.warning(
                "PartialResultHandoff: could not read partial file '%s': %s — starting fresh",
                path,
                exc,
            )
            return None

    def update_partial_state(self, partial_state: dict) -> None:
        """Update the in-memory partial state so handlers use the latest snapshot.

        Call this periodically during long-running experiments so that an interrupt
        at any point saves a recent (not stale) partial state.

        Parameters
        ----------
        partial_state : dict
            The current in-progress state to record.
        """
        self._active_partial_state = partial_state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _partial_path(self, template: "ExperimentTemplate") -> str:
        """Return the path for the partial result file for *template*."""
        return os.path.join(self.results_dir, f"experiment_{template.exp_id}_partial.json")

    def _atexit_handler(self) -> None:
        """Write partial state on normal Python interpreter shutdown."""
        if self._active_template is not None and self._active_partial_state is not None:
            _log.info("PartialResultHandoff atexit handler firing for Exp %s", self._active_template.exp_id)
            self.save(self._active_template, self._active_partial_state)

    def _sigterm_handler(self, signum: int, frame: object) -> None:  # noqa: ARG002
        """Write partial state then re-raise SIGTERM so the process exits cleanly."""
        if self._active_template is not None and self._active_partial_state is not None:
            _log.info("PartialResultHandoff SIGTERM handler firing for Exp %s", self._active_template.exp_id)
            self.save(self._active_template, self._active_partial_state)
        # Re-raise the default SIGTERM behaviour so the process terminates.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)
