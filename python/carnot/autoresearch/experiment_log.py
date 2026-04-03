"""Experiment logging for the autoresearch pipeline.

**Researcher summary:**
    Structured JSON experiment log recording hypothesis lifecycle:
    proposed → sandbox result → evaluation verdict → outcome.
    Append-only for audit trail. Supports querying rejected hypotheses
    to prevent re-proposal.

**Detailed explanation for engineers:**
    Every time the autoresearch loop evaluates a hypothesis, the full
    record is appended to an experiment log. This serves three purposes:

    1. **Audit trail**: You can review every hypothesis that was tried,
       why it was accepted or rejected, and what metrics it produced.
       This is essential for debugging when the loop isn't finding
       improvements — you can see what it already tried.

    2. **Rejected registry**: Hypotheses that failed are recorded so
       the proposal stage can avoid re-proposing the same thing.
       Without this, the loop would waste time retrying known failures.

    3. **Progress tracking**: By looking at accepted hypotheses over
       time, you can see the trajectory of improvement and identify
       when the loop has plateaued.

    The log is stored as a JSON file with one entry per hypothesis.
    Entries are append-only — old entries are never modified or deleted.

Spec: REQ-AUTO-008, REQ-AUTO-007
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ExperimentEntry:
    """A single experiment record in the autoresearch log.

    **Researcher summary:**
        Full lifecycle record: hypothesis spec, sandbox metrics, evaluation
        verdict, and final outcome (accepted/rejected/review).

    **Detailed explanation for engineers:**
        This captures everything about one hypothesis evaluation cycle:

        - ``id``: Unique identifier (e.g., "auto-2026-04-03-001")
        - ``timestamp``: When the experiment ran (ISO 8601)
        - ``hypothesis_code``: The full Python source code of the hypothesis
        - ``hypothesis_description``: Human-readable description of what was tried
        - ``sandbox_success``: Did the code run without errors?
        - ``sandbox_metrics``: The metrics dict returned by the hypothesis
        - ``sandbox_error``: Error message if the sandbox failed
        - ``sandbox_wall_clock``: How long the hypothesis took to run
        - ``eval_verdict``: PASS, FAIL, or REVIEW
        - ``eval_reason``: Why it got that verdict
        - ``eval_improvements``: Which benchmarks improved
        - ``eval_regressions``: Which benchmarks regressed
        - ``outcome``: Final status — "accepted", "rejected", or "pending_review"

    Spec: REQ-AUTO-008
    """

    id: str
    timestamp: str
    hypothesis_code: str
    hypothesis_description: str = ""
    sandbox_success: bool = False
    sandbox_metrics: dict[str, Any] = field(default_factory=dict)
    sandbox_error: str | None = None
    sandbox_wall_clock: float = 0.0
    sandbox_timed_out: bool = False
    eval_verdict: str = ""
    eval_reason: str = ""
    eval_improvements: list[str] = field(default_factory=list)
    eval_regressions: list[str] = field(default_factory=list)
    outcome: str = ""  # "accepted", "rejected", "pending_review"


class ExperimentLog:
    """Append-only experiment log for the autoresearch pipeline.

    **Researcher summary:**
        JSON-backed log of all hypothesis evaluations. Supports append,
        query rejected IDs, and count consecutive failures for circuit breaker.

    **Detailed explanation for engineers:**
        The experiment log is a list of ExperimentEntry records stored as
        a JSON array on disk. Key operations:

        - ``append(entry)``: Add a new experiment record
        - ``rejected_ids()``: Get IDs of all rejected hypotheses (for dedup)
        - ``consecutive_failures()``: Count recent failures (for circuit breaker)
        - ``save()`` / ``load()``: Persist to / read from disk

        The circuit breaker (REQ-AUTO-009) uses ``consecutive_failures()``
        to detect when the loop is stuck and should halt for human review.

    For example::

        log = ExperimentLog()
        log.append(ExperimentEntry(
            id="auto-001",
            timestamp="2026-04-03T00:00:00Z",
            hypothesis_code="def run(d): return {'final_energy': -5.0}",
            eval_verdict="PASS",
            outcome="accepted",
        ))
        log.save(Path("experiments.json"))

    Spec: REQ-AUTO-008
    """

    def __init__(self) -> None:
        self.entries: list[ExperimentEntry] = []

    def append(self, entry: ExperimentEntry) -> None:
        """Add an experiment record to the log.

        Spec: REQ-AUTO-008
        """
        self.entries.append(entry)

    def rejected_ids(self) -> set[str]:
        """Return IDs of all rejected hypotheses.

        Used by the proposal stage to avoid re-proposing known failures.

        Spec: REQ-AUTO-007
        """
        return {e.id for e in self.entries if e.outcome == "rejected"}

    def consecutive_failures(self) -> int:
        """Count consecutive failures from the end of the log.

        The circuit breaker (REQ-AUTO-009) halts the loop when this count
        exceeds a configured threshold. "Failure" means the hypothesis was
        rejected (not pending_review or accepted).

        Returns 0 if the log is empty or the last entry was not a failure.

        Spec: REQ-AUTO-009
        """
        count = 0
        # Walk backwards from the most recent entry
        for entry in reversed(self.entries):
            if entry.outcome == "rejected":
                count += 1
            else:
                break  # streak broken by a non-rejection
        return count

    def save(self, path: Path) -> None:
        """Save the experiment log to a JSON file.

        Spec: REQ-AUTO-008
        """
        data = [asdict(e) for e in self.entries]
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> ExperimentLog:
        """Load an experiment log from a JSON file.

        Spec: REQ-AUTO-008
        """
        log = cls()
        if path.exists():
            data = json.loads(path.read_text())
            for entry_dict in data:
                log.entries.append(ExperimentEntry(**entry_dict))
        return log

    def __len__(self) -> int:
        return len(self.entries)
