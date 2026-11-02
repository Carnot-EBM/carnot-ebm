"""Bound ARC generation dispatches before transport starts.

An episode can reach the generator through several policy branches. A limit in
the launcher does not stop those branches from dispatching. This module gives
them one optional, shared counter at the common inference boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
import threading
import time
import uuid
from typing import Any, Iterator


REQUEST_BUDGET_ATTR = "_carnot_episode_request_budget"
_REQUEST_CONTEXT: ContextVar[tuple[str, str | None]] = ContextVar(
    "carnot_arc_request_context", default=("generation", None)
)


class RequestBudgetError(RuntimeError):
    """Base error for a request that the episode budget does not authorize."""


class RequestBudgetExhausted(RequestBudgetError):
    """The episode has no unreserved generation slot."""


class EpisodeDeadlineExceeded(RequestBudgetError):
    """The one episode deadline passed before a new dispatch."""


class LateRequestCompletion(RequestBudgetError):
    """A response arrived after its episode closed and cannot be committed."""


class RequestAlreadyCompleted(RequestBudgetError):
    """A cold restart tried to dispatch an already completed request identity."""


@dataclass(frozen=True)
class RequestReservation:
    """A single reserved dispatch with terminal methods owned by its budget."""

    budget: EpisodeRequestBudget
    request_id: str

    def complete(self) -> None:
        """Accept the response only while the episode is still open."""

        self.budget._terminal(self.request_id, "completed")

    def fail(self, error: BaseException | str) -> None:
        """Keep a failed or timed-out dispatch as a consumed terminal slot."""

        self.budget._fail(self.request_id, error)


class EpisodeRequestBudget:
    """Reserve generation calls atomically for one policy episode.

    The lock protects the limit and terminal accounting together. Cancellation
    marks current calls terminal. Their later responses therefore fail before
    the inference boundary returns content to policy or memory code.
    """

    def __init__(
        self,
        episode_id: str,
        *,
        limit: int,
        deadline_s: float,
        clock_ns: Callable[[], float] = time.monotonic,
        prior_rows: Sequence[Mapping[str, Any]] = (),
        elapsed_before_restart_s: float = 0.0,
    ) -> None:
        if not str(episode_id).strip():
            raise ValueError("episode_id must be non-empty")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if not isinstance(deadline_s, (int, float)) or deadline_s <= 0:
            raise ValueError("deadline_s must be positive")
        self.episode_id = str(episode_id)
        self.limit = int(limit)
        self._clock = clock_ns
        self._started = float(self._clock())
        self._deadline = self._started + float(deadline_s)
        self._elapsed_before_restart_s = max(0.0, float(elapsed_before_restart_s))
        self._lock = threading.RLock()
        self._rows: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._closed = False
        self._cancel_reason: str | None = None
        self._deadline_exceeded = False
        self._late_completions_discarded = 0
        self._late_write_violations = 0
        self._deadline_violations = 0
        self._replayed_completed_refusals = 0
        self._load_prior_rows(prior_rows)

    @classmethod
    def from_receipt(
        cls,
        receipt: Mapping[str, Any],
        *,
        deadline_s: float,
        clock_ns: Callable[[], float] = time.monotonic,
    ) -> EpisodeRequestBudget:
        """Rebuild terminal identities without replaying completed transport."""

        rows = receipt.get("callback_rows")
        if not isinstance(rows, list):
            raise ValueError("restart receipt callback_rows must be a list")
        return cls(
            str(receipt.get("episode_id") or ""),
            limit=int(receipt.get("limit") or 0),
            deadline_s=deadline_s,
            clock_ns=clock_ns,
            prior_rows=[dict(row) for row in rows if isinstance(row, Mapping)],
            elapsed_before_restart_s=float(receipt.get("elapsed_s") or 0.0),
        )

    def _load_prior_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        if len(rows) > self.limit:
            raise ValueError("restart receipt exceeds request limit")
        for source in rows:
            row = deepcopy(dict(source))
            request_id = row.get("request_id")
            disposition = row.get("disposition")
            if (
                not isinstance(request_id, str)
                or not request_id
                or request_id in self._by_id
                or disposition not in {"completed", "failed", "cancelled"}
            ):
                raise ValueError("restart receipt contains a non-terminal or duplicate row")
            row["recovered_after_restart"] = True
            self._rows.append(row)
            self._by_id[request_id] = row

    def reserve(self, *, branch: str, request_id: str | None = None) -> RequestReservation:
        """Consume one slot before a caller can enter transport."""

        with self._lock:
            now = float(self._clock())
            identifier = request_id or f"request-{uuid.uuid4().hex}"
            previous = self._by_id.get(identifier)
            if previous is not None:
                if previous.get("disposition") == "completed":
                    self._replayed_completed_refusals += 1
                    raise RequestAlreadyCompleted(identifier)
                raise RequestBudgetExhausted(f"request identity already reserved: {identifier}")
            if now >= self._deadline:
                self._deadline_exceeded = True
                self._cancel_locked("episode_deadline", now)
                raise EpisodeDeadlineExceeded(self.episode_id)
            if self._closed:
                raise RequestBudgetExhausted(
                    f"episode closed: {self._cancel_reason or 'cancelled'}"
                )
            if len(self._rows) >= self.limit:
                raise RequestBudgetExhausted(
                    f"request budget exhausted: {len(self._rows)}/{self.limit}"
                )
            row = {
                "episode_id": self.episode_id,
                "request_id": identifier,
                "branch": str(branch or "generation"),
                "reservation_index": len(self._rows),
                "reserved_monotonic": now,
                "terminal_monotonic": None,
                "elapsed_s": 0.0,
                "disposition": "in_flight",
                "error": None,
                "cancel_reason": None,
                "recovered_after_restart": False,
            }
            self._rows.append(row)
            self._by_id[identifier] = row
            return RequestReservation(self, identifier)

    def _terminal(self, request_id: str, disposition: str) -> None:
        with self._lock:
            row = self._by_id[request_id]
            if row["disposition"] != "in_flight":
                if disposition == "completed" and row["disposition"] == "cancelled":
                    self._late_completions_discarded += 1
                    raise LateRequestCompletion(request_id)
                raise RequestBudgetError(f"request is already terminal: {request_id}")
            now = float(self._clock())
            # Closing holds this same lock and terminalizes every in-flight row,
            # so an in-flight row can reach here only through deadline expiry.
            if now >= self._deadline:
                self._deadline_exceeded = True
                self._cancel_locked("episode_deadline", now)
                self._late_completions_discarded += 1
                raise LateRequestCompletion(request_id)
            row["disposition"] = disposition
            row["terminal_monotonic"] = now
            row["elapsed_s"] = max(0.0, now - float(row["reserved_monotonic"]))

    def _fail(self, request_id: str, error: BaseException | str) -> None:
        with self._lock:
            row = self._by_id[request_id]
            if row["disposition"] != "in_flight":
                return
            now = float(self._clock())
            timed_out = isinstance(error, TimeoutError) or "timed out" in str(error).lower()
            if self._closed or now >= self._deadline or timed_out:
                reason = (
                    "request_timeout"
                    if timed_out
                    else "episode_deadline"
                    if now >= self._deadline
                    else self._cancel_reason or "episode_closed"
                )
                if now >= self._deadline:
                    self._deadline_exceeded = True
                self._cancel_row(row, reason, now)
            else:
                row["disposition"] = "failed"
                row["terminal_monotonic"] = now
                row["elapsed_s"] = max(0.0, now - float(row["reserved_monotonic"]))
                row["error"] = self._error_text(error)

    @staticmethod
    def _error_text(error: BaseException | str) -> str:
        if isinstance(error, BaseException):
            return f"{type(error).__name__}: {error}"[:500]
        return str(error)[:500]

    @staticmethod
    def _cancel_row(row: dict[str, Any], reason: str, now: float) -> None:
        if row["disposition"] != "in_flight":
            return
        row["disposition"] = "cancelled"
        row["terminal_monotonic"] = now
        row["elapsed_s"] = max(0.0, now - float(row["reserved_monotonic"]))
        row["cancel_reason"] = reason

    def _cancel_locked(self, reason: str, now: float) -> None:
        self._closed = True
        self._cancel_reason = str(reason)
        for row in self._rows:
            self._cancel_row(row, self._cancel_reason, now)

    def cancel(self, reason: str = "episode_cancelled") -> None:
        """Close the episode and give every active dispatch one terminal state."""

        with self._lock:
            self._cancel_locked(reason, float(self._clock()))

    def receipt(self) -> dict[str, Any]:
        """Return a detached accounting snapshot for checkpoints and artifacts."""

        with self._lock:
            now = float(self._clock())
            counts = {
                name: sum(row["disposition"] == name for row in self._rows)
                for name in ("completed", "failed", "cancelled", "in_flight")
            }
            attempted = len(self._rows)
            return {
                "schema": "carnot.arc_episode_request_budget.v1",
                "episode_id": self.episode_id,
                "limit": self.limit,
                "attempted": attempted,
                **counts,
                "remaining": self.limit - attempted,
                "accounting_valid": attempted == sum(counts.values()),
                "closed": self._closed,
                "cancel_reason": self._cancel_reason,
                "deadline_exceeded": self._deadline_exceeded,
                "deadline_violations": self._deadline_violations,
                "late_completions_discarded": self._late_completions_discarded,
                "late_write_violations": self._late_write_violations,
                "replayed_completed_refusals": self._replayed_completed_refusals,
                "elapsed_s": self._elapsed_before_restart_s + max(0.0, now - self._started),
                "callback_rows": deepcopy(self._rows),
            }


def attach_request_budget(proposer: Any, budget: EpisodeRequestBudget) -> None:
    """Opt one proposer into a shared episode budget without changing defaults."""

    setattr(proposer, REQUEST_BUDGET_ATTR, budget)


@contextmanager
def request_budget_scope(branch: str, *, request_id: str | None = None) -> Iterator[None]:
    """Label one live callback without storing thread-local state on the proposer."""

    token = _REQUEST_CONTEXT.set((str(branch), request_id))
    try:
        yield
    finally:
        _REQUEST_CONTEXT.reset(token)


def reserve_for_proposer(proposer: Any, operation: str) -> RequestReservation | None:
    """Reserve only generation work for proposers that explicitly opt in."""

    budget = getattr(proposer, REQUEST_BUDGET_ATTR, None)
    if operation != "generation" or not isinstance(budget, EpisodeRequestBudget):
        return None
    branch, request_id = _REQUEST_CONTEXT.get()
    return budget.reserve(branch=branch, request_id=request_id)
