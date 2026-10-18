"""REQ-REPORT-8649: record observed request sites without changing decisions.

A reset's pixels cannot explain who requested it. These receipts bind the request
before dispatch and keep ambiguous ownership visible instead of inventing intent.
"""

from __future__ import annotations

from copy import deepcopy
import time
from typing import Any, Callable

from carnot.agentic.arc_e3_outcome_transport import normalize_action
from carnot.agentic.arc_outcome_observer import visible
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]


def public_hash(frame: Any) -> str:
    """Hash public pixels too, because equal level counters can hide different states."""
    value = visible(frame)
    pixels = getattr(frame, "frame", None)
    if value is not None and pixels is not None:
        value["pixels"] = pixels.tolist() if hasattr(pixels, "tolist") else pixels
    return str(canonical_hash(value))


class ResetProvenance:
    """Bound request storage independently of the agent's unchanged action budget."""

    def __init__(self, limit: int = 512) -> None:
        self.limit = limit
        self.index = 0
        self.context: Json = {}
        self.requests: list[Json] = []
        self.terminals: list[Json] = []
        self.issues: list[Json] = []

    def begin(
        self,
        observation: Any,
        remaining_actions: int | None = None,
        remaining_seconds: float | None = None,
    ) -> None:
        """Start at choose_action, so two requesters share the same action identity."""
        self.index += 1
        self.context = dict(
            action_index=self.index,
            pre_observation_sha256=public_hash(observation),
            remaining_actions=remaining_actions,
            remaining_time_s=remaining_seconds,
        )

    def request(self, branch: str, origin: str = "agent_reset") -> None:
        """Only an executing call site supplies a cause; discarded requests stay visible."""
        if len(self.requests) >= self.limit:
            if not self.issues:
                self.issues.append(dict(reason="record_limit"))
            return
        context = dict(
            action_index=self.index,
            pre_observation_sha256=public_hash(None),
            remaining_actions=None,
            remaining_time_s=None,
        )
        context.update(self.context)
        self.requests.append(
            dict(
                context,
                reset_request_id=f"{self.index}:{len(self.requests) + 1}",
                origin=origin,
                initiating_policy_branch=branch,
                post_observation_sha256=None,
                sdk_status=None,
            )
        )

    def selected(self, action: Any) -> None:
        """Uninstrumented requesters remain unknown rather than borrowing another cause."""
        if normalize_action(action)["kind"] == "RESET" and not self._current():
            self.request("missing_request_provenance", "unknown")

    def _current(self) -> list[Json]:
        return [r for r in self.requests if r["action_index"] == self.index]

    def returned(self, action: Any, observation: Any, error: str | None = None) -> None:
        """SDK terminal status is an observation, separate from an agent reset request."""
        value = visible(observation)
        state = str((value or {}).get("state", ""))
        status = dict(
            state=state,
            full_reset=(value or {}).get("full_reset"),
            observation_missing=observation is None,
            error=error,
        )
        if normalize_action(action)["kind"] == "RESET":
            for row in self._current():
                row.update(post_observation_sha256=public_hash(observation), sdk_status=status)
        if state in {"WIN", "GAME_OVER", "GameState.WIN", "GameState.GAME_OVER"}:
            if len(self.terminals) < self.limit:
                self.terminals.append(
                    dict(
                        action_index=self.index,
                        origin="environment_terminal",
                        sdk_status=status,
                        observation_sha256=public_hash(observation),
                    )
                )

    def receipt(self) -> Json:
        """Copies prevent a reader from rewriting the passive ownership ledger."""
        ownership = []
        for index in dict.fromkeys(r["action_index"] for r in self.requests):
            rows = [r for r in self.requests if r["action_index"] == index]
            owners = {(r["origin"], r["initiating_policy_branch"]) for r in rows}
            status = (
                "conflicting" if len(owners) > 1 else "duplicate" if len(rows) > 1 else "unique"
            )
            if any(r["origin"] == "unknown" for r in rows):
                status = "unknown"
            ownership.append(
                dict(
                    action_index=index,
                    status=status,
                    reset_request_ids=[r["reset_request_id"] for r in rows],
                )
            )
        return deepcopy(
            dict(
                reset_request_rows=self.requests,
                requester_ownership_rows=ownership,
                terminal_reason_rows=self.terminals,
                missing_cause_rows=[r for r in self.requests if r["origin"] == "unknown"],
                issues=self.issues,
            )
        )


def notify(policy: Any, method: str, *args: Any) -> None:
    """Recorder exceptions must not turn an unchanged policy into a failed game."""
    recorder = getattr(policy, "_arc_reset_provenance", None)
    if recorder is None:
        return
    try:
        getattr(recorder, method)(*args)
    except Exception as exc:
        errors = getattr(policy, "_arc_reset_provenance_errors", [])
        policy._arc_reset_provenance_errors = (errors + [type(exc).__name__])[-16:]


def request(policy: Any, branch: str, origin: str = "agent_reset") -> None:
    """A constant branch label reports only the site that actually executed."""
    notify(policy, "request", branch, origin)


def prepare(policy: Any, observation: Any) -> None:
    """Unknown budget fields remain null when the caller has no authoritative limit."""
    recorder = getattr(policy, "_arc_reset_provenance", None)
    maximum = getattr(policy, "_arc_reset_max_actions", None)
    remaining = None if maximum is None else max(0, maximum - getattr(recorder, "index", 0))
    deadline = getattr(policy, "_arc_reset_deadline", None)
    seconds = None if deadline is None else max(0.0, deadline - time.monotonic())
    notify(policy, "begin", observation, remaining, seconds)


def sdk_call(policy: Any, action: Any, call: Callable[[], Any]) -> Any:
    """Observe the existing SDK call once and preserve its return or original exception."""
    notify(policy, "selected", action)
    try:
        result = call()
    except Exception as exc:
        notify(policy, "returned", action, None, type(exc).__name__)
        raise
    notify(policy, "returned", action, result)
    return result
