"""REQ-REPORT-8649: record observed request sites without changing decisions.

A reset's pixels cannot explain who requested it. These receipts bind the request
before dispatch and keep ambiguous ownership visible instead of inventing intent.
"""

from __future__ import annotations

from copy import deepcopy
from enum import Enum
from threading import RLock
import time
import traceback
from typing import Any, Callable

import numpy as np

from carnot.agentic.arc_e3_outcome_transport import normalize_action
from carnot.agentic.arc_outcome_observer import visible
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]


def public_value(value: Any) -> Any:
    """REQ-REPORT-8663: preserve nested public SDK values in their JSON form.

    Raw SDK frames contain a list of arrays, rather than one outer array.
    Unknown objects still raise during hashing; no string fallback hides a loss.
    """
    if isinstance(value, Enum):
        return public_value(value.value)
    if isinstance(value, (np.ndarray, np.generic)):
        return public_value(value.tolist())
    if isinstance(value, dict):
        return {key: public_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_value(item) for item in value]
    return value


def public_hash(frame: Any) -> str:
    """Hash public pixels too, because equal level counters can hide different states."""
    value = visible(frame)
    pixels = getattr(frame, "frame", None)
    if value is not None and pixels is not None:
        value["pixels"] = pixels
    return str(canonical_hash(public_value(value)))


class ResetProvenance:
    """Bound request storage independently of the agent's unchanged action budget."""

    def __init__(self, limit: int = 512) -> None:
        self.limit = limit
        self.index = 0
        self.context: Json = {}
        self.requests: list[Json] = []
        self.terminals: list[Json] = []
        self.issues: list[Json] = []
        self.callbacks: list[Json] = []
        self._lock = RLock()
        self._callback_counts: dict[str, int] = {}

    def begin(
        self,
        observation: Any,
        remaining_actions: int | None = None,
        remaining_seconds: float | None = None,
    ) -> None:
        """Start at choose_action, so two requesters share the same action identity."""
        with self._lock:
            self.index += 1
            self._callback_counts = {}
            self.context = dict(
                action_index=self.index,
                pre_observation_sha256=None,
                remaining_actions=remaining_actions,
                remaining_time_s=remaining_seconds,
                context_error=None,
            )
            try:
                self.context["pre_observation_sha256"] = public_hash(observation)
            except Exception as exc:
                self.context["context_error"] = type(exc).__name__
                raise

    def request(self, branch: str, origin: str = "agent_reset") -> None:
        """Only an executing call site supplies a cause; discarded requests stay visible."""
        with self._lock:
            self._request(branch, origin)

    def _request(self, branch: str, origin: str) -> None:
        """Bind the request while begin cannot replace its action context."""
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
        if context.get("context_error"):
            origin = "unknown"
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

    def _callback(self, method: str, owner: str) -> bool:
        """A callback identity uses its action and owner, never equal frame pixels."""
        key = method + ":" + owner
        count = self._callback_counts.get(key, 0) + 1
        self._callback_counts[key] = count
        if len(self.callbacks) < self.limit:
            self.callbacks.append(
                dict(
                    action_index=self.index,
                    method=method,
                    owner=owner,
                    invocation=count,
                    duplicate_invocation=count > 1,
                )
            )
        return count > 1

    def selected(self, action: Any, owner: str = "policy_selection") -> None:
        """Uninstrumented requesters remain unknown rather than borrowing another cause."""
        self._callback("selected", owner)
        if normalize_action(action)["kind"] == "RESET" and not self._current():
            self.request("missing_request_provenance", "unknown")

    def _current(self) -> list[Json]:
        return [r for r in self.requests if r["action_index"] == self.index]

    def returned(self, action: Any, observation: Any, error: str | None = None) -> None:
        """SDK terminal status is an observation, separate from an agent reset request."""
        if self._callback("returned", "sdk_return"):
            return
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
                callback_sequence_rows=self.callbacks,
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
        site = traceback.extract_tb(exc.__traceback__)[-1]
        diagnostics = getattr(policy, "_arc_reset_provenance_diagnostics", [])
        policy._arc_reset_provenance_diagnostics = (
            diagnostics
            + [
                dict(
                    method=method,
                    error=type(exc).__name__,
                    message=str(exc)[:240],
                    site=dict(path=site.filename, line=site.lineno, function=site.name),
                    value_types=[
                        type(value).__module__ + "." + type(value).__qualname__ for value in args
                    ],
                )
            ]
        )[-16:]


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
    notify(policy, "selected", action, "sdk_dispatch")
    try:
        result = call()
    except Exception as exc:
        notify(policy, "returned", action, None, type(exc).__name__)
        raise
    notify(policy, "returned", action, result)
    return result
