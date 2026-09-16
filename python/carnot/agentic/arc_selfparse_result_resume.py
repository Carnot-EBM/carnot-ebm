"""Bound one selfparse tool result to one immediate resume request.

The guard is transport bookkeeping. It never accepts an engine or authorizes a
policy action. The caller must still run the existing verifier and policy gates.

Spec refs: REQ-ARC-WMTE-7336 and SCENARIO-ARC-WMTE-7336-*.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def attempt_identity(game: str, cell: int, prompt: str) -> str:
    """Make a stable attempt ID from inputs that already define one induction."""

    return _canonical_hash({"game": game, "cell": int(cell), "prompt": prompt})


@dataclass
class ResultResumeGuard:
    """Retain a result until one bound request receives it, then erase it."""

    episode_id: str
    attempt_id: str
    completion_limit: int
    authority_expires_monotonic: float
    reserved_completion_slots: int = 1
    result_rows: list[JsonDict] = field(default_factory=list)
    request_rows: list[JsonDict] = field(default_factory=list)
    rejections: list[JsonDict] = field(default_factory=list)
    _pending: JsonDict | None = None
    _seen_result_ids: set[str] = field(default_factory=set)

    def _reject(self, reason: str, **detail: Any) -> JsonDict:
        row = {"accepted": False, "reason": reason, "plan_authorized": False, **detail}
        self.rejections.append(row)
        return row

    def reject(self, reason: str, **detail: Any) -> JsonDict:
        """Record a caller-side rejection without exposing mutable guard state."""

        return self._reject(reason, **detail)

    def normal_request_allowed(self, *, completed_calls: int) -> bool:
        """Keep the final completion unused until a runtime result earns it."""

        normal_limit = max(0, int(self.completion_limit) - self.reserved_completion_slots)
        return self._pending is None and int(completed_calls) < normal_limit

    def offer_result(
        self,
        *,
        source_request_id: str,
        next_request_id: str,
        tool_names: Sequence[str],
        bounded_response: str,
        dispatch_results: Sequence[Mapping[str, Any]],
    ) -> JsonDict:
        """Bind actual dispatch output; an empty fixture cannot create authority."""

        identity_input = {
            "episode_id": self.episode_id,
            "attempt_id": self.attempt_id,
            "source_request_id": source_request_id,
            "next_request_id": next_request_id,
            "tool_names": list(tool_names),
            "bounded_response": bounded_response,
            "dispatch_results": [dict(row) for row in dispatch_results],
        }
        result_id = _canonical_hash(identity_input)
        if not bounded_response or not dispatch_results:
            return self._reject(
                "absent_result", source_request_id=source_request_id, result_id=result_id
            )
        if any(row.get("ok") is not True for row in dispatch_results):
            return self._reject(
                "unsuccessful_result", source_request_id=source_request_id, result_id=result_id
            )
        if result_id in self._seen_result_ids or self._pending is not None:
            return self._reject(
                "duplicate_result", source_request_id=source_request_id, result_id=result_id
            )
        row: JsonDict = {
            **identity_input,
            "result_id": result_id,
            "state": "pending",
            "delivery_count": 0,
            "receipt_captured": False,
            "can_authorize_plan": False,
        }
        self._seen_result_ids.add(result_id)
        self.result_rows.append(row)
        self._pending = row
        return {"accepted": True, "reason": "pending", "result_id": result_id}

    def prepare_next_request(
        self,
        *,
        request_id: str,
        episode_id: str,
        attempt_id: str,
        now_monotonic: float,
    ) -> str | None:
        """Return the payload once only when all bindings still match."""

        request: JsonDict = {
            "request_id": request_id,
            "episode_id": episode_id,
            "attempt_id": attempt_id,
            "result_id": self._pending.get("result_id") if self._pending else None,
            "dispatched": False,
            "response_received": False,
            "receipt_captured": False,
        }
        self.request_rows.append(request)
        if self._pending is None:
            self._reject("absent_result", request_id=request_id)
            return None
        if self._pending["state"] != "pending":
            self._reject("duplicate_result_delivery", request_id=request_id)
            return None
        if episode_id != self.episode_id:
            self._pending["state"] = "rejected"
            self._reject("stale_episode", request_id=request_id)
            return None
        if attempt_id != self.attempt_id:
            self._pending["state"] = "rejected"
            self._reject("mismatched_attempt", request_id=request_id)
            return None
        if request_id != self._pending["next_request_id"]:
            self._pending["state"] = "rejected"
            self._reject("stale_request", request_id=request_id)
            return None
        if float(now_monotonic) >= float(self.authority_expires_monotonic):
            self._pending["state"] = "rejected"
            self._reject("expired_authority", request_id=request_id)
            return None
        self._pending["state"] = "dispatched"
        self._pending["delivery_count"] = 1
        request["dispatched"] = True
        return str(self._pending["bounded_response"])

    def complete_request(
        self, *, request_id: str, response_received: bool, timed_out: bool
    ) -> bool:
        """Capture receipt only after the bound request returns successfully."""

        request = next(
            (
                row
                for row in reversed(self.request_rows)
                if row["request_id"] == request_id and row["dispatched"]
            ),
            None,
        )
        if request is None or self._pending is None:
            self._reject("absent_result", request_id=request_id)
            return False
        if timed_out:
            self._pending["state"] = "rejected"
            self._reject("timeout_after_dispatch", request_id=request_id)
            self._pending = None
            return False
        if not response_received:
            self._pending["state"] = "rejected"
            self._reject("response_not_received", request_id=request_id)
            self._pending = None
            return False
        request["response_received"] = True
        request["receipt_captured"] = True
        self._pending["state"] = "receipt_captured"
        self._pending["receipt_captured"] = True
        self._pending = None
        return True

    @property
    def has_pending_result(self) -> bool:
        return self._pending is not None and self._pending.get("state") == "pending"

    def receipt(self) -> JsonDict:
        """Return immutable evidence for artifact reduction and diagnostics."""

        return {
            "enabled": True,
            "episode_id": self.episode_id,
            "attempt_id": self.attempt_id,
            "completion_limit": int(self.completion_limit),
            "reserved_completion_slots": self.reserved_completion_slots,
            "result_rows": deepcopy(self.result_rows),
            "request_rows": deepcopy(self.request_rows),
            "rejections": deepcopy(self.rejections),
            "pending_result": self.has_pending_result,
            "plan_authorized": False,
        }
