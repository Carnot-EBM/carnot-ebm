"""REQ-REPORT-8677: validate every callback before delivering one action result."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from carnot.agentic.arc_outcome_observer import visible
from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]


def payload(frame: Any) -> Json | None:
    """Include public pixels so unchanged level metadata cannot conceal a conflict."""
    result: Json | None = visible(frame)
    if result is not None and not isinstance(frame, dict):
        pixels: Any = getattr(frame, "frame", None)
        result["pixels"] = None if pixels is None else np.asarray(pixels).tolist()
        result["available_actions"] = list(getattr(frame, "available_actions", []))
        result["win_levels"] = getattr(frame, "win_levels", None)
    return result


def deliver(policy: Any, frame: Any, owner: str) -> None:
    """The first callback owns delivery; every later callback still checks its bytes.

    The ledger belongs to the installed observer, so reinstalling an observer
    starts a new episode. Action indices distinguish successive identical frames.
    Original observer validation remains available to direct callers unchanged.
    """
    observer = getattr(policy, "_arc_outcome_observer", None)
    if observer is None or observer.action_index == 0:
        return
    try:
        _deliver(observer, frame, owner)
    except Exception as exc:
        observer.issues.append(dict(reason="observer_error", error=type(exc).__name__))
        observer.finish("observer_error")


def _deliver(observer: Any, frame: Any, owner: str) -> None:
    """Keep receipt failures separate from the policy's original action path."""
    if not hasattr(observer, "delivery_rows"):
        observer.delivery_rows = []
        observer.delivery_seen = {}
    index = observer.action_index
    digest = canonical_hash(payload(frame))
    previous = observer.delivery_seen.get(index)
    status = (
        "canonical"
        if previous is None
        else "duplicate_callback"
        if previous == digest
        else "conflicting_duplicate"
    )
    observer.delivery_rows.append(
        dict(
            action_index=index,
            action=deepcopy(observer._actions[index]),
            owner=owner,
            observation_sha256=digest,
            status=status,
        )
    )
    if previous is None:
        observer.delivery_seen[index] = digest
        observer.observe(index, frame)
    elif previous != digest:
        observer.issues.append(dict(action_index=index, reason="conflicting_duplicate"))
        observer.finish("conflicting_duplicate")
