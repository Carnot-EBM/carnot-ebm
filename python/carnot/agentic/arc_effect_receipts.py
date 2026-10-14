"""REQ-REPORT-8522: observe existing mutations without choosing policy actions.

Only visible pixels and the fields changed by the curated arms enter receipts.
Invalid state refuses evidence while the existing policy continues unchanged.
"""

from __future__ import annotations

from dataclasses import asdict
import os
from typing import Any

import numpy as np

from carnot.reporting.current_work_receipt import canonical_hash

Json = dict[str, Any]
FLAG = "CARNOT_ARC_EFFECT_RECEIPTS"
SCHEMA = "carnot.arc.direct_supervisor_effect.v1"


def relevant_state(policy: Any) -> Json:
    """Refuse missing operands rather than convert unknown state into false."""
    try:
        explorer = policy.explorer
        value = dict(
            goal_bias_present=explorer.goal_bias is not None,
            induced=policy.induced,
            hybrid_diversity=explorer._hybrid_diversity,
            steps_since_progress=explorer._steps_since_progress,
            stall_threshold=explorer._stall_threshold,
        )
        if any(type(value[k]) is not bool for k in ["induced", "hybrid_diversity"]):
            raise ValueError("state_boolean")
        if any(
            type(value[k]) is not int or value[k] < 0
            for k in ["steps_since_progress", "stall_threshold"]
        ):
            raise ValueError("state_counter")
        return value
    except (AttributeError, TypeError) as exc:
        raise ValueError("state_missing") from exc


def action_policy(state: Json) -> Json:
    """A latch change alone does not establish changed action selection."""
    return dict(
        goal_bias_present=state["goal_bias_present"],
        diversity_active=state["hybrid_diversity"]
        and state["steps_since_progress"] > state["stall_threshold"],
    )


def begin(policy: Any, latest: Any) -> Json | None:
    """The exact opt-in keeps normal runs free of receipt state and pixel copies."""
    if os.environ.get(FLAG) != "1":
        return None
    try:
        visible = dict(
            pixels=np.asarray(latest.frame).tolist(),
            level=int(latest.levels_completed),
            available_actions=list(latest.available_actions),
        )
        state = relevant_state(policy)
        return dict(
            before=state,
            observation=visible,
            induction_attempt_start_index=len(policy.induction_attempts),
            error=None,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        return dict(before=None, observation=None, error=type(exc).__name__ + ":" + str(exc))


def finish(
    policy: Any,
    token: Json | None,
    snapshot: Any,
    predicates: list[Json],
    redirect: Any,
    mode: str,
    action_index: int,
    error: str | None,
) -> None:
    """Record after the actual application; errors never become effect evidence."""
    if token is None:
        return
    before = token["before"]
    try:
        after = relevant_state(policy)
    except ValueError as exc:
        after = None
        error = str(exc)
    refused = bool(error or token["error"] or before is None or after is None)
    selected = redirect is not None
    changed = not refused and before != after
    disposition = (
        "state_refused"
        if refused
        else "not_selected"
        if not selected
        else "selected_shadow"
        if mode == "shadow"
        else "state_changed"
        if changed
        else "already_active_no_op"
    )
    row = dict(
        schema=SCHEMA,
        event_id=f"{policy.short}:{action_index}",
        action_index=action_index,
        wrapper_action_index=getattr(policy, "_effect_action_index", 0) + 1,
        observation=token["observation"],
        observation_sha256=canonical_hash(token["observation"]),
        snapshot=asdict(snapshot) if snapshot is not None else None,
        eligibility_predicates=predicates,
        mode=mode,
        arm=redirect.arm if selected else None,
        selected=selected,
        applied=bool(changed and selected and mode == "applied"),
        application_disposition=disposition,
        before_state=before,
        after_state=after,
        before_state_sha256=canonical_hash(before) if before is not None else None,
        after_state_sha256=canonical_hash(after) if after is not None else None,
        effective_action_policy_change=bool(
            changed and action_policy(before) != action_policy(after)
        ),
        induction_attempt_start_index=token.get("induction_attempt_start_index"),
        induction_attempt=None,
        induction_outcome="not_observed",
        model_action_executed=False,
        observation_error=error or token["error"],
        action=None,
    )
    if not hasattr(policy, "_trajectory_effect_receipts"):
        policy._trajectory_effect_receipts = []
    policy._trajectory_effect_receipts.append(row)


def bind_action(policy: Any, kind: Any, data: Any) -> None:
    """Join receipts to the wrapper's returned action and same-step induction outcome."""
    if os.environ.get(FLAG) != "1":
        return
    policy._effect_action_index = getattr(policy, "_effect_action_index", 0) + 1
    for row in getattr(policy, "_trajectory_effect_receipts", []):
        if row["wrapper_action_index"] == policy._effect_action_index:
            row["action"] = dict(kind=kind, data=data)
        start = row["induction_attempt_start_index"]
        requested = row["arm"] == "allow_reinduction" and row["mode"] == "applied"
        if (
            requested
            and row["induction_attempt"] is None
            and start is not None
            and len(policy.induction_attempts) > start
        ):
            attempt = policy.induction_attempts[start]
            row["induction_attempt"] = {
                k: attempt.get(k)
                for k in ["reason", "goal_level", "transition_count", "planned", "skipped"]
            }
            row["induction_outcome"] = (
                "model_disabled"
                if attempt.get("skipped") == "disabled_by_env"
                else "attempt_observed"
            )
            row["induction_attempt_wrapper_action_index"] = policy._effect_action_index
            row["model_action_executed"] = (
                False if attempt.get("skipped") == "disabled_by_env" else None
            )


def read(policy: Any) -> list[Json]:
    """Return a detached JSON value so downstream readers cannot mutate the policy."""
    from copy import deepcopy

    return list(deepcopy(getattr(policy, "_trajectory_effect_receipts", [])))
