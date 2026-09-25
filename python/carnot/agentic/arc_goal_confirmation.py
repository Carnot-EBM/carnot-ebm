"""REQ-REPORT-7666: compare an executed model goal with independent SDK progress."""

from __future__ import annotations

from typing import Any


def shadow_goal_only_decision(predicted_goal: bool, sdk_status: str) -> dict[str, str]:
    """Compare old and guarded decisions from one executed endpoint observation.

    The shadow is a decision comparison. It does not replay a second episode or
    turn a public-game action trace into a counterfactual solve rate.
    """
    old = "accept_goal" if predicted_goal else "no_assertion"
    guarded = {
        "confirmed": "accept_goal",
        "contradiction": "reject_goal" if predicted_goal else "defer",
    }.get(sdk_status, "defer")
    return {"old_goal_only": old, "observed_guard": guarded}


class GoalConfirmation:
    """Keep one pending endpoint and bounded, serializable observation receipts."""

    def __init__(self) -> None:
        self.pending: dict[str, Any] | None = None
        self.receipts: list[dict[str, Any]] = []

    def arm(
        self,
        frame: Any,
        *,
        frames_seen: int,
        level: int,
        predicted_goal: bool,
        plan_length: int,
    ) -> None:
        """Record the actual executed endpoint, not an in-model search node."""
        self.pending = {
            "frame_id": id(frame),
            "frames_seen": frames_seen,
            "level_before": level,
            "predicted_goal": predicted_goal,
            "plan_length": plan_length,
        }

    def observe(self, frame: Any, *, frames_seen: int) -> dict[str, Any]:
        """Read only SDK level and terminal state; unresolved signals remain unknown."""
        pending = self.pending
        if pending is None:
            return {"status": "unknown", "reason": "no_executed_endpoint", "contradiction": False}
        state = str(getattr(frame, "state", "") or "").upper()
        level_raw = getattr(frame, "levels_completed", None)
        try:
            level = int(level_raw) if level_raw is not None else None
        except (TypeError, ValueError):
            level = None
        layers = getattr(frame, "frame", None)
        settling = isinstance(layers, (list, tuple)) and len(layers) > 1
        fresh = (
            frame is not None
            and frames_seen > pending["frames_seen"]
            and id(frame) != pending["frame_id"]
        )
        reason = ""
        status = "unknown"
        if not fresh:
            reason = "stale_frame"
        elif level is None:
            reason = "unknown_level"
        elif level > pending["level_before"] or state in {"WIN", "SUCCESS"}:
            status = "confirmed"
            reason = "sdk_progress"
        elif settling:
            reason = "animation_unsettled"
        elif state in {"NOT_FINISHED", "PLAYING", "GAME_OVER", "LOSE"}:
            status = "contradiction" if pending["predicted_goal"] else "unknown"
            reason = "sdk_no_progress" if pending["predicted_goal"] else "no_goal_assertion"
        else:
            reason = "unknown_terminal_state"
        receipt = {
            "status": status,
            "reason": reason,
            "predicted_goal": pending["predicted_goal"],
            "sdk_level": level,
            "sdk_state": state,
            "level_before": pending["level_before"],
            "frames_seen": frames_seen,
            "settling": settling,
            "contradiction": status == "contradiction",
            "plan_length": pending["plan_length"],
        }
        self.receipts.append(receipt)
        if status != "unknown":
            self.pending = None
        return receipt

    def timeout(self) -> dict[str, Any]:
        """Close an unresolved endpoint without turning absence into a negative label."""
        self.pending = None
        receipt = {"status": "unknown", "reason": "timeout", "contradiction": False}
        self.receipts.append(receipt)
        return receipt
